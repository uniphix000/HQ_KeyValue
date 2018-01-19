#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datautils import use_cuda, flatten



class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout, lang, max_utterance_num):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.V = lang.word_size
        self.embedding = nn.Embedding(self.V, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
        self.seq_lstm_h = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.seq_lstm_c = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.cos = nn.CosineSimilarity(dim=0)
        self.softmax = nn.Softmax()
        self.max_utterance_num = max_utterance_num
        self.linear_h = nn.Linear((max_utterance_num-1) * self.hidden_size, self.hidden_size)
        self.linear_c = nn.Linear((max_utterance_num-1) * self.hidden_size, self.hidden_size)

    def forward(self, batch_input, sentences_lens, keys, pad_idx, batch_size, n, lst):
        '''
            将句子一起过encoder，并sum
        :param batch_input: ((n-1)*b_s, m_l)
        :param keys: [(,),...] 每个key已经被idx表示了,而keys里的每个元素都是词典中前2kv个word中的某一个, (kv, 2, 1)
        :param
        :return:  h_c: (1, b_s, 2*h_s)
        '''
        # embed_key = self.embedding(keys.squeeze(2))
        # k = torch.sum(embed_key, 1)  # (kv, embed_size)

        lst_reverse = sorted(lst, key = lambda d: lst[d])

        # 计算weight
        embed = self.embedding(batch_input)  # ((n-1)*b_s, m_l, e_s)
        sentence_embed = torch.sum(embed, 1)  # ((n-1)*b_s, e_s)  # fixme 怎么排除pad的影响
        sentence_embed = [sentence_embed[lst_reverse[i]] for i in range((n-1) * batch_size)]
        cos_value = torch.cat(flatten([[self.cos(sentence_embed[i+j], sentence_embed[i+n-2]) for j in range(0, n-1)] \
                     for i in range(0, (n-1)*batch_size, (n-1))])).view(batch_size, -1)  #
        weight = self.softmax(cos_value).view(-1, 1)

        #
        input = self.dropout(embed)
        batch_input_packed = pack_padded_sequence(input, sentences_lens, batch_first=True)  # fixme
        encoder_outputs_packed, (h_last, c_last) = self.lstm(batch_input_packed)  # h_last: (1，(n-1)*b_s, h_s)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)  # fixme 怎么指定pad_idx

        # sum
        hidden_size = h_last.size()[2]
        tmp = [Variable(torch.FloatTensor([0]*hidden_size)).cuda() for k in range(self.max_utterance_num - n)] if use_cuda else \
            [Variable(torch.FloatTensor([0]*hidden_size)) for k in range(self.max_utterance_num - n)]

        h_last = [h_last[0][lst_reverse[i]] for i in range((n-1) * batch_size)] # ((n-1)*b_s, h_s) h_last排列
        h_input = torch.cat(h_last).view(batch_size, n-1, -1)
        h_last, _ = self.seq_lstm_h(h_input)  # ((n-1)*b_s, h_s)
        h_last_weight = torch.cat(h_last).view((n-1)*batch_size, -1) * weight
        h_last = [torch.cat([h_last_weight[i+j] for j in range(0, n-1)]  + tmp) \
                            for i in range(0, (n-1)*batch_size, (n-1))]  # cat
        h_last = torch.cat(h_last).view(batch_size, -1)  # (b_s, (max-n)*h_s)
        h_last = self.linear_h(h_last).view(1, batch_size, -1)  #


        c_last = [c_last[0][lst_reverse[i]] for i in range((n-1) * batch_size)]  # ((n-1)*b_s, h_s)
        c_input = torch.cat(c_last).view(batch_size, n-1, -1)
        c_last, _ = self.seq_lstm_c(c_input)  # ((n-1)*b_s, h_s)
        c_last_weight = torch.cat(c_last).view((n-1)*batch_size, -1) * weight
        c_last = [torch.cat([c_last_weight[i+j] for j in range(0, n-1)]  + tmp )\
                    for i in range(0, (n-1)*batch_size, (n-1))]  # cat
        c_last = torch.cat(c_last).view(batch_size, -1)  # (b_s, (max-n)*h_s)
        c_last = self.linear_c(c_last).view(1, batch_size, -1)  #
        return encoder_outputs, (h_last, c_last)

        # Todo
        # lst_reverse = sorted(lst, key = lambda d: lst[d])
        # h_last = [h_last[0][lst_reverse[i]] for i in range((n-1) * batch_size)]  # ((n-1)*b_s, 2*h_s)
        # weight = [self.cos(h_last[i+j].view(1,-1), h_last[i+n-2].view(1,-1)) \
        #                             for j in range(0, n-2) for i in range(0, (n-1)*batch_size, (n-1))].view(n-1, -1)
        # print (weight);exit(0)


class SumDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout, lang, key_flag='True'):
        super(SumDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.V = lang.word_size
        self.lstmcell = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.key_flag = key_flag
        self.softmax = nn.LogSoftmax()
        self.ssoftmax = nn.Softmax()
        #self.attn_linear = nn.Linear(2*self.hidden_size, self.V)
        self.linear = nn.Linear(self.hidden_size, self.V)
        self.embedding = nn.Embedding(self.V, self.embed_size)  #

    def forward(self, input, h_c, attn_flag=True):
        '''

        :param input:
        :param h_c:
        :param encoder_outputs: (b_s, m_l, h_s)
        :param attn_flag:  是否使用Attention
        :return:
        '''
        embed = self.embedding(input)  #
        input = self.dropout(embed)
        h_t, c_t = self.lstmcell(input, h_c)  #
        o_t = self.linear(h_t)
        y_t = self.softmax(o_t)
        h_c = (h_t, c_t)
        return y_t, h_c


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout, lang,):  # fixme dropout设置成不同的
        super(EncoderDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()
        self.V = lang.word_size
        self.embedding = nn.Embedding(self.V, self.embed_size)


    def forward(self, batch_input, batch_output, sentences_lens, keys, encoder, decoder, \
                pad_idx, embed_size, batch_size, n, lst, length_limitation=100):
        '''

        :param batch_to_be_generated:  形如[[[],],]的idx表示，需要送入encoder进行拼接操作后才能batch化
        :param keys:
        :param encoder:
        :param decoder:
        :param pad_idx:
        :param embed_size:
        :param length_limitation:
        :return:
        '''
        # encode
        max_length = batch_output.size()[1] if self.training else length_limitation  # fixme
        _, (h_last, c_last) = encoder.forward(batch_input, sentences_lens, \
                                                               keys, pad_idx, batch_size, n, lst)  # 这里encoder_outputs是双向的

        # decode
        decoder_input = Variable(torch.LongTensor([0]*batch_size).view(batch_size)).cuda() if use_cuda else \
            Variable(torch.LongTensor([0]*batch_size).view(batch_size))
        h_c = (h_last[0], c_last[0])  # 降维
        loss = 0
        predict_box = []
        for i in range(max_length - 1):
            y_t, h_c = decoder.forward(decoder_input, h_c)  # (batch_size, V)
            if self.training:
                decoder_input = batch_output.transpose(0,1)[i]  # fixme 第一步预测谁？？？？？
                loss += self.loss(y_t, decoder_input)
            else:
                _, predict_input = torch.max(y_t, 1)  # (batch_size, 1)
                decoder_input = predict_input
                predict_box.append(predict_input.data[:])  # [[],]

        if self.training:
            return loss
        else:
            return predict_box









































