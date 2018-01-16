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
from datautils import use_cuda


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout, lang):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.V = lang.word_size
        self.embedding = nn.Embedding(self.V, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)

    def forward(self, input, sentence_lens, keys, pad_idx):
        '''
        这里要得到kj
        :param: keys: [(,),...] 每个key已经被idx表示了,而keys里的每个元素都是词典中前2kv个word中的某一个, (kv, 2, 1)
        :param input: (batch_size, max_length)
        :return:
        '''
        embed_key = self.embedding(keys.squeeze(2))
        k = torch.sum(embed_key, 1)  # (kv, embed_size)

        embed = self.embedding(input)
        input = self.dropout(embed)
        batch_input_packed = pack_padded_sequence(input, sentence_lens, batch_first=True)  # fixme

        encoder_outputs_packed, (h_last, c_last) = self.lstm(batch_input_packed)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)  # fixme 怎么指定pad_idx

        return encoder_outputs, (h_last, c_last), k


class AttnDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout, lang, key_flag='True'):
        super(AttnDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.V = lang.word_size
        self.lstmcell = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.key_flag = key_flag
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, 1)
        )
        self.attn_key = nn.Sequential(
            nn.Linear(self.embed_size + self.hidden_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, 1)
        )
        self.softmax = nn.LogSoftmax()
        self.ssoftmax = nn.Softmax()
        #self.attn_linear = nn.Linear(2*self.hidden_size, self.V)
        self.linear = nn.Linear(2*self.hidden_size, self.V)
        self.linear0 = nn.Linear(self.hidden_size, self.V)
        self.embedding = nn.Embedding(self.V, self.embed_size)  # bug

    def forward(self, input, h_c, encoder_outputs, k, attn_flag=False):
        '''

        :param input:
        :param h_c:
        :param encoder_outputs: (b_s, m_l, h_s)
        :param attn_flag:  是否使用Attention
        :return:
        '''
        embed = self.embedding(input)
        input = self.dropout(embed)
        h_t, c_t = self.lstmcell(input, h_c)  # input:(b_s, e_s) h_c:元组,应该变为(b_s, h_s) h_t, c_t:(b_s, h_s)
        if (attn_flag == True):
            h_t_extend = torch.cat([h_t.unsqueeze(1)] * encoder_outputs.size()[1], 1)  # (b_s, m_l, h_s)
            u_t = self.attn(torch.cat((encoder_outputs, h_t_extend), 2))  # (b_s, m_l, 1)
            #a_t = self.softmax(u_t)  # ?softmax？
            a_t = self.ssoftmax(u_t.squeeze(2)).unsqueeze(2)
            #a_t = u_t
            h_t_ = (torch.sum((a_t * encoder_outputs), 1))  # (b_s, h_s)
            batch_size = len(h_t_)
            kv = len(k)
            h_t_extend_k = torch.cat([h_t.unsqueeze(1)] * kv, 1)  # (b_s, kv, h_s)
            k_extend = torch.cat([k.unsqueeze(0)] * batch_size, 0)  # (b_s, kv, e_s)
            u_k_t = self.attn_key(torch.cat((h_t_extend_k, k_extend), 2))  # (b_s, kv, 1)
            tmp = Variable(torch.FloatTensor([0.0] * batch_size * (self.V - kv)).view(batch_size, \
                         -1, 1))  # fixme requires_grad
            tmp = tmp.cuda() if use_cuda else tmp
            v_k_t = torch.cat([tmp, u_k_t], 1).squeeze(2)  # (b_s, V)
            o_t = self.linear(torch.cat((h_t, h_t_), 1)) + v_k_t if self.key_flag=='True' \
                else self.linear(torch.cat((h_t, h_t_), 1)) # (batch_size, V)
            y_t = self.softmax(o_t)  # 用于预测下一个word
        else:
            o_t = self.linear0(h_t)
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


    def forward(self, batch_input, batch_output, sentence_lens, keys, encoder, decoder, pad_idx, embed_size, length_limitation=100):
        '''

        :param batch_input: [[],] (b_s, m_l)
        :param batch_output:
        :return:
        '''
        # encode
        batch_size = len(batch_input)
        max_length = batch_output.size()[1] if self.training else length_limitation
        encoder_outputs, (h_last, c_last), k = encoder.forward(batch_input, sentence_lens, keys, pad_idx)  # 这里encoder_outputs是双向的

        # decode
        decoder_input = Variable(torch.LongTensor([2]*batch_size).view(batch_size)).cuda() if use_cuda else \
            Variable(torch.LongTensor([2]*batch_size).view(batch_size))
        h_c = (h_last[0], c_last[0])  # 降维
        loss = 0
        predict_box = []
        for i in range(max_length - 1):
            y_t, h_c = decoder.forward(decoder_input, h_c, encoder_outputs, k)  # (batch_size, V)
            if self.training:
                decoder_input = batch_output.transpose(0,1)[i]  # fixme 第一步预测谁？？？？？？？？？？
                loss += self.loss(y_t, decoder_input)
            else:
                _, predict_input = torch.max(y_t, 1)  # (batch_size, 1)
                decoder_input = predict_input
                predict_box.append(predict_input.data[:])  # [[],]

        if self.training:
            return loss
        else:
            return predict_box









































