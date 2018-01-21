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
from datautils import use_cuda, sum_embed_kb


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, kb_row, dropout, lang, max_utterance_num):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.kb_row = kb_row
        self.dropout = nn.Dropout(dropout)
        self.V = lang.word_size
        self.embedding = nn.Embedding(self.V, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
        self.cos = nn.CosineSimilarity(dim=1)
        self.max_utterance_num = max_utterance_num
        self.linear_h = nn.Linear((max_utterance_num-1) * self.hidden_size, self.hidden_size)
        self.linear_c = nn.Linear((max_utterance_num-1) * self.hidden_size, self.hidden_size)

    def forward(self, batch_input, sentences_lens, batch_kb, keys, pad_idx, batch_size, n, lst):
        '''
            将句子一起过encoder，并sum
        :param batch_input: ((n-1)*b_s, m_l)
        :param batch_kb: (b_s, kb_row*2*kb_col)
        :param keys: [(,),...] 每个key已经被idx表示了,而keys里的每个元素都是词典中前2kv个word中的某一个, (kv, 2, 1)
        :param
        :return:  h_c: (1, b_s, 2*h_s)
        '''
        embed_key = self.embedding(keys.squeeze(2))
        keys = torch.sum(embed_key, 1)  # (kv, embed_size)

        embed_input = self.embedding(batch_input)  # ((n-1)*b_s, m_l, e_s)
        input = self.dropout(embed_input)

        embed_kb = self.embedding(batch_kb).view(batch_size, self.kb_row, -1, self.embed_size)  # (b_s, kb_row*2*kb_col, e_s)
        kb_row = sum_embed_kb(embed_kb)  # (b_s, kb_row, kb_col, e_s)

        batch_input_packed = pack_padded_sequence(input, sentences_lens, batch_first=True)  # fixme
        encoder_outputs_packed, (h_last, c_last) = self.lstm(batch_input_packed)  # h_last: (1，(n-1)*b_s, h_s)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)  # fixme 怎么指定pad_idx

        # cat
        hidden_size = h_last.size()[2]
        lst_reverse = sorted(lst, key = lambda d: lst[d])
        tmp = [Variable(torch.FloatTensor([0]*hidden_size)).cuda() for k in range(self.max_utterance_num - n)] if use_cuda else \
            [Variable(torch.FloatTensor([0]*hidden_size)) for k in range(self.max_utterance_num - n)]

        h_last = [h_last[0][lst_reverse[i]] for i in range((n-1) * batch_size)]  # ((n-1)*b_s, h_s)
        h_last = [torch.cat([h_last[i+j] for j in range(0, n-1)]  + tmp) \
                            for i in range(0, (n-1)*batch_size, (n-1))]  # cat
        h_last = torch.cat(h_last).view(batch_size, -1)  # (b_s, (max-n)*h_s)
        h_last = self.linear_h(h_last).view(1, batch_size, -1)  #

        c_last = [c_last[0][lst_reverse[i]] for i in range((n-1) * batch_size)]  # ((n-1)*b_s, h_s)
        c_last = [torch.cat([c_last[i+j] for j in range(0, n-1)]  + tmp )\
                            for i in range(0, (n-1)*batch_size, (n-1))]  # cat
        c_last = torch.cat(c_last).view(batch_size, -1)  # (b_s, (max-n)*h_s)
        c_last = self.linear_c(c_last).view(1, batch_size, -1)  #

        if (n>2):
            encoder_outputs = [encoder_outputs[lst_reverse[i]] for i in range ((n-1) * batch_size)]
            encoder_outputs = torch.cat([encoder_outputs[i+n-2] \
                                         for i in range(0, (n-1)*batch_size, (n-1))]).view(batch_size, -1, hidden_size)
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        return encoder_outputs, (h_last, c_last), keys, kb_row




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
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, 1)
        )
        self.attn_key = nn.Sequential(
            nn.Linear(self.embed_size + 2*self.hidden_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, 1)
        )
        self.linear = nn.Linear(self.hidden_size, self.V)
        self.attn_linear = nn.Linear(2*self.hidden_size, self.V)
        self.embedding = nn.Embedding(self.V, self.embed_size)  #

    def forward(self, input, h_c, k, context_vector, encoder_outputs, attn_flag=True):
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
        if attn_flag == True:











            h_t_extend = torch.cat([h_t.unsqueeze(1)] * encoder_outputs.size()[1], 1)  # (b_s, m_l, h_s)
            u_t = self.attn(torch.cat((encoder_outputs, h_t_extend), 2))  # (b_s, m_l, 1)
            a_t = self.ssoftmax(u_t.squeeze(2)).unsqueeze(2)
            h_t_ = (torch.sum((a_t * encoder_outputs), 1))  # (b_s, h_s)
            batch_size = len(h_t)
            kv = len(k)
            h_t = h_t.cuda() if use_cuda else h_t
            context_vector = context_vector.cuda() if use_cuda else context_vector
            k = k.cuda() if use_cuda else k
            h_t_extend_k = torch.cat([h_t.unsqueeze(1)] * kv, 1)  # (b_s, kv, h_s)
            context_vector_extend_k = torch.cat([context_vector.unsqueeze(1)] * kv, 1)  # (b_s, kv, h_s)
            k_extend = torch.cat([k.unsqueeze(0)] * batch_size, 0)  # (b_s, kv, e_s)
            u_k_t = self.attn_key(torch.cat((context_vector_extend_k, h_t_extend_k, k_extend), 2))  # (b_s, kv, 1) #[context;h_t;kj]
            tmp = Variable(torch.FloatTensor([0.0] * batch_size * (self.V - kv)).view(batch_size, \
                         -1, 1))  # fixme requires_grad
            tmp = tmp.cuda() if use_cuda else tmp
            v_k_t = torch.cat([tmp, u_k_t], 1).squeeze(2)  # (b_s, V)
            o_t = self.attn_linear(torch.cat((h_t, h_t_), 1)) + v_k_t if self.key_flag=='True' \
                else self.linear(h_t) # (batch_size, V)
            y_t = self.softmax(o_t)  # 用于预测下一个word
        else:
            o_t = self.linear(h_t)
            y_t = self.softmax(o_t)
        h_c = (h_t, c_t)
        return y_t, h_c


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout, lang,):  # fixme dropout设置成不同的
        super(EncoderDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()
        self.V = lang.word_size
        self.embedding = nn.Embedding(self.V, self.embed_size)
        self.attn_kb = nn.Sequential(
            nn.Linear(self.embed_size + self.hidden_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, 1)
        )


    def forward(self, batch_input, batch_output, batch_kb, sentences_lens, keys, encoder, decoder, \
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
        encoder_outputs, (h_last, c_last), k, kb = encoder.forward(batch_input, sentences_lens, batch_kb,\
                                                               keys, pad_idx, batch_size, n, lst)  # 这里encoder_outputs是双向的

        # 用h_last与kb_row作attn，选出kb中的某一行作为支撑该组对话的kb
        kb_row = kb.size()[1]
        kb = torch.sum(kb, 2)   # (b_s, kb_row, kb_col, e_s)-->(b_s, kb_row, e_s)
        h_last = torch.cat([h_last.transpose(0,1)]*kb_row).view(batch_size, kb_row, -1)#(1, b_s, h_s)-->(b_s, kb_row, h_s)
        u_kb = self.attn_kb(torch.cat((kb,h_last), 2))  # (b_s, kb_row, 1)
        _, kb_picked = torch.max(u_kb, 1)  #
        print (kb_picked);exit(0)




        # decode
        decoder_input = Variable(torch.LongTensor([0]*batch_size).view(batch_size)).cuda() if use_cuda else \
            Variable(torch.LongTensor([0]*batch_size).view(batch_size))
        h_c = (h_last[0], c_last[0])  # 降维
        context_vector = h_c[0]
        loss = 0
        predict_box = []
        for i in range(max_length - 1):
            y_t, h_c = decoder.forward(decoder_input, h_c, k, context_vector, encoder_outputs)  # (batch_size, V)
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









































