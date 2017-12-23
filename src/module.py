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

    def forward(self, input, sentence_lens, pad_idx):
        '''

        :param input: (batch_size, max_length)
        :return:
        '''
        embed = self.embedding(input)
        input = self.dropout(embed)
        batch_input_packed = pack_padded_sequence(input, sentence_lens, batch_first=True)  # fixme

        encoder_outputs_packed, (h_last, c_last) = self.lstm(batch_input_packed)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)  # fixme 怎么指定pad_idx

        return encoder_outputs, (h_last, c_last)


class AttnDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout, lang):
        super(AttnDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.V = lang.word_size
        self.lstmcell = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, 1)
        )
        self.softmax = nn.LogSoftmax()
        self.attn_linear = nn.Linear(2*self.hidden_size, self.V)
        self.linear = nn.Linear(2*self.hidden_size, self.V)
        self.embedding = nn.Embedding(self.V, self.embed_size)

    def forward(self, input, h_c, encoder_outputs, flag=True):
        '''

        :param input:
        :param h_c:
        :param encoder_outputs:
        :param flag:  是否使用Attention
        :return:
        '''
        embed = self.embedding(input)
        input = self.dropout(embed)
        #input = input.squeeze(1)
        h_t, c_t = self.lstmcell(input, h_c)  # input:(b_s, e_s) h_c:元组,应该变为(b_s, h_s) h_t, c_t:(b_s, h_s)
        if (flag == True):
            #h_t = h_t.unsqueeze(1)  # (b_s, 1, h_s)
            h_t_extend = torch.cat([h_t.unsqueeze(1)] * encoder_outputs.size()[1], 1)  # (b_s, m_l, h_s)
            u_t = self.attn(torch.cat((encoder_outputs, h_t_extend), 2))  # (b_s, m_l, 1)
            a_t = self.softmax(u_t)
            h_t_ = (torch.sum((a_t * encoder_outputs), 1)).squeeze(1)  # (b_s, h_s)
            o_t = self.linear(torch.cat((h_t, h_t_), 1))  # (batch_size, V)
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
        self.hidden = hidden_size
        self.dropout = dropout
        self.loss = nn.NLLLoss()


    def forward(self, batch_input, batch_output, sentence_lens, encoder, decoder, pad_idx, embed_size, length_limitation=100):
        '''

        :param batch_input: [[],] (b_s, m_l)
        :param batch_output:
        :return:
        '''
        batch_size = len(batch_input)
        max_length = batch_output.size()[1] if self.training else length_limitation
        encoder_outputs, (h_last, c_last) = encoder.forward(batch_input, sentence_lens, pad_idx)  # 这里encoder_outputs是双向的
        decoder_input = Variable(torch.LongTensor([0]*batch_size).view(batch_size)).cuda() if use_cuda else \
            Variable(torch.LongTensor([0]*batch_size).view(batch_size))
        h_c = (h_last[0], c_last[0])  #fixme 为什么要取0
        loss = 0
        predict_box = []
        for i in range(max_length - 1):
            y_t, h_c = decoder.forward(decoder_input, h_c, encoder_outputs, True)  # (batch_size, V)
            if self.training:
                decoder_input = batch_output.transpose(0,1)[i]
                loss += self.loss(y_t, decoder_input)
            else:
                _, predict_input = torch.max(y_t, 1)  # (batch_size, 1)
                decoder_input = predict_input
                predict_box.append(predict_input.data[:])  # [[],]

        if self.training:
            return loss
        else:
            return predict_box









































