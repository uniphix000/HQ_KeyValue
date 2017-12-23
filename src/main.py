#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')

import argparse
from datautils import *
import logging
import random
import torch.optim as optim
from module import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')

def main():
    cmd = argparse.ArgumentParser('Key-Value by H&Q')
    cmd.add_argument('--data_path', help='', default='../data/')
    cmd.add_argument('--hidden_size', help='', type=int, default=64)
    cmd.add_argument('--embed_size', help='', type=int, default=100)
    cmd.add_argument('--batch_size', help='', type=int, default=32)
    cmd.add_argument('--lr', help='', type=float, default=0.001)
    cmd.add_argument('--lr_decay', help='', type=float, default=0.95)
    cmd.add_argument('--max_epoch', help='', type=int, default=10)
    cmd.add_argument('--seed', help='', type=int, default=1234)
    cmd.add_argument('--dropout', help='', type=float, default=0)




    args = cmd.parse_args()
    print (args)

    random.seed(args.seed)

    # 数据预处理: 把标点和词分开,没有标点的统一加上.
    train_dialogs, valid_dialogs, test_dialogs = data_preprocess(args.data_path)
    # 提取keys, triples, entities
    keys, triples, entities, value_to_abstract_keys = key_extraction(train_dialogs, args.data_path)

    # 生成词典,先将key变成下划线形式加入词典,再将对话加入词典
    lang = Lang()
    lang = generate_dict(keys, train_dialogs, lang, value_to_abstract_keys)
    logging.info('dict generated! dict size:{0}'.format(lang.word_size))

    # 生成训练数据instances
    train_instances = generate_instances(keys, train_dialogs, triples, value_to_abstract_keys)
    valid_instances = generate_instances(keys, valid_dialogs, triples, value_to_abstract_keys)
    test_instances = generate_instances(keys, test_dialogs, triples, value_to_abstract_keys)
    logging.info('instances sample: {0}'.format(train_instances[:3]))

    # Word2idx
    train_instances_idx = sentence_to_idx(lang, train_instances)  # [([],[]),()]
    valid_instances_idx = sentence_to_idx(lang, valid_instances)
    test_instances_idx = sentence_to_idx(lang, test_instances)
    #print (train_instances_idx[:10])

    train_instances_size = len(train_instances_idx)
    valid_instances_size = len(valid_instances_idx)
    test_instances_size = len(test_instances_idx)
    logging.info('trainging size:{0} valid size:{0} test size:{0}'.format(train_instances_size, valid_instances_size, \
                                                                          test_instances_size))

    encoder = Encoder(args.embed_size, args.hidden_size, args.dropout, lang)
    decoder = AttnDecoder(args.embed_size, args.hidden_size, args.dropout, lang)
    encoderdecoder = EncoderDecoder(args.embed_size, args.hidden_size, args.dropout, lang)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    # train
    order = list(range(len(train_instances_idx)))
    for i in range(args.max_epoch):
        random.shuffle(order)
        start_id = 0
        for start_id in range(0, train_instances_size, args.batch_size):
            end_id = start_id + args.batch_size if start_id + args.batch_size < train_instances_size else train_instances_size
            batch_size = end_id - start_id
            batch_to_be_generated = [train_instances_idx[ids] for ids in order[start_id:end_id]]
            batch_input, batch_output, sentence_lens = generate_batch(batch_to_be_generated, batch_size, lang.word2idx['pad'])

            # train
            encoder.train()
            decoder.train()
            encoder.zero_grad()
            decoder.zero_grad()
            loss = encoderdecoder.train(batch_input, batch_output, sentence_lens, encoder, decoder, lang.word2idx['pad'], args.embed_size)
            print ('loss',loss)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()







if __name__ == '__main__':
    main()













