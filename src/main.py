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
from datautils import use_cuda
import codecs
import os
import subprocess
from torch.nn.utils import clip_grad_norm
from utils import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')

def main():
    cmd = argparse.ArgumentParser('Key-Value by H&Q')
    cmd.add_argument('--data_path', help='', default='../data/')
    cmd.add_argument('--hidden_size', help='', type=int, default=200)
    cmd.add_argument('--embed_size', help='', type=int, default=200)
    cmd.add_argument('--batch_size', help='', type=int, default=32)
    cmd.add_argument('--lr', help='', type=float, default=0.001)
    cmd.add_argument('--lr_decay', help='', type=float, default=1.0)
    cmd.add_argument('--max_epoch', help='', type=int, default=200)
    cmd.add_argument('--seed', help='', type=int, default=1234)
    cmd.add_argument('--dropout', help='', type=float, default=0.8)
    cmd.add_argument('--bleu_path', help='', default='../bleu/')
    cmd.add_argument('--grad_clip', help='', type=float, default=10)
    cmd.add_argument('--parallel_suffix', help='', type=str, default='123')
    cmd.add_argument('--model_save_path', help='', type=str, default='../model/')
    cmd.add_argument('--l2', help='', type=float, default=0.000005)



    args = cmd.parse_args()
    print (args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 数据预处理: 把标点和词分开,没有标点的统一加上.
    train_dialogs, valid_dialogs, test_dialogs = data_preprocess(args.data_path)
    # 提取keys, triples, entities
    keys, triples, entities, value_to_abstract_keys = key_extraction(train_dialogs, args.data_path)

    # 生成词典,先将key变成下划线形式加入词典,再将对话加入词典
    lang = Lang()
    lang, underlined_keys = generate_dict(keys, train_dialogs, lang, value_to_abstract_keys)
    logging.info('dict generated! dict size:{0}'.format(lang.word_size))

    # 生成训练数据instances
    train_instances = generate_instances(keys, train_dialogs, triples, value_to_abstract_keys)
    valid_instances = generate_instances(keys, valid_dialogs, triples, value_to_abstract_keys)
    test_instances = generate_instances(keys, test_dialogs, triples, value_to_abstract_keys)
    #logging.info('instances sample: {0}'.format(train_instances))

    # Word2idx
    train_instances_idx = sentence_to_idx(lang, train_instances)  # [([],[]),()]
    valid_instances_idx = sentence_to_idx(lang, valid_instances)
    test_instances_idx = sentence_to_idx(lang, test_instances)

    # keys2idx
    keys_idx = key_to_idx(lang, underlined_keys)

    train_instances_size = len(train_instances_idx)
    valid_instances_size = len(valid_instances_idx)
    test_instances_size = len(test_instances_idx)
    logging.info('trainging size:{0} valid size:{1} test size:{2}'.format(train_instances_size, valid_instances_size, \
                                                                          test_instances_size))

    encoder = Encoder(args.embed_size, args.hidden_size, args.dropout, lang)
    decoder = AttnDecoder(args.embed_size, args.hidden_size, args.dropout, lang)
    encoderdecoder = EncoderDecoder(args.embed_size, args.hidden_size, args.dropout, lang)
    encoder = encoder.cuda() if use_cuda else encoder
    decoder = decoder.cuda() if use_cuda else decoder
    encoderdecoder = encoderdecoder.cuda() if use_cuda else encoderdecoder
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
    encoderdecoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.l2)


    # train
    best_valid_bleu_score, best_test_bleu_score = 0, 0
    best_valid_f, best_test_f = 0, 0
    order = list(range(len(train_instances_idx)))
    for i in range(args.max_epoch):
        logging.info('--------------------Round {0}---------------------'.format(i))
        random.shuffle(order)
        start_id = 0
        count = 0
        total_loss = 0
        for start_id in range(0, train_instances_size, args.batch_size):
            end_id = start_id + args.batch_size if start_id + args.batch_size < train_instances_size else train_instances_size
            batch_size = end_id - start_id
            batch_to_be_generated = [train_instances_idx[ids] for ids in order[start_id:end_id]]
            batch_gold = [train_instances[ids] for ids in order[start_id:end_id]]  # 对于train来说没有用
            batch_input, batch_output, _, sentence_lens = generate_batch(batch_to_be_generated, batch_gold, batch_size, lang.word2idx['pad'])

            # train
            encoder.train()
            decoder.train()
            encoderdecoder.train()
            encoder.zero_grad()
            decoder.zero_grad()
            encoderdecoder.zero_grad()
            loss = encoderdecoder.forward(batch_input, batch_output, sentence_lens, keys_idx, encoder, decoder, lang.word2idx['pad'], args.embed_size)
            loss.backward()
            #print ('!!!!!!!!!!!!!!!!',decoder.parameters()['params'])
            clip_grad_norm(encoder.parameters(), args.grad_clip)
            clip_grad_norm(decoder.parameters(), args.grad_clip)
            clip_grad_norm(encoderdecoder.parameters(), args.grad_clip)
            encoder_optimizer.step()
            decoder_optimizer.step()
            encoderdecoder_optimizer.step()

            total_loss += loss.data
            count += 1

            # if (count % 100 == 0):
            #     logging.info('average loss: {0}'.format(total_loss*1.0/count))

        valid_bleu_score, valid_f = evaluate(keys_idx, encoder, decoder, encoderdecoder, valid_instances_idx, valid_instances, lang, \
                              args.batch_size, args.embed_size, args.hidden_size, args.bleu_path, args.parallel_suffix)
        # if (valid_bleu_score > best_valid_bleu_score):
        #     test_bleu_score, test_f = evaluate(keys_idx, encoder, decoder, encoderdecoder, test_instances_idx, test_instances, lang, \
        #               args.batch_size, args.embed_size, args.hidden_size, args.bleu_path, args.parallel_suffix)
        #     best_test_bleu_score = max(best_test_bleu_score, test_bleu_score)
        #
        #     logging.info('New Record! test bleu score now: {0} best test bleu score ever: {1}'.format(\
        #         test_bleu_score, best_test_bleu_score))

        if (valid_f > best_valid_f):
            torch.save(encoder.state_dict(), os.path.join(args.model_save_path, 'encoder.pkl'))
            torch.save(encoder.state_dict(), os.path.join(args.model_save_path, 'decoder.pkl'))
            torch.save(encoderdecoder.state_dict(), os.path.join(args.model_save_path, 'encoderdecoder.pkl'))
            test_bleu_score, test_f = evaluate(keys_idx, encoder, decoder, encoderdecoder, test_instances_idx, test_instances, lang, \
                      args.batch_size, args.embed_size, args.hidden_size, args.bleu_path, args.parallel_suffix)
            best_test_f = max(best_test_f, test_f)
            best_test_bleu_score = max(best_test_bleu_score, test_bleu_score)
            logging.info('New Record! test F now: {0} best test F ever: {1} test bleu score now: {2} best test bleu score ever: {3}'.format(\
                test_f, best_test_f, test_bleu_score, best_test_bleu_score))
        best_valid_f = max(best_valid_f, valid_f)
        best_valid_bleu_score = max(best_valid_bleu_score, valid_bleu_score)
        logging.info('valid F: {0} best valid F ever: {1}'.format(valid_f, best_valid_f))

        logging.info('valid bleu score: {0} best valid bleu score ever: {1}'.format(valid_bleu_score, best_valid_bleu_score))
    logging.info('Trianing complete! best valid bleu score: {0} best test bleu score: {1} best valid F: {2} best test F: {3}'\
                 .format(best_valid_bleu_score, best_test_bleu_score, best_valid_f, best_test_f))


def evaluate(keys_idx, encoder, decoder, encoderdecoder, instances_idx, instances, lang, \
                              batch_size, embed_size, hidden_size, bleu_path, parallel_suffix):
    '''

    :param encoder:
    :param decoder:
    :param encoderdecoder:
    :param instances_idx:
    :param lang:
    :param batch_size:
    :param embed_size:
    :param hidden_size:
    :return:
    '''
    order = list(range(len(instances_idx)))
    instances_size = len(instances_idx)
    random.shuffle(order)
    start_id = 0
    predict_all = []
    gold_all = []
    for start_id in range(0, instances_size, batch_size):
        end_id = start_id + batch_size if start_id + batch_size < instances_size else instances_size
        batch_size = end_id - start_id
        batch_to_be_generated = [instances_idx[ids] for ids in order[start_id:end_id]]
        batch_gold = [instances[ids] for ids in order[start_id:end_id]]  # 为了跳过idx的转化
        batch_input, batch_output, batch_gold_output, sentence_lens = generate_batch(batch_to_be_generated, batch_gold, batch_size, lang.word2idx['pad'])

        # eval
        encoder.eval()
        decoder.eval()
        encoderdecoder.eval()
        batch_predict = encoderdecoder.forward(batch_input, batch_output, sentence_lens, keys_idx, encoder, decoder, lang.word2idx['pad'], embed_size)
        predict_all.append(batch_predict)
        gold_all.append(batch_gold_output)
    predict_sentences, gold_sentences = transfor_idx_to_sentences(predict_all, gold_all, lang)

    with codecs.open(os.path.join(bleu_path, ''.join(['predict', parallel_suffix])), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(predict_sentences))
        fp.close()
    with codecs.open(os.path.join(bleu_path, ''.join(['gold', parallel_suffix])), 'w', encoding='utf-8') as fg:
        fg.write('\n'.join(gold_sentences))
        fg.close()

    # 下面对预测和gold的进行f值计算,we should cal micro_f
    # for entity in predict_sentences:
    #     for word in entity.split():
    #         if is_entity(word):
    #             model_entities.add(word)
    #
    # for entity in gold_sentences:
    #     for word in entity.split():
    #         if is_entity(word):
    #             gold_entities.add(word)
    total_f, total_num = 0., 0
    for index, predict_sentence in enumerate(predict_sentences):
        model_entities, gold_entities = set(), set()
        gold_sentence = gold_sentences[index]
        for predict_word in predict_sentence.split():
            if is_entity(predict_word):
                model_entities.add(predict_word)
        for gold_word in gold_sentence.split():
            if is_entity(gold_word):
                gold_entities.add(gold_word)
        if len(model_entities) == 0 and len(gold_entities) == 0:  # 防止遇到空
            # total_num += 1
            # total_f += 1
            continue

        total_num += 1
        total_f += cal_f(gold_entities, model_entities)
    print ('total_num:',total_num)
    # f = cal_f(gold_entities, model_entities)
    f = total_f / total_num
    #logging.info("f = {0}".format(f))
    p = subprocess.Popen(['perl', '../bleu/multi-bleu.pl', '../bleu/gold'+str(parallel_suffix)], stdin=open('../bleu/predict' + str(parallel_suffix)), stdout=subprocess.PIPE)
    # 原来的命令<代表重定向,相当于开了一个
    for line in p.stdout.readlines():
        return (float(line.split()[0][:-1])), f


def transfor_idx_to_sentences(predict_all, gold_all, lang):
    '''

    :param batch_predict: [[[],],] 里面第一个list是一个batch, batch里的每个list是列数据
    :param gold_all: [[,],] 每个batch的batch_size个回复
    :param lang:
    :return:
    '''
    predict_sentences = []
    gold_sentences = []
    for predict_batch in predict_all:
        batch_size = len(predict_batch[0])
        sentences_idx = torch.cat([col.view(batch_size, -1) for col in predict_batch], 1)
        for sentence_idx in sentences_idx:
            sentence = ''
            for idx in sentence_idx:
                if (idx != lang.word2idx['<EOS>']):
                    if (idx != lang.word2idx['<BOS>']):
                        word = lang.get_idx_to_word()[idx]
                        sentence = sentence + word + ' '
                else: break
            predict_sentences.append(sentence)
    for gold_batch in gold_all:
        for gold_sentence in gold_batch:
            gold_sentences.append(gold_sentence)

    return predict_sentences, gold_sentences


if __name__ == '__main__':
    main()













