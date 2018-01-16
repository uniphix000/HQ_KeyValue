#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import json
import os
import codecs
from collections import defaultdict
import copy
from torch.autograd import Variable
import torch

use_cuda = torch.cuda.is_available()
oov = 1

def preprocess(path):
    '''

    :param dialog:
    :return:
    '''
    with codecs.open(path, 'r', encoding='utf-8') as fo:
        dialogs = json.loads(fo.read())
        fo.close()

    for idx,dialog in enumerate(dialogs):
        #if (1):
        if (dialog['scenario']['task']['intent'] == 'schedule'):
            for dialogue in dialog['dialogue']:
                    dialogue['data']['utterance'] = dialogue['data']['utterance'] + ' '  #fixme 怎么复制并且共享操作
                    if len(dialogue['data']['utterance']) == 0:
                        continue
                    else:
                        dialogue['data']['utterance'] = dialogue['data']['utterance'].lower()
                        # if (dialogue['data']['utterance'][-1] == '.') or (dialogue['data']['utterance'][-1] == '?') or \
                        #                 (dialogue['data']['utterance'][-1] == '!') and (dialogue['data']['utterance'][-2] != ' '):
                        #     dialogue['data']['utterance'] = dialogue['data']['utterance'][:-1] + ' ' + dialogue['data']['utterance'][-1]
                        # elif (dialogue['data']['utterance'][-1] != '.') or (dialogue['data']['utterance'][-1] != '?') \
                        #         or (dialogue['data']['utterance'][-1] != '!'):
                        #     dialogue['data']['utterance'] += ' .'
                        dialogue['data']['utterance'] = dialogue['data']['utterance'].replace('. ', ' . ')
                        dialogue['data']['utterance'] = dialogue['data']['utterance'].replace('?', ' ?')
                        dialogue['data']['utterance'] = dialogue['data']['utterance'].replace('!', ' !')
                        dialogue['data']['utterance'] = dialogue['data']['utterance'].replace(',', ' ,')
        else:
            dialogs[idx] = None
    return dialogs


def data_preprocess(path):
    '''
    数据预处理: 把标点和词分开,没有标点的统一加上.
    :param path:
    :return:
    '''

    train_path = os.path.join(path, 'kvret_train_public.json')
    valid_path = os.path.join(path, 'kvret_dev_public.json')
    test_path = os.path.join(path, 'kvret_test_public.json')

    # train_path = os.path.join(path, 'test.json')
    # valid_path = os.path.join(path, 'test.json')
    # test_path = os.path.join(path, 'test.json')
    return preprocess(train_path), preprocess(valid_path), preprocess(test_path)


def key_extraction(train_dialogs, path):
    '''
    提取keys, triples, entities
    :param train_dialogs:
    :return:
    '''
    keys = set()
    value_to_abstract_keys = {}
    triples = defaultdict(lambda :defaultdict(int))  #双层dict  #fixme
    with codecs.open(os.path.join(path, 'kvret_entities.json'), 'r', encoding='utf-8') as fo:
        entities = json.loads(fo.read())

    for dialog in train_dialogs:
        if dialog is not None:
            if (dialog['scenario']['kb']['items']) is not None:
                domin = dialog['scenario']['kb']['kb_title']
                primary_key = ''
                if (domin == 'location information'):
                    primary_key = 'poi'
                elif (domin == "weekly forecast"):
                    primary_key = 'location'
                elif (domin == "calendar"):
                    primary_key = 'event'
                for item in dialog['scenario']['kb']['items']:  # item是一个包含键值信息的dict
                    subject = item[primary_key]
                    for (relation, value) in item.items():
                        value = value.lower()
                        key = (subject, relation) = (subject.lower(), relation.lower())
                        keys.add(key)
                        triples[key][value] += 1
                        value_to_abstract_keys[value] = "<" + '_'.join(key[0].split()) + ":" + "_".join(key[1].split()) + '>' #fixme

    return keys, triples, entities, value_to_abstract_keys


def key_to_idx(lang, underlined_keys):
    '''

    :param lang:
    :param underlined_keys:
    :return:
    '''
    keys_idx = []
    for (key_0, key_1) in underlined_keys:
        (key_0_ul, key_1_ul) = ([lang.word2idx[key_0]],[lang.word2idx[key_1]])
        keys_idx.append((key_0_ul, key_1_ul))
    keys_idx = Variable(torch.LongTensor(keys_idx))
    keys_idx = keys_idx.cuda() if use_cuda else keys_idx
    return keys_idx


class Lang:
    def __init__(self):
        self.word2idx = {'pad':0, 'oov':1, '<BOS>':2, '<EOS>':3}
        self.idx2word = {}
        self.word_size = 4

    def add_word(self, word):
        '''
        add word to dict
        :param word:
        :return:
        '''
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.word_size += 1

    def add_sentence(self, sentence):
        '''
        add sentence to dict
        :param sentence:
        :return:
        '''
        for word in sentence.strip().split():
            self.add_word(word)

    def get_idx_to_word(self):
        '''
        get idx_to_word
        :return:
        '''
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}
        return self.idx2word

    def sentence_to_idx(self, sentence):
        '''
        '''
        return [self.word2idx.get(word, 1) for word in sentence.split()]


def generate_dict(keys, train_dialogs, lang, value_to_abstract_keys):
    '''
    生成词典,先将key变成下划线形式加入词典,再将对话将入词典
    :param keys:
    :param train_dialogs:
    :param lang:
    :return:
    '''
    underlined_keys = []
    for key in keys:
        key_0 = '_'.join(key[0].split())
        key_1 = '_'.join(key[1].split())
        lang.add_word(key_0)
        lang.add_word(key_1)
        underlined_keys.append((key_0,key_1))

    for dialog in train_dialogs:
        if dialog is not None:
            for dialogue in dialog['dialogue']:
                lang.add_sentence(dialogue['data']['utterance'])

    for (value, key) in value_to_abstract_keys.items():
        lang.add_word(key)

    return lang, underlined_keys


def normalize_key(sentence, keys):
    '''
    把句子中出现的key替换成抽象的key
    :param sentence:
    :param keys:
    :return:
    '''
    sentence = copy.deepcopy(sentence)
    for key in keys:
        sentence = sentence.replace(key[0], '_'.join(key[0].split()))
        sentence = sentence.replace(key[1], '_'.join(key[1].split()))
    return sentence


def noralize_value(sentence, value_to_abstract_keys):
    '''
    把句子中出现的value替换成对应的抽象的<:>
    :param sentence:
    :return:
    '''
    sentence = copy.deepcopy(sentence)
    for value in value_to_abstract_keys:
        sentence = sentence.replace(' ' + value, ' ' + value_to_abstract_keys[value]) #fixme
    return sentence


def generate_instances(keys, train_dialogs, triples, value_to_abstract_keys):
    '''
    生成形如[(u1 s1 u2, s2),...]的数据
    :param keys:
    :param train_dialogs:
    :param triples:
    :param value_to_abstract_keys:
    :return:
    '''
    instances = []
    for dialog in train_dialogs:
        if dialog is not None:
            #if (1):
            if (dialog['scenario']['task']['intent'] == 'schedule'):
                flag = True
                for dialogue in dialog['dialogue']:
                    if (dialogue['turn'] == 'assistant'):
                        output_sentence = noralize_value(normalize_key(dialogue['data']['utterance'], keys), value_to_abstract_keys)
                        instances.append((input_sentence, output_sentence))
                        input_sentence += ' '
                    elif (dialogue['turn'] == 'driver'):
                        if flag:
                            input_sentence = ''
                            flag = False
                            pass
                        else:
                            input_sentence += ' '
                    input_sentence += normalize_key(dialogue['data']['utterance'], keys)

    return instances


def sentence_to_idx(lang, instances):
    '''

    :param lang:
    :param train_instances: [(),()]
    :return: [([],[]),()]
    '''
    idx_instances = []
    for instance in instances:
        instance_0 = [lang.word2idx['<BOS>']]+ lang.sentence_to_idx(instance[0])
        instance_1 = [lang.word2idx['<BOS>']]+ lang.sentence_to_idx(instance[1]) + [lang.word2idx['<EOS>']]
        idx_instances.append((instance_0, instance_1))
    return idx_instances


def generate_batch(instances, batch_gold, batch_size, pad_idx):
    '''

    :param instances: [([],[]),]
    :param batch_gold:
    :param pad_idx:
    :return: [[],] (batch_size, max_length)
    '''
    batch_input = []
    batch_output = []
    for (input, output) in instances:
        batch_input.append(input)
        batch_output.append(output)
    batch_gold_output = []
    for (_, gold_output) in batch_gold:
        batch_gold_output.append(gold_output)
    lst = range(batch_size)
    lst = sorted(lst, key = lambda d: -len(batch_input[d]))
    batch_input = [batch_input[ids] for ids in lst]
    batch_output = [batch_output[ids] for ids in lst]
    batch_gold_output = [batch_gold_output[ids] for ids in lst]

    input_max_length = len(batch_input[0])
    output_max_length = max([len(batch_output[i]) for i in range(batch_size)])

    batch_input = [batch_input[i] + [pad_idx] * (input_max_length - len(batch_input[i])) for i in range(batch_size)]
    batch_output = [batch_output[i] + [pad_idx] * (output_max_length - len(batch_output[i])) for i in range(batch_size)]

    batch_input = Variable(torch.LongTensor(batch_input)).cuda() if use_cuda else Variable(torch.LongTensor(batch_input))
    batch_output = Variable(torch.LongTensor(batch_output)).cuda() if use_cuda else Variable(torch.LongTensor(batch_output))
    sentence_lens = [len(batch_input[i]) for i in range(batch_size)]
    return batch_input, batch_output, batch_gold_output, sentence_lens



















