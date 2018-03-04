#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/28 上午10:53
# @Author  : yizhen
# @Site    : 
# @File    : datautils.py
# @Software: PyCharm

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
import itertools

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

    knowledge_base = []
    poi_selected = []
    last_num = 0
    for idx,dialog in enumerate(dialogs):
        #if (1):
        kb_flag = True
        if (dialog['scenario']['task']['intent'] == 'navigate'):
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
                        if kb_flag == True:
                            if 'slots' in dialogue['data']:
                                if 'poi_type' in dialogue['data']['slots']:
                                    poi_selected.append(dialogue['data']['slots']['poi_type'].lower())
                                    kb_flag = False
                        else:
                            if 'slots' in dialogue['data']:
                                if 'poi_type' in dialogue['data']['slots']:
                                    poi_selected[-1] = dialogue['data']['slots']['poi_type'].lower()
            # if (len(poi_selected)) == last_num:
            #     print (dialogs[idx]);exit(0)
            # last_num = len(poi_selected)
        else:
            dialogs[idx] = None
        if (dialog['scenario']['task']['intent'] == 'navigate'):
            if (dialog['scenario']['kb']['items'] is not None):
                for item in dialog['scenario']['kb']['items']:
                    for key in item:
                        item[key] = item[key].lower()
            knowledge_base.append(dialog['scenario']['kb']['items'])
        else:
            pass
    dialogs_navi = []
    for item in dialogs:
        if item is not None:
            dialogs_navi.append(item)
    # print (poi_selected)
    # print (len(knowledge_base))
    # print (len(poi_selected));exit(0)
    return dialogs_navi, knowledge_base


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
    提取keys, triples, entities, kb
    :param train_dialogs:
    :return:
    '''
    keys = set()
    value_to_abstract_keys = {}
    triples = defaultdict(lambda :defaultdict(int))  #双层dict  #fixme
    knowledge_base = []
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

def generate_dict_manual(lang, path):
    '''
    将手动的数据加入字典中，原来已经加过了，但是这里手动改了数据，防止不必要的错误，这里进行加上。
    :param lang:
    :param path:
    :return:
    '''
    train_path = os.path.join(path,'train.txt')
    with codecs.open(train_path, 'r', encoding='utf-8') as fp:
        fp_content = fp.read().strip().split('\n\n')
        for dialog in fp_content:
            for sentence in dialog.strip().split('\n'):
                lang.add_sentence(sentence)
    return lang

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
        sentence = sentence.replace(' ' + value, ' ' + value_to_abstract_keys[value]) # fixme
    return sentence

def generate_instances_manual(path, type, kb):
    '''
    generate instances
    生成形如[((u1 s1 u2, s2),kb),...]的数据
    :param path: manual data path
    :param type: train、valid、test
    :return:
    '''
    data_path = os.path.join(path, type+'.txt')
    with codecs.open(data_path, 'r' ,encoding='utf-8') as fp:
        train_dialogs = fp.read().strip().split('\n\n')
        instances = []
        for num, dialog in enumerate(train_dialogs):
            if (dialog == None):
                continue
            else:
                dialog = dialog.strip().split('\n')
                if len(dialog) % 2 != 0:  # if illegal, drop it
                    continue
                instance_2 = []
                if dialog is not None:
                    flag = True
                    for idx, dialogue in enumerate(dialog):
                        if idx % 2 == 1:
                            output_sentence = dialogue
                            # instance_2.append(output_sentence)
                            instance_1 = (input_sentence, output_sentence)
                            instances.append((instance_1, kb_normalization(kb[num])))
                            # instances.append((instance_1, copy.deepcopy(instance_2)))
                            input_sentence += ' '
                        elif idx % 2 == 0:
                            input_sentence_2 = dialogue
                            instance_2.append(input_sentence_2)
                            if flag:
                                input_sentence = ''
                                flag = False
                                pass
                            else:
                                input_sentence += ' '
                        input_sentence += dialogue
    return instances


def kb_normalization(kb):
    '''
    标准化kb的表示
    :param kb:
    :return:
    '''
    norm_kb = []
    for row in kb:
        norm_row = []
        poi = '_'.join(row['poi'].split())
        # for key,value in row.items():
        #     norm_row.append('<'+poi+':'+key+'>')
        norm_row.append('<'+poi+':'+'poi'+'>')
        norm_row.append('<'+poi+':'+'poi_type'+'>')
        norm_kb.append(norm_row)
    return norm_kb


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
            #if (dialog['scenario']['task']['intent'] == 'navigate'):
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
    :return: [(([],[]),[[],[],..]),()]
    '''
    idx_instances = []
    for (instance, kb) in instances:
        instance_0 = [lang.word2idx['<BOS>']]+ lang.sentence_to_idx(instance[0])
        instance_1 = [lang.word2idx['<BOS>']]+ lang.sentence_to_idx(instance[1]) + [lang.word2idx['<EOS>']]
        kb_ = copy.deepcopy(kb)
        if kb_ is not None:
            for item in kb_:
                for idx,key in enumerate(item):
                    item[idx] = lang.word2idx.get(key, 1)
        idx_instances.append(((instance_0, instance_1),kb_))
    return idx_instances


def generate_batch(instances, batch_gold, batch_size, pad_idx):
    '''

    :param instances: [(([],[]),[[],[],..]),()]
    :param batch_gold:
    :param pad_idx:
    :return: [[],] (batch_size, max_length)
    '''
    batch_input = []
    batch_output = []
    batch_kb = []
    for ((input, output), kb) in instances:
        batch_input.append(input)
        batch_output.append(output)
        batch_kb.append(kb)

    batch_gold_output = []

    for (_, gold_output) in batch_gold:
        batch_gold_output.append(gold_output)

    lst = range(batch_size)
    lst = sorted(lst, key = lambda d: -len(batch_input[d]))
    batch_input = [batch_input[ids] for ids in lst]
    batch_output = [batch_output[ids] for ids in lst]  # 这里是统一按照lst进行排序，我们后面是按照output的原始顺序来做
    batch_gold_output = [batch_gold_output[ids] for ids in lst]
    batch_kb = [batch_kb[ids] for ids in lst]  # [[[],..],..]

    batch_poi = []
    batch_type = []
    max_poi, max_type = 0, 0
    for idx,item in enumerate(batch_kb):  # 行要满8
        if item is not None:
            # for key in item:
            #     max_poi = max(len(key['poi']), max_poi)
            #     max_type = max(len(key['poi_type']), max_type)
            if len(item) < 8:
                batch_kb[idx].append([1,1])  # fixme 这里用0还是1
        else:
            raise (IOError)
    # batch_poi = [[key['poi'] + [0] * (max_poi - len(key['poi'])) for key in item] \
    #     for item in batch_kb]
    # batch_type = [[key['poi_type'] + [0] * (max_type - len(key['poi_type'])) for key in item] \
    #     for item in batch_kb]

    input_max_length = len(batch_input[0])
    output_max_length = max([len(batch_output[i]) for i in range(batch_size)])
    sentence_lens = [len(batch_input[i]) for i in range(batch_size)]

    batch_input = [batch_input[i] + [pad_idx] * (input_max_length - len(batch_input[i])) for i in range(batch_size)]
    batch_output = [batch_output[i] + [pad_idx] * (output_max_length - len(batch_output[i])) for i in range(batch_size)]

    batch_input = Variable(torch.LongTensor(batch_input)).cuda() if use_cuda else Variable(torch.LongTensor(batch_input))
    batch_output = Variable(torch.LongTensor(batch_output)).cuda() if use_cuda else Variable(torch.LongTensor(batch_output))

    # batch_poi = Variable(torch.LongTensor(batch_poi)).cuda() if use_cuda else Variable(torch.LongTensor(batch_poi))
    # batch_type = Variable(torch.LongTensor(batch_type)).cuda() if use_cuda else Variable(torch.LongTensor(batch_type))
    batch_kb = Variable(torch.LongTensor(batch_kb)).cuda() if use_cuda else Variable(torch.LongTensor(batch_kb))

    return batch_input, batch_output, batch_gold_output, batch_kb, batch_poi, batch_type, sentence_lens


def flatten(lst):
    return list(itertools.chain.from_iterable(lst))

















