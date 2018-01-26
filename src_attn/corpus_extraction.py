#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')

import os
import codecs
import json
from collections import defaultdict
import copy

def preprocess(path):
    '''

    :param dialog:
    :return:
    '''
    with codecs.open(path, 'r', encoding='utf-8') as fo:
        dialogs = json.loads(fo.read())
        fo.close()
    for dialog in dialogs:
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
        if (dialog['scenario']['task']['intent'] == 'navigate'):
            flag = True
            text = []
            for dialogue in dialog['dialogue']:
                if (dialogue['turn'] == 'assistant'):
                    text.append('<<< '+noralize_value(normalize_key(dialogue['data']['utterance'], keys), value_to_abstract_keys))
                elif (dialogue['turn'] == 'driver'):
                    text.append('>>> '+normalize_key(dialogue['data']['utterance'], keys))
            instances.append(text)

    return instances

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
    return instances


if __name__ == '__main__':
    train_dialogs, valid_dialogs, test_dialogs = data_preprocess('../data')
    train_dialogs = valid_dialogs
    keys, triples, entities, value_to_abstract_keys = key_extraction(train_dialogs, '../data')
    train_instances = generate_instances(keys, train_dialogs, triples, value_to_abstract_keys)
    with codecs.open('../data/valid_corpus', 'w', encoding='utf-8') as fp:
        for dialog in train_instances:
            fp.write('\n'.join(dialog))
            fp.write('\n\n\n')


