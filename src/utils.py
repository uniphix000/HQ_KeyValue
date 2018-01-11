#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')

def is_entity(word):
    """
    judge word is in entities or not
    :param word:  word to be judge
    :param entities:
    :return:
    """

    flag = False
    for i in range(len(word)):
        if word[i] == '<' or word[i] == '>' or word[i] == '_':
            flag = True
            break
    return flag

def cal_f(gold_entities, model_entities):
    precision, recall = 0., 0.
    for entity0 in model_entities:
        if entity0 in gold_entities:
            recall += 1.0 / len(gold_entities)
    for entity1 in gold_entities:
        if entity1 in model_entities:
            precision += 1.0 / len(model_entities)
    if (precision + recall) == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


if __name__ == '__main__':
    gold = [3,0]
    predict = [0,5,3]
    print (cal_f(gold, predict))






