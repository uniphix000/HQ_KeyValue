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

def count_num(gold_entities, model_entities):
    count = 0
    for entity in model_entities:
        if entity in gold_entities:
            count += 1
    return count

def cal_macro_F(predict_sentences, gold_sentences):
    '''

    :param predict_sentences:
    :return:
    '''
    num, total_num = 0, 0
    gold_entities_num, model_entities_num = 0, 0
    for index, predict_sentence in enumerate(predict_sentences):
        model_entities, gold_entities = set(), set()  #
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
        num += count_num(gold_entities, model_entities)
        gold_entities_num += len(gold_entities)
        model_entities_num += len(model_entities)
    precision = num * 1.0 / model_entities_num if model_entities_num != 0 else 0
    recall = num * 1.0 / gold_entities_num if gold_entities_num != 0 else 0
    return  2 * precision * recall / (precision + recall) if (precision * recall != 0) else 0


if __name__ == '__main__':
    gold = [3,0]
    predict = [0,5,3]
    print (cal_f(gold, predict))






