#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')


def cal_F_score(data, model, keys, word_idx):
    # now ,convert it to one sample
    n_tests = len(data)
    standard_sentence = ""
    predict_sentence = ""

    used_pair_count = 0
    total_f = 0.
    idx_to_word = word_idx.get_idx_to_word()
    for i in range(n_tests):
        # p_method_2, r_method_2 = 0., 0.

        # batch_ids = np.arange(n_tests)
        batch_ids = [i]
        test_inputs, test_input_lens, max_test_input_len = get_batch(data, batch_ids, 0)
        test_outputs, test_output_lens, max_test_output_len = get_batch(data, batch_ids, 1)
        # print("test_outputs={0}".format(test_outputs))
        # print(test_outputs)
        model_outputs = model.inference(test_inputs, test_input_lens, keys)  # 要估测的话，一定要写好inference
        # bleu = 0.
        # print(model_outputs)
        # 需要得到真实的句子和model预测出来的句子
        # print(test_outputs)
        model_entities = set()
        gold_entities = set()
        for k in range(1):
            model_output = ""
            model_output_word = ""
            true_output_word = ""
            for j in range(len(model_outputs[k])):
                if model_outputs[k][j].data[0] == EOS_TOKEN:
                    break
                predict_word = idx_to_word[model_outputs[k][j]]
                if is_entity(predict_word):
                    #print("predict_word = {0}".format(predict_word))
                    model_entities.add(predict_word)

                model_output_word += predict_word

                model_output += " " + str(model_outputs[k][j])
                model_output_word += " "
            predict_sentence += model_output_word
            predict_sentence += '\n'  # 每个加个换行
            # print(model_output_word)
            # print(model_output)
            true_output = ""
            for j in range(1, len(test_outputs[k])):
                if test_outputs[k][j].data[0] == EOS_TOKEN:
                    break
                gold_word = idx_to_word[test_outputs[k][j].data[0]]
                if is_entity(gold_word):
                    # print("gold_entity = {0}".format(gold_word))
                    gold_entities.add(gold_word)
                true_output += " " + str(test_outputs[k][j].data[0])
                true_output_word += gold_word
                true_output_word += " "
            standard_sentence += true_output_word
            standard_sentence += '\n'  # 每个加个换行
        if len(model_entities) == 0 and len(gold_entities) == 0:
            total_f += 1
            used_pair_count += 1
            continue
        total_f += cal_f(gold_entities, model_entities)
        used_pair_count += 1
    #LOG.info("overall F1 = {0}".format(total_f/used_pair_count))  # 这里代表每个句子算一个f了
    return (total_f/used_pair_count)