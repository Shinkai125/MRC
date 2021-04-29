# -*- coding: utf-8 -*-
# !nvidia-smi

# import os
# path = "/content/drive/My Drive"

# os.chdir(path)
# os.listdir(path)

# !pip install transformers

# !pip install tensorflow-gpu -U


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import transformers

transformers.logging.set_verbosity_error()

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from collections import OrderedDict

import json
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras import layers
from tqdm import tqdm
from transformers import BertTokenizerFast, TFBertModel


# 以下是中文阅读理解的的评估函数，使用的是CMRC的评测脚本
##########################评测函数开始#################################
def _tokenize_chinese_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def _normalize(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
        u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
        u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > max_len:
                    max_len = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - max_len:p], max_len


def evaluate(ref_ans, pred_ans, verbose=False):
    """
    ref_ans: reference answers, dict
    pred_ans: predicted answer, dict
    return:
        f1_score: averaged F1 score
        em_score: averaged EM score
        total_count: number of samples in the reference dataset
        skip_count: number of samples skipped in the calculation due to unknown errors
    """
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for document in ref_ans:
        para = document[1].strip()
        total_count += 1
        query_id = document[0]
        query_text = document[2].strip()
        answers = document[3]
        try:
            prediction = pred_ans[str(query_id)]
        except:
            skip_count += 1
            if verbose:
                print("para: {}".format(para))
                print("query: {}".format(query_text))
                print("ref: {}".format('#'.join(answers)))
                print("Skipped")
                print('----------------------------')
            continue
        _f1 = calc_f1_score(answers, prediction)
        f1 += _f1
        em += calc_em_score(answers, prediction)
        if verbose:
            print("para: {}".format(para))
            print("query: {}".format(query_text))
            print("ref: {}".format('#'.join(answers)))
            print("cand: {}".format(prediction))
            print("score: {}".format(_f1))
            print('----------------------------')

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = _tokenize_chinese_chars(_normalize(ans))
        prediction_segs = _tokenize_chinese_chars(_normalize(prediction))
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        prec = 1.0 * lcs_len / len(prediction_segs)
        rec = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * prec * rec) / (prec + rec)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _normalize(ans)
        prediction_ = _normalize(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def evaluate_predictions(ref_ans, pred_ans):
    F1, EM, TOTAL, SKIP = evaluate(ref_ans, pred_ans, verbose=False)
    output_result = OrderedDict()
    output_result['F1'] = '%.3f' % F1
    output_result['EM'] = '%.3f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    return output_result


##########################评测函数结束#################################

def paragraph_selection(context, answer_text, answer_start, max_len=450):
    """
    理训练集答案开始位置过于靠后的情况。以答案为基本中心裁剪context, 并重新计算answer_start
    :param max_len: BERT输入的最大长度
    :param context: 段落文本
    :param answer_text: 答案文本
    :param answer_start: 答案开始位置
    :return: context, answer_start
    """
    standard = max_len - 30
    standard_mid = standard // 2
    if len(context) < standard:
        return context, answer_start
    answer_end = answer_start + len(answer_text)
    if answer_end < standard:
        return context, answer_start
    answer_mid = (answer_start + answer_end) // 2
    select_start = answer_mid - standard_mid
    select_end = answer_mid + standard_mid
    if select_start < 0:
        select_start = 0
    if select_end > len(context):
        select_end = len(context)
    context = context[select_start: select_end]
    answer_start = context.find(answer_text)
    if answer_start < 0:
        print(select_start, select_end, len(context))
    return context, answer_start


def load_dataset(data_path):
    """
    加载CMRC2018格式的数据集
    """
    with open(data_path) as f:
        input_data = json.load(f)['data']

    examples = []
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"].strip()
            try:
                title = paragraph["title"].strip()
            except KeyError:
                title = ''

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question = qa["question"].strip()

                is_impossible = False

                if "is_impossible" in qa.keys():
                    is_impossible = qa["is_impossible"]

                answer_starts = [answer["answer_start"] for answer in qa.get("answers", [])]
                answers = [answer["text"].strip() for answer in qa.get("answers", [])]

                if len(answer_starts) == 0 or answer_starts[0] == -1:
                    examples.append({
                        "id": qas_id,
                        "title": title,
                        "context": context,
                        "question": question,
                        "answers": answers,
                        "answer_starts": answer_starts,
                        "is_impossible": is_impossible
                    })
                else:
                    answer_start = answer_starts[0]
                    answer_text = answers[0]
                    cut_context, answer_start = paragraph_selection(context, answer_text, answer_start, max_len=450)
                    examples.append({
                        "id": qas_id,
                        "title": title,
                        "context": cut_context,
                        "question": question,
                        "answers": [answer_text],
                        "answer_starts": [answer_start],
                        "is_impossible": is_impossible
                    })
    return examples


def convert_to_features(examples, tokenizer, max_len=450, stride=128):
    """
    把文本转换为BERT的输入
    """
    questions = [examples[i]['question'] for i in range(len(examples))]
    contexts = [examples[i]['context'] for i in range(len(examples))]
    tokenized_examples = tokenizer(questions,
                                   contexts,
                                   padding="max_length",
                                   max_length=max_len,
                                   truncation="only_second",
                                   stride=stride,
                                   return_offsets_mapping=True,
                                   return_overflowing_tokens=False)

    tokenized_examples = pd.DataFrame.from_dict(tokenized_examples, orient="index").T
    tokenized_examples = tokenized_examples.to_dict(orient="records")

    for i, tokenized_example in enumerate(tqdm(tokenized_examples)):
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        offsets = tokenized_example['offset_mapping']
        sequence_ids = tokenized_example['token_type_ids']

        answers = examples[i]['answers']
        answer_starts = examples[i]['answer_starts']

        # If no answers are given, set the cls_index as answer.
        if len(answer_starts) == 0 or answer_starts[0] == -1:
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
            tokenized_examples[i]['answerable_label'] = 0
        else:
            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            end_char = start_char + len(answers[0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 2
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
                tokenized_examples[i]['answerable_label'] = 0
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples[i]["start_positions"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples[i]["end_positions"] = token_end_index + 1
                tokenized_examples[i]['answerable_label'] = 1

        tokenized_examples[i]["example_id"] = examples[i]['id']

    dataset_dict = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "start_positions": [],
                    "end_positions": [], }
    for item in tokenized_examples:
        for key in dataset_dict:
            dataset_dict[key].append(item[key])
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [dataset_dict["input_ids"], dataset_dict["token_type_ids"], dataset_dict["attention_mask"]]
    y = [dataset_dict["start_positions"], dataset_dict["end_positions"]]

    return tokenized_examples, x, y


def build_model(model_path, max_len):
    """
    建立模型
    """
    encoder = TFBertModel.from_pretrained(model_path, from_pt=True)
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    sequence_output = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(sequence_output)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(sequence_output)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax, name="start")(start_logits)
    end_probs = layers.Activation(keras.activations.softmax, name="end")(end_logits)

    bert_model = keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=[start_probs, end_probs],
                             name="BERTForQuestionAnswer")
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Nadam(lr=3e-5)
    bert_model.compile(optimizer=optimizer, loss=[loss, loss], metrics=['acc'])
    return bert_model


def TrainEMF1(eval_examples, tokenized_examples, tokenizer):
    """
    用于检验分词后的答案位置是否和原来的一样
    """
    em, f1, total_count = 0, 0, 0
    count = 0
    for idx, example in enumerate(tqdm(eval_examples, desc="Evaluation")):
        total_count += 1
        offsets = tokenized_examples[idx]['offset_mapping']
        start = tokenized_examples[idx]["start_positions"]
        end = tokenized_examples[idx]["end_positions"]
        if (start >= len(offsets) or end >= len(offsets) or offsets[start] is None or
                offsets[end] is None or offsets[start] == (0, 0) or offsets[end] == (0, 0)):
            prediction = ""
            print(start, end, offsets[start], offsets[end])
            print("".join(tokenizer.decode(tokenized_examples[idx]["input_ids"])))
            print(example["answers"])
            count += 1
        else:
            pred_char_start = offsets[start][0]
            pred_char_end = offsets[end][1]
            prediction = example["context"][pred_char_start:pred_char_end]

        answers = example["answers"]
        f1 += calc_f1_score(answers, prediction)
        em += calc_em_score(answers, prediction)
    print(count)
    f1_score = 100.0 * f1 / (total_count - 0)
    em_score = 100.0 * em / (total_count - 0)
    tqdm.write(f"F1 score={f1_score:.3f},  EM score={em_score:.3f}")


class EM_F1Score(keras.callbacks.Callback):
    """
    回调函数，每轮训练后计算EM和F1
    """

    def __init__(self, eval_x, eval_y, eval_examples, tokenized_examples):
        super().__init__()
        self.x_eval = eval_x
        self.y_eval = eval_y
        self.eval_examples = eval_examples
        self.tokenized_examples = tokenized_examples

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        em, f1, total_count = 0, 0, 0
        for idx, (start, end) in enumerate(tqdm(list(zip(pred_start, pred_end)), desc="Evaluation")):
            total_count += 1
            example = self.eval_examples[idx]
            offsets = self.tokenized_examples[idx]['offset_mapping']
            start = np.argmax(start)
            end = np.argmax(end)
            if (start >= len(offsets) or end >= len(offsets) or offsets[start] is None or
                    offsets[end] is None or offsets[start] == (0, 0) or offsets[end] == (0, 0)):
                prediction = ""
            else:
                pred_char_start = offsets[start][0]
                pred_char_end = offsets[end][1]
                prediction = example["context"][pred_char_start:pred_char_end]

            answers = example["answers"]
            f1 += calc_f1_score(answers, prediction)
            em += calc_em_score(answers, prediction)
        f1_score = 100.0 * f1 / total_count
        em_score = 100.0 * em / total_count
        logs['F1'] = f1_score
        logs['EM'] = em_score
        tqdm.write(f"epoch={epoch + 1}, F1 score={f1_score:.3f},  EM score={em_score:.3f}")


# 设置CMRC2018数据路径
train_path = './cmrc2018/cmrc2018_train.json'
eval_path = './cmrc2018/cmrc2018_dev.json'

# 设置BERT相关参数
max_len = 450
stride = 128
bert_model_path = "./chinese_roberta_wwm_ext_pytorch"
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)

# 加载训练集和验证集
train_examples = load_dataset(data_path=train_path)
train_tokenized_examples, x_train, y_train = convert_to_features(train_examples, tokenizer, max_len=max_len,
                                                                 stride=stride)
print(f"{len(train_examples)} train_examples {len(train_tokenized_examples)} train_tokenized_examples.")

eval_examples = load_dataset(data_path=eval_path)
eval_tokenized_examples, x_eval, y_eval = convert_to_features(eval_examples, tokenizer, max_len=max_len,
                                                              stride=stride)
print(f"{len(eval_examples)} eval_examples {len(eval_tokenized_examples)} eval_tokenized_examples.")

# 这里可以做训练集和验证集分词后的label正确性检验
# TrainEMF1(train_examples, train_tokenized_examples, tokenizer)
# TrainEMF1(eval_examples, eval_tokenized_examples, tokenizer)

# 训练模型 支持TPU
use_tpu = True
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = build_model(model_path=bert_model_path, max_len=max_len)
else:
    model = build_model(model_path=bert_model_path, max_len=max_len)
EMF1_Callback = EM_F1Score(x_eval, y_eval, eval_examples, eval_tokenized_examples)
model.fit(x_train, y_train, epochs=3, verbose=1, batch_size=32, callbacks=[EMF1_Callback])
