# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import logging
import multiprocessing
import math
import os
import csv

import sys
import copy
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy
import pickle
import random
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
# model reasoning
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix
# word-level tokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from Models.StagedModel_vul import Model

from simmodel import SimilarityModel

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertModel, BertTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)


def get_ids(src, block_size, tokenizer):
    code = ' '.join(src.replace('\\n', '\n').replace('\\t', '\t').split())
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    return source_ids

class TokenFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids,
                 row_idx):
        self.input_ids = input_ids
        self.row_idx = row_idx


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 tokens_features,
                 row_num,
                 input_ids_sim,
                 label_c,
                 label_f):
        self.input_tokens = input_tokens
        self.tokens_features = tokens_features
        self.row_num = row_num
        self.input_ids_sim = input_ids_sim
        self.label_c = label_c
        self.label_f = label_f

def convert_examples_to_features(data):
    func, label_c, label_f, tokenizer, args = data

    # Tokenize and prepare code tokens
    rows = str(func).split('\n')
    rows = ['\n' if x == '' else x for x in rows]
    code_tokens = [tokenizer.tokenize(x) for x in rows if tokenizer.tokenize(x) != []]
    row_idx = [[idx + 1] * len(row_token) for idx, row_token in enumerate(code_tokens)]

    code_tokens = [y for x in code_tokens for y in x]
    row_idx = [y for x in row_idx for y in x]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]

    tokens_features = []
    start_idx = 0
    seg_num = math.ceil((len(code_tokens) + 1) / args.block_size)
    seg_num = min(seg_num, args.seg_num)
    row_num = 0

    for i in range(seg_num):
        end_token_idx = start_idx + args.block_size - 2
        if end_token_idx < len(code_tokens) - 1:
            last_row = row_idx[end_token_idx]
            end_idx = row_idx.index(last_row)
        else:
            end_idx = len(code_tokens)

        seq_source_tokens = [tokenizer.cls_token] + code_tokens[start_idx:end_idx]
        seq_row_indices = [0] + (np.array(row_idx[start_idx:end_idx]) - row_idx[start_idx] + 1).tolist()
        row_num += seq_row_indices[-1]

        seq_input_ids = tokenizer.convert_tokens_to_ids(seq_source_tokens)
        padding_length = args.block_size - len(seq_input_ids)
        seq_input_ids += [tokenizer.pad_token_id] * padding_length

        tokens_feature = TokenFeatures(seq_input_ids, seq_row_indices)
        tokens_features.append(tokens_feature)

        start_idx = end_idx

    padding_length = args.seg_num - len(tokens_features)
    tokens_features += [None] * padding_length

    if isinstance(label_f, str):
        label_f = label_f.split(',')
        label_f = [(int(x) + 1) for x in label_f]

    return InputFeatures(source_tokens, tokens_features, row_num, None, label_c, label_f)

# Evaluation을 위한 window 기반 convert_examples_to_features 유지 
def convert_examples_to_features_window(data):
    func, label_c, label_f, tokenizer, tokenizer_sim, args = data
    # source
    all_rows = str(func).split('\n')

    windows = []
    len_rows = len(all_rows)
    max_line_length = args.ws
    overlap = args.ov

    if len_rows > args.ws:

        partitions = []
        process_index = 0

        while True:
            if process_index + max_line_length > len_rows:
                partitions.append([process_index, len_rows])
                break
            else:
                partitions.append([process_index, process_index + max_line_length])
                process_index += max_line_length - overlap

        for partition in partitions:
            windows.append(all_rows[partition[0]:partition[1]])

    else:
        windows.append(all_rows)

    source_tokens_list = []
    tokens_features_list = []
    row_num_list = []
    input_ids_sim_list = []

    for rows in windows:
        raw_code = '\n'.join(rows)
        input_ids_sim_list.append(get_ids(raw_code, 400, tokenizer_sim))

        rows = ['\n' if x == '' else x for x in rows]
        code_tokens = [tokenizer.tokenize(x) for x in rows if tokenizer.tokenize(x) != []]
        row_idx = [[idx + 1] * len(row_token) for idx, row_token in enumerate(code_tokens)]

        code_tokens = [y for x in code_tokens for y in x]  # 平铺token_list
        row_idx = [y for x in row_idx for y in x]

        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]

        # # 截断
        # code_tokens = code_tokens[:args.block_size - 2]
        # row_idx = row_idx[:args.block_size - 2]
        #
        # # <cls>(第0行) + token(1~n行) + <eos>(第n+1行)
        # code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        # row_idx = [0] + row_idx + [row_idx[-1] + 1]
        # # token_id
        # source_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        # # 填充
        # padding_length = args.block_size - len(source_ids)
        # source_ids += [tokenizer.pad_token_id] * padding_length
        #
        tokens_features = []
        # row_num = row_idx[-2]
        start_idx = 0
        seg_num = math.ceil((len(code_tokens) + 1) / args.block_size)
        seg_num = min(seg_num, args.seg_num)
        row_num = 0

        for i in range(seg_num):
            end_token_idx = start_idx + args.block_size - 2
            if end_token_idx < len(code_tokens) - 1:
                last_row = row_idx[end_token_idx]
                end_idx = row_idx.index(last_row)
            else:
                end_idx = len(code_tokens)

            seq_source_tokens = [tokenizer.cls_token] + code_tokens[start_idx:end_idx]
            seq_row_indices = [0] + (np.array(row_idx[start_idx:end_idx]) - row_idx[start_idx] + 1).tolist()
            row_num += seq_row_indices[-1]

            seq_input_ids = tokenizer.convert_tokens_to_ids(seq_source_tokens)

            padding_length = args.block_size - len(seq_input_ids)
            seq_input_ids += [tokenizer.pad_token_id] * padding_length

            tokens_feature = TokenFeatures(seq_input_ids, seq_row_indices)
            tokens_features.append(tokens_feature)

            start_idx = end_idx

        padding_length = args.seg_num - len(tokens_features)
        tokens_features += [None] * padding_length

        source_tokens_list.append(source_tokens)
        tokens_features_list.append(tokens_features)
        row_num_list.append(row_num)

    # label_f
    if isinstance(label_f, str):
        label_f = label_f.split(',')
        label_f = [(int(x) + 1) for x in label_f]

    InputFeaturesList = []
    for ind, source_tokens in enumerate(source_tokens_list):
        InputFeaturesList.append(InputFeatures(source_tokens, tokens_features_list[ind], row_num_list[ind], input_ids_sim_list[ind], label_c, label_f))

    return InputFeaturesList

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='', file_type="train"):
        self.args = args
        data_path = os.path.join(os.path.dirname(file_path), file_type + "_lp.pkl")

        if file_path.endswith(".csv"):
            data_path = file_path[:-4]
        data_path += "_lp_ws_" + str(args.ws) + "_ov_"+ str(args.ov) + ".pkl"

        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                self.examples = pickle.load(f)
                print("loaded at %s", data_path)
        else:
            self.examples = []
            df = pd.read_csv(file_path)
            funcs = df["func"].tolist()
            labels_c = df["target"].tolist()
            labels_f = df["target"].tolist()
            tokenizers = [tokenizer] * len(funcs)

            arg = [args] * len(funcs)
            source = list(zip(funcs, labels_c, labels_f, tokenizers, arg))
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            self.examples = pool.map(convert_examples_to_features, tqdm(source, total=len(source)))
            #for i in tqdm(range(len(funcs))):
            # self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, args))
            print("parse done!")

            with open(data_path, 'wb') as f:
                pickle.dump(self.examples, f)
            print("saved at %s", data_path)

        '''if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label_c))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".
                            format(' '.join(map(str, tokenizer.convert_tokens_to_ids(example.input_tokens)))))'''

    def __len__(self):
        return len(self.examples)

    def get_feature(self, feature):
        max_length = self.args.block_size
        if feature is None:
            return (torch.ones(max_length).long(),
                    torch.zeros(max_length, max_length).bool(),
                    torch.zeros(max_length, max_length),
                    torch.zeros(max_length).bool(),
                    torch.zeros(1).bool())
        token_row_idx = feature.row_idx
        row_token_nums = Counter(token_row_idx)
        row_idx = [np.where(np.array(token_row_idx) == x)[0][0] for x in row_token_nums.keys()] + [len(token_row_idx)]
        row_num = len(row_token_nums)

        # self-attention mask
        attn_mask = torch.zeros(max_length, max_length)
        attn_mask[:len(token_row_idx), :len(token_row_idx)] = 1
        attn_mask = attn_mask.bool()

        # token行坐标
        row2row_mask = torch.zeros(max_length, max_length)
        for idx in range(row_num):
            row2row_mask[row_idx[idx]:row_idx[idx + 1], row_idx[idx]:row_idx[idx + 1]] = 1

        row2row_mask[0, :len(token_row_idx)] = 1

        # 每行第一个token位置
        row_mask = torch.zeros(max_length)
        row_mask[row_idx[:-1]] = 1
        row_mask = row_mask.bool()

        return (torch.tensor(feature.input_ids),
                attn_mask,
                row2row_mask,
                row_mask,
                torch.ones(1).bool())

    def __getitem__(self, i):
        max_length = self.args.block_size

        row_num = self.examples[i].row_num + 1

        # label_f padding to max length
        label_f = torch.zeros(row_num)
        if isinstance(self.examples[i].label_f, list):
            vul_idx = [x for x in self.examples[i].label_f if x < row_num]
            label_f[vul_idx] = 1
        label_f = label_f.tolist()
        label_f += [-1] * (max_length - row_num)

        a = [self.get_feature(self.examples[i].tokens_features[j]) for j in range(self.args.seg_num)]
        b = torch.tensor(self.examples[i].label_c)
        c = torch.tensor(label_f)

        return a, b, c
    

# SlideDataset 클래스 - Evaluation용
class SlideDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='', file_type="train"):
        self.args = args

        data_path = os.path.join(os.path.dirname(file_path), file_type + "_lp.pkl")

        if file_path.endswith(".csv"):
            data_path = file_path[:-4]
        data_path += "_lp_ws_" + str(args.ws) + "_ov_"+ str(args.ov) + ".pkl"

        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                self.examples = pickle.load(f)
                print("loaded at %s", data_path)
        else:
            self.examples = []
            df = pd.read_csv(file_path)
            funcs = df["func"].tolist()
            labels_c = df["target"].tolist()
            labels_f = df["target"].tolist()
            tokenizers = [tokenizer] * len(funcs)
            tokenizer_sim = RobertaTokenizer.from_pretrained("roberta-base",
                                                                do_lower_case=False,
                                                                cache_dir=None)
            tokenizers_sim = [tokenizer_sim] * len(funcs)
            arg = [args] * len(funcs)
            source = list(zip(funcs, labels_c, labels_f, tokenizers, tokenizers_sim, arg))
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            self.examples = pool.map(convert_examples_to_features_window, tqdm(source, total=len(source)))
            print("parse done!")

            with open(data_path, 'wb') as f:
                pickle.dump(self.examples, f)
            print("saved at %s", data_path)

    def __len__(self):
        return len(self.examples)

    def get_feature(self, feature):
        max_length = self.args.block_size
        if feature is None:
            return (torch.ones(max_length).long(),
                    torch.zeros(max_length, max_length).bool(),
                    torch.zeros(max_length, max_length),
                    torch.zeros(max_length).bool(),
                    torch.zeros(1).bool())
        token_row_idx = feature.row_idx
        row_token_nums = Counter(token_row_idx)
        row_idx = [np.where(np.array(token_row_idx) == x)[0][0] for x in row_token_nums.keys()] + [len(token_row_idx)]
        row_num = len(row_token_nums)

        # self-attention mask
        attn_mask = torch.zeros(max_length, max_length)
        attn_mask[:len(token_row_idx), :len(token_row_idx)] = 1
        attn_mask = attn_mask.bool()

        # token行坐标
        row2row_mask = torch.zeros(max_length, max_length)
        for idx in range(row_num):
            row2row_mask[row_idx[idx]:row_idx[idx + 1], row_idx[idx]:row_idx[idx + 1]] = 1

        row2row_mask[0, :len(token_row_idx)] = 1

        # 每行第一个token位置
        row_mask = torch.zeros(max_length)
        row_mask[row_idx[:-1]] = 1
        row_mask = row_mask.bool()

        return (torch.tensor(feature.input_ids),
                attn_mask,
                row2row_mask,
                row_mask,
                torch.ones(1).bool())

    def __getitem__(self, i):
        max_length = self.args.block_size
        a_list = []
        c_list = []
        d_list = []
        for examples_i in self.examples[i]:

            row_num = examples_i.row_num + 1

            # label_f padding to max length
            label_f = torch.zeros(row_num)
            if isinstance(examples_i.label_f, list):
                vul_idx = [x for x in examples_i.label_f if x < row_num]
                label_f[vul_idx] = 1
            label_f = label_f.tolist()
            label_f += [-1] * (max_length - row_num)

            a = [self.get_feature(examples_i.tokens_features[j]) for j in range(self.args.seg_num)]
            a_list.append(a)
            b = torch.tensor(examples_i.label_c)
            c = torch.tensor(label_f)
            c_list.append(c)
            d_list.append(examples_i.input_ids_sim)

        return a_list, b, c_list, d_list


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader) // 2
    args.warmup_steps = args.max_steps // 5
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mcc = 0.0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader), position=0)
        tr_num = 0
        train_loss = 0
        train_acc = 0
        for step, batch in enumerate(bar):
            (labels, _) = [x.to(args.device) for x in batch[1:]]
            tokens_features = [[y.to(args.device) for y in x] for x in batch[0]]
            model.train()
            loss, logits, row_mask, _ = model(tokens_features, labels=labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            _, predicted = torch.max(logits[:, 0, :].data, 1)
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()
            train_acc += len(labels[predicted == labels]) / len(labels)

            avg_acc = round(train_acc / tr_num, 5)

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {} acc {}".format(idx, avg_loss, avg_acc))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    logger.info("***** Epoch {} Running evaluation *****".format(idx))
                    results = validate(args, model, tokenizer, eval_dataset, eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_mcc'] >= best_mcc:
                        best_mcc = results['eval_mcc']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best mcc:%s", round(best_mcc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

def validate(args, model, tokenizer, eval_dataset, eval_when_training=False, best_threshold=0.5):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader), position=0)
    for batch in bar:
        (labels, _) = [x.to(args.device) for x in batch[1:]]
        tokens_features = [[y.to(args.device) for y in x] for x in batch[0]]
        with torch.no_grad():
            loss, logit, row_mask, _ = model(tokens_features, labels=labels)
            eval_loss += loss.mean().item()
            logits.append(logit[:, 0, :].cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    bar.close()

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    IoU = (y_preds & y_trues).sum() / (y_preds | y_trues).sum()
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    mcc = matthews_corrcoef(y_trues, y_preds)
    conf_matrix = confusion_matrix(y_trues, y_preds)
    TN, FP, FN, TP = conf_matrix.ravel()

    result = {
        "eval_accuracy": float(acc),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_IoU": float(IoU),
        "eval_threshold": best_threshold,
        "eval_mcc": mcc,
        "eval_TP": TP,
        "eval_TN": TN,
        "eval_FP": FP,
        "eval_FN": FN
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result



def evaluate(args, model, tokenizer, eval_dataset, train_file_path, eval_when_training=False, best_threshold=0.5):

    config_class = RobertaConfig
    model_class = RobertaModel
    tokenizer_class = RobertaTokenizer
    config = config_class.from_pretrained("microsoft/codebert-base", cache_dir=None)
    config.num_labels = 1
    sim_tokenizer = tokenizer_class.from_pretrained("roberta-base",
                                                do_lower_case=False,
                                                cache_dir=None)
    block_size = 400
    block_size = min(block_size, sim_tokenizer.max_len_single_sentence)
    smodel = model_class.from_pretrained("microsoft/codebert-base",
                                        from_tf=False,
                                        config=config,
                                        cache_dir=None)

    smodel = SimilarityModel(smodel, config, sim_tokenizer)

    if args.n_gpu > 1:
        smodel = torch.nn.DataParallel(smodel)
    smodel.to(args.device)

    checkpoint_prefix = './../../checkpoint-best-map/model.bin'
    output_dir = os.path.join('./', '{}'.format(checkpoint_prefix))
    smodel.load_state_dict(torch.load(output_dir), strict=False)
    smodel.eval()

    if os.path.exists("./../../checkpoint-best-map/database_5.pkl"):
        database, compared_list, compared_tensor, database_vec = pickle.load(open("./../../checkpoint-best-map/database_5.pkl", "rb"))

    else:
        with open("./../../paired_primevul/range_10/train.csv", 'r') as f:
            reader = csv.reader(f)
            entries = []
            for row in reader:
                entries.append(row[0])

        stored_func = None
        stored_entry = None
        database = []
        for entry in tqdm(entries[1:]):
            func_name = entry.split("(")[0]
            new_entry = get_ids(entry, block_size, sim_tokenizer)

            if func_name == stored_func:
                database.append([stored_entry, new_entry])
                stored_func = None
                stored_entry = None
            else:
                if stored_func != None:
                    database.append([stored_entry, None])
                stored_func = func_name
                stored_entry = new_entry

        compared_list = []
        for dat in database:
            compared_list.append(dat[0])
        compared_tensor = torch.tensor(compared_list)

        batch_size = 64
        database_vec = []
        for cur_compare in tqdm(torch.split(compared_tensor, batch_size)):
            cur_compare = cur_compare.to(args.device)
            with torch.no_grad():
                vec = smodel.encoder(cur_compare, attention_mask=cur_compare.ne(1))
                if len(vec) > 1:
                    vec = vec[1]
                else:
                    vec = vec[0][:, 0, :]
                vec = vec.cpu().numpy()
                database_vec.append(vec)
        database_vec = numpy.vstack(database_vec)

        pickle.dump((database, compared_list, compared_tensor, database_vec), open("./../../checkpoint-best-map/database_5.pkl", "wb"))

    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0, collate_fn=lambda x: x)

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    model.eval()
    y_preds = []
    y_trues = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader), position=0)
    for batch in bar:
        labels = [x[1].to(args.device) for x in batch]

        for ind, x in enumerate(batch): # each batch
            tokens_features = []
            for seg in range(args.seg_num):
                len_feat = 5
                mixed_features = []
                for feat in range(len_feat):
                    cur_feat = []
                    for ind2, y in enumerate(x[0]): # y is one of the windows
                        cur_feat.append(y[seg][feat].to(args.device))
                    cur_feat = torch.stack(cur_feat, dim=0)
                    mixed_features.append(cur_feat)
                tokens_features.append(mixed_features)

            cur_label = (torch.stack([labels[ind]] * len(x[0]), dim=0))
            with torch.no_grad():
                loss, logit, row_mask, _ = model(tokens_features, labels=cur_label)
                eval_loss += loss.mean().item()
                logits = logit[:, 0, 1].cpu().numpy().tolist()

            temp_list = []
            for log in logits:
                temp_list.append(log > best_threshold)

            with torch.no_grad():
                pre_result = any(temp_list)
                if pre_result:
                    copy_temp_list = copy.deepcopy(temp_list)
                    for tem_idx, tem in enumerate(copy_temp_list):
                        if tem:
                            compare_database = copy.deepcopy(database_vec)
                            compare_now = torch.tensor([x[3][tem_idx]]).to(args.device)
                            vec = smodel.encoder(compare_now, attention_mask=compare_now.ne(1))
                            if len(vec) > 1:
                                vec = vec[1]
                            else:
                                vec = vec[0][:, 0, :]
                            vec_now = vec.cpu().numpy()

                            scores = np.matmul(vec_now, compare_database.T)
                            score_max_ind = np.argmax(scores, axis=1).item()
                            score_vul = scores[0][score_max_ind]
                            compare_patched = database[score_max_ind][1]
                            if compare_patched:
                                compare_tensor = torch.tensor([compare_patched]).to(args.device)
                                vec = smodel.encoder(compare_tensor, attention_mask=compare_tensor.ne(1))
                                if len(vec) > 1:
                                    vec = vec[1]
                                else:
                                    vec = vec[0][:, 0, :]
                                vec_patched = vec.cpu().numpy()
                                scores = np.matmul(vec_now, vec_patched.T)
                                score_patched = scores[0][0]
                                if score_patched > score_vul:
                                    logits[tem_idx] -= 0.32

            temp_list = []
            for log in logits:
                temp_list.append(log > best_threshold)
            # print("3", len(temp_list))
            y_preds.append(any(temp_list))
            y_trues.append(labels[ind].cpu().numpy().tolist())
        nb_eval_steps += 1
    bar.close()

    # calculate scores
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    IoU = (y_preds & y_trues).sum() / (y_preds | y_trues).sum()
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    mcc = matthews_corrcoef(y_trues, y_preds)
    conf_matrix = confusion_matrix(y_trues, y_preds)
    TN, FP, FN, TP = conf_matrix.ravel()

    result = {
        "eval_accuracy": float(acc),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_IoU": float(IoU),
        "eval_threshold": best_threshold,
        "eval_mcc": mcc,
        "eval_TP": TP,
        "eval_TN": TN,
        "eval_FP": FP,
        "eval_FN": FN
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    print(result)

    return result


def main():
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument("--train_data_file", default="./../../paired_primevul/range_10/train.csv", type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default="../new_model/vul/", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs "
                             "(take into account special tokens).")
    parser.add_argument("--eval_data_file", default="./../../paired_primevul/range_10/val.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="./../../paired_primevul/original/test.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name", default="model_2048_new.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default="../resource/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="../resource/codebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="../resource/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--seg_num", default=4, type=int,
                        help="Optional numbers of Code segment.")

    parser.add_argument("--do_train", default=False, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", default=True, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=12345,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=30,
                        help="training epochs")

    parser.add_argument('--ws', type=int, default=45,
                        help="window size")

    parser.add_argument('--ov', type=int, default=5,
                        help="overlap")
                        
    parser.add_argument('--bt', type=float, default=0.5,
                        help="best threshold")

    # num of attention heads
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")
    # initialize the model
    parser.add_argument("--initialize_model_name", default="../resource/staged-models/pretrain_model.bin",
                        type=str,
                        help="initialize the model")
    args = parser.parse_args()

    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    log_name = "log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    logging.basicConfig(filename="../log/vul/" + log_name,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    config.num_hidden_layers = 6
    # config.hidden_dropout_prob = 0
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model_fine = RobertaForSequenceClassification(config=config)
    model_coarse = RobertaForSequenceClassification(config=config)
    model = Model(model_fine, model_coarse, config, tokenizer, args)

    # Training
    if args.do_train:
        if not args.use_non_pretrained_model:
            initialize_model_name = args.initialize_model_name
            state_dict = torch.load(initialize_model_name)
            for layer_name, params in state_dict.items():
                if layer_name in model.state_dict():
                    model.state_dict()[layer_name].copy_(params)
        nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("The number of parameter is {}".format(nums))
        logger.info("Training/evaluation parameters %s", args)

        training_file_path = args.train_data_file
        eval_file_path = args.eval_data_file
        train_dataset = TextDataset(tokenizer, args, file_path=training_file_path, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_path=eval_file_path, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset)

    # Evaluation
    results = {}
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
        model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)
        test_file_path = args.test_data_file
        test_dataset = SlideDataset(tokenizer, args, file_path=test_file_path,
                                   file_type='test')
        evaluate(args, model, tokenizer, test_dataset, train_file_path=args.train_data_file, best_threshold=args.bt)
    return results

if __name__ == "__main__":
    main()