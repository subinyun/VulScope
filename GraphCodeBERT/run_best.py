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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel)
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score

from tqdm import tqdm, trange
import multiprocessing
import pandas as pd
from model import Model
from collections import Counter

cpu_cont = 16
logger = logging.getLogger(__name__)

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

# Global set to avoid duplicate idx
used_indices = set()

def get_unique_random_index(range_start=0, range_end=100000):
    while True:
        rand_idx = random.randint(range_start, range_end)
        if rand_idx not in used_indices:
            used_indices.add(rand_idx)
            return rand_idx
    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

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
             input_tokens_1,
             input_ids_1,
             position_idx_1,
             dfg_to_code_1,
             dfg_to_dfg_1,
             label,

    ):
        #The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1=position_idx_1
        self.dfg_to_code_1=dfg_to_code_1
        self.dfg_to_dfg_1=dfg_to_dfg_1
        
        #label
        self.label=label

class SlideInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 tokens_features,
                 row_num,
                 label_c):
        self.input_tokens = input_tokens
        self.tokens_features = tokens_features
        self.row_num = row_num
        self.label_c = label_c

def convert_examples_to_features(item):
    # Unpack input
    idx, func, label, tokenizer, args, cache = item
    parser = parsers['java']  # Example: assume language is Java
    
    if idx not in cache:
        logger.info(f"Processing example {idx}")
        # Extract data flow
        code_tokens, dfg = extract_dataflow(func, parser, 'java')
        code_tokens = [
            tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x)
            for idx, x in enumerate(code_tokens)
        ]
        
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        code_tokens = [y for x in code_tokens for y in x]

        # Truncate and process
        code_tokens = code_tokens[:args.code_length + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][:512 - 3]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        source_ids += [tokenizer.unk_token_id for x in dfg]
        padding_length = args.code_length + args.data_flow_length - len(source_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length

        # Reindex
        reverse_index = {x[1]: idx for idx, x in enumerate(dfg)}
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
        cache[idx] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg
    
    source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg = cache[idx]
    return InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, label)



def convert_examples_to_features_window(data):
    func, label_c, tokenizer, args = data
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

    for rows in windows:

        rows = ['\n' if x == '' else x for x in rows]
        code_tokens = [tokenizer.tokenize(x) for x in rows if tokenizer.tokenize(x) != []]
        row_idx = [[idx + 1] * len(row_token) for idx, row_token in enumerate(code_tokens)]

        code_tokens = [y for x in code_tokens for y in x]
        row_idx = [y for x in row_idx for y in x]

        args.max_source_length = 512

        # 512 토큰으로 제한
        if len(code_tokens) > args.max_source_length - 2:
            code_tokens = code_tokens[:args.max_source_length - 2]
            row_idx = row_idx[:args.max_source_length - 2]

        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        row_indices = [0] + row_idx + [row_idx[-1] + 1]
        row_num = row_indices[-1]

        # 토큰 ID 변환 및 패딩
        input_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.max_source_length - len(input_ids)
        if padding_length > 0:
            input_ids += [tokenizer.pad_token_id] * padding_length

        # 단일 feature 생성
        tokens_feature = TokenFeatures(input_ids, row_indices)
        tokens_features_list.append(tokens_feature)  # 중첩 리스트 방지

        source_tokens_list.append(source_tokens)
        row_num_list.append(row_num)

    return SlideInputFeatures(source_tokens_list, tokens_features_list, row_num_list, label_c)


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
            if file_path.endswith(".jsonl"):
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        js = json.loads(line)
                        code = js.get("func_before") or js.get("func")
                        if code is None:
                            continue
                        label = js["target"]
                        idx = js.get("idx", get_unique_random_index())
                        data.append((code, label, idx))
            else:  # assume CSV
                df = pd.read_csv(file_path)
                code_column = "func_before" if "func_before" in df.columns else "func"
                df["idx"] = df.get("idx", [get_unique_random_index() for _ in range(len(df))])
                data = list(zip(df[code_column], df["target"], df["idx"]))

            # Tokenizer 병렬 처리
            tokenizers = [tokenizer] * len(data)
            arg = [args] * len(data)
            source = [(code, label, idx, tok, a) for (code, label, idx), tok, a in zip(data, tokenizers, arg)]

            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            self.examples = pool.map(convert_examples_to_features_window, tqdm(source, total=len(source)))
            print("parse done!")

            with open(data_path, 'wb') as f:
                pickle.dump(self.examples, f)
            print("saved at %s", data_path)

    def __len__(self):
        return len(self.examples)

    def get_feature(self, feature):
        max_length = 512
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
        features = [self.get_feature(tf) for tf in self.examples[i].tokens_features]
        label = torch.tensor(self.examples[i].label_c)
        return features, label



class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        
        if file_path.endswith(".jsonl"):
            with open(file_path, 'r') as f:
                for line in f:
                    js = json.loads(line.strip())
                    js["code"] = js.get("func_before") or js.get("func")
                    if js["code"] is None:
                        continue
                    js["idx"] = js.get("idx", get_unique_random_index())
                    self.examples.append(convert_examples_to_features(js, tokenizer, args))

        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            code_column = "func_before" if "func_before" in df.columns else "func"
            if "idx" not in df.columns:
                df["idx"] = [get_unique_random_index() for _ in range(len(df))]

            for _, row in df.iterrows():
                js = {
                    "code": row[code_column],
                    "target": row["target"],
                    "idx": row["idx"]
                }
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)
     

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    
    #build dataloader
    logger.info("Creating DataLoader")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    logger.info("DataLoader created")
    
    args.max_steps=args.epochs*len( train_dataloader)
    args.save_steps=len( train_dataloader)//10
    args.warmup_steps=args.max_steps//5
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_acc=0

    model.zero_grad()
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs_ids, position_idx, attn_mask, labels = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(inputs_ids, position_idx, attn_mask, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            train_loss += loss.item()
            bar.set_description(f"Epoch {idx} Loss {train_loss / (step + 1):.4f}")
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_acc']>best_acc:
                        best_acc=results['eval_acc']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_acc,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.saved_model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

                    # 마지막 모델 체크포인트도 저장 (항상 최신 상태로 갱신)
                    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last-acc')
                    if not os.path.exists(checkpoint_last):
                        os.makedirs(checkpoint_last)
                    last_model_path = os.path.join(checkpoint_last, '{}'.format(args.saved_model_name))
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), last_model_path)
                    logger.info("Saving last model checkpoint to %s", last_model_path)

    
def evaluate(args, model, tokenizer, eval_when_training=False):
    # Build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Multi-GPU evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    all_logits = []
    y_trues = []

    for batch in eval_dataloader:
        inputs_ids, position_idx, attn_mask, labels = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logits = model(inputs_ids, position_idx, attn_mask, labels)
            eval_loss += loss.mean().item()
            all_logits.append(logits.cpu().numpy())  # Append logits
            y_trues.append(labels.cpu().numpy())  # Append true labels
        nb_eval_steps += 1
    
    # Concatenate results
    all_logits = np.concatenate(all_logits, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)

    # Predictions
    best_threshold = 0.5
    y_preds = all_logits[:, 1] > best_threshold

    # Calculate metrics
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    mcc = matthews_corrcoef(y_trues, y_preds)
    conf_matrix = confusion_matrix(y_trues, y_preds)
    TN, FP, FN, TP = conf_matrix.ravel()
    IoU = TP / (TP + FP + FN)
    
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_acc": float(acc),
        "mcc": float(mcc),
        "eval_IoU": float(IoU),
        "eval_threshold": best_threshold,
        "eval_TP": int(TP),
        "eval_TN": int(TN),
        "eval_FP": int(FP),
        "eval_FN": int(FN),
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result



import numpy as np
from sklearn.metrics import roc_curve
from simmodel import SimilarityModel
import csv
import copy
import numpy

def test(args, model, tokenizer, eval_dataset, eval_when_training=False, best_threshold=0.5):
    config_class = RobertaConfig
    model_class = RobertaModel
    tokenizer_class = RobertaTokenizer
    config = config_class.from_pretrained("microsoft/codebert-base", cache_dir=None)
    config.num_labels = 1
    sim_tokenizer = tokenizer_class.from_pretrained("roberta-base",
                                                do_lower_case=False,
                                                cache_dir=None)
    block_size = 512
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
    similarity_db_path = args.similarity_db_path

    if os.path.exists(similarity_db_path):
        database, compared_list, compared_tensor, database_vec = pickle.load(open(similarity_db_path, "rb"))

    else:
        entries = []
        with open(args.train_data_file, 'r') as f:
            for line in f:
                js = json.loads(line.strip())
                code = js.get("func_before") or js.get("func")
                if code is not None:
                    entries.append(code)


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

        pickle.dump((database, compared_list, compared_tensor, database_vec), open(similarity_db_path, "wb"))

    # build dataloader
    if args.local_rank != -1:
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
        args.eval_batch_size = int(args.eval_batch_size / torch.distributed.get_world_size())
    else:
        eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0, collate_fn=lambda x: x)

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    all_logits = []
    y_trues = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader), position=0)

    for batch in bar:
        batch_logits = []
        batch_labels = []

        for features, label in batch:
            logits_list = []
            label = label.unsqueeze(0).to(args.device)

            for feature in features:
                input_ids, position_idx, attention_mask, row2row_mask, row_mask, bool_mask = feature
                input_ids = input_ids.unsqueeze(0).to(args.device)
                position_idx = position_idx.unsqueeze(0).to(args.device)
                attention_mask = attention_mask.unsqueeze(0).to(args.device)
    

                with torch.no_grad():
                    loss, logits = model(
                        inputs_ids=input_ids,
                        position_idx=position_idx,
                        attn_mask=attention_mask,
                        labels=label
                    )

                    # eval_loss += loss.mean().item()
                    current_logits = logits.cpu().numpy()
                
                # 현재 배치의 예측값 계산
                batch_preds = np.argmax(current_logits, axis=1)  # 각 샘플에 대한 예측 클래스
                positive_indices = np.where(batch_preds == 1)[0]  # positive로 예측된 인덱스들
                
                # Similarity check and adjustment
                if len(positive_indices) > 0:  # positive 예측이 있는 경우
                    for tem_idx in positive_indices:
                        compare_database = copy.deepcopy(database_vec)
                        # input_ids에서 해당 인덱스의 데이터 사용
                        compare_now = input_ids[tem_idx].unsqueeze(0).to(args.device)
                        vec = smodel.encoder(compare_now, attention_mask=attention_mask[tem_idx].unsqueeze(0).to(args.device))
                        if len(vec) > 1:
                            vec = vec[1]
                        else:
                            vec = vec[0][:, 0, :]
                        vec_now = vec.detach().cpu().numpy()

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
                            vec_patched = vec.detach().cpu().numpy()
                            scores = np.matmul(vec_now, vec_patched.T)
                            score_patched = scores[0][0]
                            if score_patched > score_vul:
                                current_logits[tem_idx] -= 0.32
                
            all_logits.append(current_logits)
            y_trues.append(np.expand_dims(label.cpu().numpy(), axis=0))

        nb_eval_steps += 1

    bar.close()

    # 최적 임계값 찾기
    all_logits = np.concatenate(all_logits, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    print(f"all_logits shape: {all_logits.shape}")  # 디버깅 출력
    # fpr, tpr, thresholds = roc_curve(y_trues, all_logits)  # all_logits 자체 사용
    # best_idx = np.argmax(tpr - fpr)
    # best_threshold = thresholds[best_idx]
    best_threshold = 0.5

    # 최적 임계값으로 예측
    y_preds = (all_logits[:, 1] > best_threshold).astype(int)
    y_trues = y_trues.ravel()  # or y_trues.flatten()

    IoU = (y_preds & y_trues).sum() / (y_preds | y_trues).sum() if (y_preds | y_trues).sum() > 0 else 0
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds, zero_division=1)
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

    ## Required parameters
    parser.add_argument("--train_data_file", type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--saved_model_name", default="model.bin", type=str,
                        help="saved_model name")

    ## Other parameters
    parser.add_argument("--eval_data_file", type=str, required=True,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                   
    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=128, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    parser.add_argument('--ws', type=int, default=90,
                        help="window size")

    parser.add_argument('--ov', type=int, default=10,
                        help="overlap")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--similarity_db_path', type=str, required=True,
                        help="Path to the similarity database file (e.g., .pkl)")


    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)


    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels=1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)    

    model=Model(model,config,tokenizer,args)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        logger.info("Starting training process")
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
        logger.info(f"Loaded training dataset with {len(train_dataset)} examples")
        train(args, train_dataset, model, tokenizer)
        logger.info("Finished training process")

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-acc'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
        output_dir = os.path.join(output_dir, '{}'.format(args.saved_model_name)) 
        model.load_state_dict(torch.load(output_dir))      
        logger.info("Loading model checkpoint to %s", output_dir)

        model.to(args.device)
        result=evaluate(args, model, tokenizer)

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
        
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
        output_dir = os.path.join(output_dir, '{}'.format(args.saved_model_name)) 
        model.load_state_dict(torch.load(output_dir))  
        model.to(args.device)
        test_dataset = SlideDataset(tokenizer, args, file_path=args.test_data_file,
                                    file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)

    return results


if __name__ == "__main__":
    main()

