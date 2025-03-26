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

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import pandas as pd
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, RobertaModel, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

# Global set to avoid duplicate idx
used_indices = set()

def get_unique_random_index(range_start=0, range_end=100000):
    while True:
        rand_idx = random.randint(range_start, range_end)
        if rand_idx not in used_indices:
            used_indices.add(rand_idx)
            return rand_idx

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label

def load_data(file_path):
    data = []
    if file_path.endswith(".jsonl"):
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                data.append(js)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        data = df.to_dict(orient='records')
    else:
        raise ValueError("Unsupported file format. Use .jsonl or .csv")
    return data

def normalize_code(js):
    if 'func' in js:
        code = js['func']
    elif 'func_before' in js:
        code = js['func_before']
    else:
        raise ValueError("Missing both 'func' and 'func_before'")

    label = js['target']
    idx = js.get('idx', get_unique_random_index())

    return code, label, idx

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

class TokenFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids,
                 row_idx):
        self.input_ids = input_ids
        self.row_idx = row_idx

def convert_examples_to_features(js,tokenizer,args):
    #source
    code, label, idx = normalize_code(js)
    code = code.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,idx,js['target'])


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
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_acc=0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    # Initialize early stopping parameters at the start of training
    early_stopping_counter = 0
    best_loss = None
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)        
            labels=batch[1].to(args.device) 
            model.train()
            loss,logits = model(inputs,labels)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer,eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))                    
                        # Save model checkpoint
                        
                    if results['eval_acc']>best_acc:
                        best_acc=results['eval_acc']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best acc:%s",round(best_acc,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.saved_model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

        # Calculate average loss for the epoch
        avg_loss = train_loss / tr_num

        # Check for early stopping condition
        if args.early_stopping_patience is not None:
            if best_loss is None or avg_loss < best_loss - args.min_loss_delta:
                best_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    logger.info("Early stopping")
                    break  # Exit the loop early
                        



def evaluate(args, model, tokenizer,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5
    eval_acc=np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    TN, FP, FN, TP = conf_matrix.ravel()
    IoU = (TP) / (TP + FP + FN)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_IoU": float(IoU),
        "eval_mcc": float(mcc),
        "eval_TP": int(TP),
        "eval_TN": int(TN),
        "eval_FP": int(FP),
        "eval_FN": int(FN)
    }
    return result
  
import numpy as np
from sklearn.metrics import roc_curve
from simmodel import SimilarityModel
import numpy
import csv
import copy

def get_ids(src, block_size, tokenizer):
    code = ' '.join(src.replace('\\n', '\n').replace('\\t', '\t').split())
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    return source_ids

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

    nb_eval_steps = 0
    model.eval()
    all_logits = []
    y_trues = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader), position=0)

    for batch in bar:
        for features, label in batch:
            label = label.to(args.device)

            for feature in features:
                input_ids, attention_mask, row2row_mask, row_mask, bool_mask = feature
                input_ids = input_ids.unsqueeze(0).to(args.device)
                attention_mask = attention_mask.unsqueeze(0).to(args.device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids.to(args.device),
                    )
                    logits = outputs

                current_logits = logits.cpu().numpy()
                
                batch_preds = np.argmax(current_logits, axis=1) 

                positive_indices = np.where(batch_preds == 1)[0] 
                
                # Similarity check and adjustment
                if len(positive_indices) > 0:  
                    for tem_idx in positive_indices:
                        compare_database = copy.deepcopy(database_vec)
          
                        compare_now = input_ids[tem_idx].unsqueeze(0).to(args.device)
                        vec = smodel.encoder(compare_now, attention_mask=attention_mask[tem_idx].unsqueeze(0).to(args.device))
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
                                current_logits[tem_idx] -= 0.32
                
            all_logits.append(current_logits)
            y_trues.append(np.expand_dims(label.cpu().numpy(), axis=0))

        nb_eval_steps += 1

    bar.close()

    all_logits = np.concatenate(all_logits, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    print(f"all_logits shape: {all_logits.shape}")  
    fpr, tpr, thresholds = roc_curve(y_trues, all_logits)  
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]

    y_preds = (all_logits > best_threshold).astype(int) 

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
    parser.add_argument("--output_dir", default="./saved_models/codebert", type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--saved_model_name", default="model.bin", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", type=str, required=True,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

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
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=5,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')

    parser.add_argument('--ws', type=int,
                        help="window size")

    parser.add_argument('--ov', type=int,
                        help="overlap")
    parser.add_argument('--similarity_db_path', type=str, required=True,
                        help="Path to the similarity database file (e.g., .pkl)")

    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)



    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
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
            
    if args.do_test and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-acc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            output_dir = os.path.join(output_dir, '{}'.format(args.saved_model_name)) 
            model.load_state_dict(torch.load(output_dir))                  
            model.to(args.device)
            logger.info("Training/evaluation parameters %s", args)
            test_file_path = args.test_data_file
            test_dataset = SlideDataset(tokenizer, args, file_path=test_file_path,
                                    file_type='test')
            test(args, model, tokenizer, test_dataset, best_threshold=0.5)

    return results


if __name__ == "__main__":
    main()


