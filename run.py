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
import sys 
import argparse
import logging
import os
import pickle
import random
from itertools import cycle
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import json
import numpy as np
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
import utils

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,
                 idx
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.idx = idx
        
def convert_examples_to_features(js,tokenizer,args,idx):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"],idx)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js) 

        for idx, js in enumerate(data):
            self.examples.append(convert_examples_to_features(js,tokenizer,args,idx))
                
        # if "train" in file_path:
        #     for idx, example in enumerate(self.examples[:3]):
        #         logger.info("*** Example ***")
        #         logger.info("idx: {}".format(idx))
        #         logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
        #         logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
        #         logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
        #         logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids), self.examples[i].idx)         

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, 
                                                            rank=global_rank, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4)
    
    if args.distributed:
        dist.barrier()
        
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # # get optimizer and scheduler
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)
    
    # Train!
    if utils.is_main_process(): 
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        # logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
        logger.info("  Total train batch size  = %d", args.train_batch_size)
        logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    model.train()

    # set early stop
    patience = 0
    patience_threshold = 3

    tr_num,tr_loss,best_mrr = 0,0,0 
    for epoch in range(args.num_train_epochs): 
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        
        for step,batch in enumerate(train_dataloader):
            # get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            idx = batch[2].to(args.device)

            if epoch>0:
                alpha = args.alpha
            else:
                alpha = args.alpha*min(1, step/len(train_dataloader))

            loss = model(code_inputs, nl_inputs, alpha=alpha, idx=idx, 
                         no_match=args.no_match, match_queue=args.match_queue)                  

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                if utils.is_main_process():
                    logger.info("epoch {} step {} loss {}".format(epoch,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        # evaluate
        if utils.is_main_process():  
            results = evaluate(args, model, tokenizer,args.eval_data_file, eval_when_training=True)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value,4))    
            
            # save best model
            if results['eval_mrr']>best_mrr:
                best_mrr = results['eval_mrr']
                logger.info("  "+"*"*20)  
                logger.info("  Best mrr:%s",round(best_mrr,4))
                logger.info("  "+"*"*20)                          

                checkpoint_prefix = 'checkpoint-best-mrr'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)

                # results = evaluate(args, model, tokenizer,args.test_data_file, eval_when_training=True)
                # logger.info("-------------TEST RESULT------------:")
                # for key, value in results.items():
                #     logger.info("  %s = %s", key, round(value,3))

                patience = 0
            else:
                patience += 1
        if args.distributed:
            dist.barrier()     
            torch.cuda.empty_cache()
        
        # flag = torch.tensor(patience, dtype=torch.int32).to(args.device)
        # # 使用 `dist.all_reduce` 函数将所有进程的 `flag` 值相加
        # dist.all_reduce(flag, op=dist.ReduceOp.SUM)
        # # 如果 `flag` 的值等于进程的数量，那么所有进程都满足条件，可以停止训练
        # if flag.item() == patience_threshold:
        #     break
        if patience == patience_threshold:
            break

def evaluate(args, model, tokenizer,file_name,eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    
    # Eval!
    if utils.is_main_process():
        logger.info("***** Running evaluation *****")
        logger.info("  Num queries = %d", len(query_dataset))
        logger.info("  Num codes = %d", len(code_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    model = model.module if hasattr(model,'module') else model

    code_vecs = [] 
    nl_vecs = []

    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_output = model.encoder(nl_inputs,attention_mask=nl_inputs.ne(1), modality_type="nl")[0]
            nl_embeds = (nl_output*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]            
            nl_feat = F.normalize(nl_embeds,dim=-1) 
            nl_vecs.append(nl_feat.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_output = model.encoder(code_inputs,attention_mask=code_inputs.ne(1), modality_type="code")[0] 
            code_embeds = (code_output*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]            
            code_feat = F.normalize(code_embeds,dim=-1)
            code_vecs.append(code_feat.cpu().numpy())  
    model.train()

    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)

    scores = np.matmul(nl_vecs,code_vecs.T)
    
    # 相似度计算，并按从大到小排序，排序结果是索引
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]  

    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls,sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
                break
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = cal_r1_r5_r10(ranks) # 字典
    result["eval_mrr"] = float(np.mean(ranks))

    return result

def continue_fine_tune(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)

    # Train!
    if utils.is_main_process():
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
        logger.info("  Total train batch size  = %d", args.train_batch_size)
        logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    for epoch in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)

            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs, continue_fine_tune=True)
            nl_vec = model(nl_inputs=nl_inputs, continue_fine_tune=True)
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))

            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                if utils.is_main_process():
                    logger.info("epoch {} step {} loss {}".format(epoch,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate
        if utils.is_main_process():
            results = evaluate(args, model, tokenizer,args.eval_data_file, eval_when_training=True)
            logger.info("Eval result:")
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value,4))    
                
            #save best model
            if results['eval_mrr']>best_mrr:
                best_mrr = results['eval_mrr']
                logger.info("  "+"*"*20)  
                logger.info("  Best mrr:%s",round(best_mrr,4))
                logger.info("  "+"*"*20)                          

                checkpoint_prefix = 'checkpoint-best-mrr-continue-train'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)
                    
                results = evaluate(args, model, tokenizer,args.test_data_file, eval_when_training=True)
                logger.info("Test result:")
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value,3))

def cal_r1_r5_r10(ranks):
    r1,r5,r10= 0,0,0
    data_len= len(ranks)
    for item in ranks:
        if item >=1:
            r1 +=1
            r5 += 1 
            r10 += 1
        elif item >=0.2:
            r5+= 1
            r10+=1
        elif item >=0.1:
            r10 +=1
    result = {"R@1":round(r1/data_len,3), "R@5": round(r5/data_len,3),  "R@10": round(r10/data_len,3)}
    return result

def save_json_data(data_dir, filename, data):
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, filename)
    with open(file_name, 'w') as output:
        if type(data) == list:
            if type(data[0]) in [str, list,dict]:
                for item in data:
                    output.write(json.dumps(item))
                    output.write('\n')

            else:
                json.dump(data, output)
        elif type(data) == dict:
            json.dump(data, output)
        else:
            raise RuntimeError('Unsupported type: %s' % type(data))
    logger.info("saved result in " + file_name)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--weight_dir", default=None, type=str, required=False,
                        help="The weight directory after contrast training.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument("--temp", default=0.07, type=float,
                        help="temperature hyper-parameter")  
    # parser.add_argument("--embed_dim", default=256, type=int,
    #                     help="Dimension of projection in contrastive learning") 
    parser.add_argument("--queue_size", default=65536, type=int,
                        help="")
    parser.add_argument("--momentum", default=0.995, type=float,
                        help="")
    parser.add_argument("--alpha", default=0.4, type=float,
                        help="")
    # parser.add_argument("--k_test", default=128, type=int,
    #                     help="k candidates for CTM reranking in inference")
    parser.add_argument("--fusion_layer", default=6, type=int,
                        help="The beginning layer of the fusion layer")
    parser.add_argument("--distill", action='store_true',
                        help="Whether to distill.")
    
    # 分布式训练的设置参数
    parser.add_argument('--world_size', default=1, type=int, 
                        help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_continue_fine_tune", action='store_true',
                        help="Whether to run continue training.")
    parser.add_argument("--no_mome", action='store_true',
                        help="Whether to run Mixture-of-Modality-Experts.")
    parser.add_argument("--no_match", action='store_true',
                        help="Whether to run code-nl matching.")
    parser.add_argument("--match_queue", action='store_true',
                        help="Whether to run code-nl matching in queue.")
    parser.add_argument("--inbatch_contrast", action='store_true',
                        help="Whether to run in-batch contrast.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    # print arguments
    args = parser.parse_args()

    utils.init_distributed_mode(args) 

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if utils.is_main_process():
        logger.info("device: %s, n_gpu: %s",device, args.n_gpu)

    # set seed
    set_seed(args.seed)
    
    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = Model(args)
    if utils.is_main_process():
        logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                          find_unused_parameters=True, 
                                                          broadcast_buffers=False)

    # Training
    if args.do_train:
        train(args, model, tokenizer)

    # Continue fine tune
    if args.do_continue_fine_tune:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        weight_dir = os.path.join(args.weight_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(weight_dir))   
        continue_fine_tune(args, model, tokenizer)
      
    # Evaluation
    if utils.is_main_process():
        results = {}
        if args.do_eval:
            if args.do_zero_shot is False:
                checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                model_to_load = model.module if hasattr(model, 'module') else model  
                model_to_load.load_state_dict(torch.load(output_dir))      
            model.to(args.device)
            result = evaluate(args, model, tokenizer,args.eval_data_file)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],3)))
                
        if args.do_test:
            if args.do_zero_shot is False:
                checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                model_to_load = model.module if hasattr(model, 'module') else model  
                model_to_load.load_state_dict(torch.load(output_dir))      
            model.to(args.device)
            result = evaluate(args, model, tokenizer,args.test_data_file)
            logger.info("***** FINAL TEST RESULTS *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],3)))
            save_json_data(args.output_dir, "result.jsonl", result)

if __name__ == "__main__":
    main()
