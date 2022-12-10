
from pyossfs.oss_bucket_manager import OSSFileManager
import os
import json
import random
import copy
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

random.seed(0)
def read_txt(data_path):
    f = open(data_path, 'r')
    results = []
    for line in f:
        results.append(line.strip())
    f.close()
    return results

class Taobao_Style_Dataset(Dataset):
    def __init__(self, rev_style, data, tokenizer, config, ordered=False):
        data = data.fillna('')
        self.maidian = np.array([v for v in data['maidian']])
        for v in data['p_v']:
            if isinstance(v, str):
                continue
            else:
                print(v)
        self.ordered = ordered
        self.p_v = np.array([v for v in data['p_v']])
        self.title = np.array([v for v in data['title']])
        self.tgts = np.array([v for v in data['text']])

        self.ori_style = np.array([v for v in data['text']])
        self.rev_style = np.array([v for v in rev_style['text']])
  
        self.config = config
        self.tokenizer = tokenizer
        
        self.pos_pv = [v.split() for v in data['p_v_pos']]
        for i in range(len(self.pos_pv)):
            self.pos_pv[i] = [int(v) for v in self.pos_pv[i]]
        self.pos_maidian = [v.split() for v in data['maidian_pos']]
        for i in range(len(self.pos_maidian)):
            self.pos_maidian[i] = [int(v) for v in self.pos_maidian[i]]

        self.pos = [self.pos_maidian[i]+self.pos_pv[i] for i in range(len(self.maidian))]
    def __getitem__(self, index):
        ind = random.randint(0, len(self.ori_style)-1)
        ind1 = random.randint(0, len(self.rev_style)-1)
        ori_style_ids, ori_style_mask = self.tokenize_text(self.ori_style[ind], self.config.max_length)
        rev_style_ids, rev_style_mask = self.tokenize_text(self.rev_style[ind1], self.config.max_length)
        if self.ordered:
            data = self.get_orded_data(index)
        else:
            data, pos = self.get_data(index)

        data_ids, data_mask = self.tokenize_input(data, self.config.length_title+self.config.length_p_v+self.config.length_maidian)
        tgt, tgt_mask = self.tokenize_output(self.tgts[index], self.config.max_length)

        # pos = self.pos[index] 
        # pos = pos+[-1] * (self.config.num_nodes - len(pos))

        return ori_style_ids, ori_style_mask, data_ids, data_mask, tgt, tgt_mask, rev_style_ids, rev_style_mask, torch.tensor(pos)

    def get_data(self, index, shuffle=False):
        pos = self.pos[index] 
        pos = pos+[-1] * (self.config.num_nodes - len(pos))
        if shuffle:
            import copy
            maidian = self.maidian[index][:self.config.length_maidian]
            maidian = maidian.split()
            p_v = self.p_v[index][:self.config.length_p_v]
            p_v = p_v.split()
            kv = maidian + p_v

            kv_ = copy.deepcopy(kv)
            pos_ = copy.deepcopy(pos)
            re = []

            index_random = list(range(len(kv)))
            random.shuffle(index_random)
            
            for i in range(len(index_random)):
                pos[i] = pos_[index_random[i]]
                kv[i] = kv_[index_random[i]]

            for i in range(len(kv)):
                re += self.tokenizer.tokenize(kv[i])
                re += [self.tokenizer.sep_token]
        else:
            re = []
            if len(self.maidian[index])+1 > self.config.length_maidian or len(self.p_v[index])+1 > self.config.length_p_v:
                print('error')
                exit()

            maidian = self.maidian[index][:self.config.length_maidian]
            maidian = maidian.split()
            for i in range(len(maidian)):
                re += self.tokenizer.tokenize(maidian[i])
                re += [self.tokenizer.sep_token]
            p_v = self.p_v[index][:self.config.length_p_v]
            p_v = p_v.split()
            
            for i in range(len(p_v)):
                re += self.tokenizer.tokenize(p_v[i])
                re += [self.tokenizer.sep_token]

        title = self.title[index][:self.config.length_title]
        re += self.tokenizer.tokenize(title)
        return re, pos

    def get_orded_data(self, index):
        re = []
        
        maidian = self.maidian[index][:self.config.length_maidian]
        maidian = maidian.split()
        p_v = self.p_v[index][:self.config.length_p_v]
        p_v = p_v.split()

        pos_maidian = self.pos_maidian[index]
        pos_pv = self.pos_pv[index]

        if len(pos_pv) != len(p_v) or len(pos_maidian) != len(maidian):
            print('index',index)
            print(maidian,' ', p_v)
            print(pos_maidian, ' ', pos_pv)
            print('error')
            exit()
        pos = pos_maidian + pos_pv
        k_v = maidian + p_v
        order = copy.deepcopy(k_v)
        for i in range(len(pos)):
            order[pos[i]] = k_v[i]

        for i in range(len(order)):
            re += self.tokenizer.tokenize(order[i])
            re += [self.tokenizer.sep_token]
        title = self.title[index][:self.config.length_title]
        re += self.tokenizer.tokenize(title)
        return re
        
    def tokenize_input(self, text, max_len):
        if len(text) > max_len:
            exit()
        tokens = self.tokenizer.convert_tokens_to_ids(text[:max_len])
        mask = [1] * len(tokens) + [0] * (max_len-len(tokens))
        tokens += [self.tokenizer.pad_token_id] * (max_len - len(tokens))
        # tokens = self.tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=True, truncation=True, return_tensors='pt', add_special_tokens=False)
        return torch.tensor(tokens), torch.tensor(mask)
        
    def tokenize_text(self, text, max_len):
        tokens = self.tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=True, truncation=True, return_tensors='pt', add_special_tokens=False)
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

    def tokenize_output(self, text, max_len):
        # tokens = [104] + self.tokenizer.encode(text, add_special_tokens=False)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) >= max_len:
            tokens = tokens[:max_len-1]
        tokens = tokens + [105]
        mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
        tokens = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(tokens), torch.tensor(mask)

    def get_tokenized(self, text, max_length):
        tokens = self.tokenizer.encode_plus(text, max_length=max_length, pad_to_max_length=True, truncation=True, return_tensors='pt')
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()
    
    def __len__(self):
        return len(self.p_v)

def read_data(config, tokenizer):
    train_asr = pd.read_csv('../data/asr_train.csv', delimiter='\t')
    train_weitao = pd.read_csv('../data/weitao_train.csv', delimiter='\t')
    train_asr_dataset = Taobao_Style_Dataset(train_weitao, train_asr, tokenizer, config)
    train_weitao_dataset = Taobao_Style_Dataset(train_asr, train_weitao, tokenizer, config)

    test_asr = pd.read_csv('../data/asr_test.csv', delimiter='\t')
    test_weitao = pd.read_csv('../data/weitao_test.csv', delimiter='\t')
    test_asr_dataset = Taobao_Style_Dataset(test_weitao, test_asr, tokenizer, config)
    test_weitao_dataset = Taobao_Style_Dataset(test_asr, test_weitao, tokenizer, config)
    config.num_workers = 1
    train_asr_dataloader = DataLoader(train_asr_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True)
    train_weitao_dataloader = DataLoader(train_weitao_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True)

    test_asr_dataloader = DataLoader(test_asr_dataset, batch_size=config.batch_size_test, num_workers=config.num_workers)
    test_weitao_dataloader = DataLoader(test_weitao_dataset, batch_size=config.batch_size_test, num_workers=config.num_workers)
    len_data = 0
    len_tgt = 0

    print('train_asr {} train_weitao {} test_asr {} train_weitao {}'.format(len(train_asr_dataset), len(train_weitao_dataset), len(test_asr_dataset), len(test_weitao_dataset)))
    return train_asr_dataloader, train_weitao_dataloader, test_asr_dataloader, test_weitao_dataloader


