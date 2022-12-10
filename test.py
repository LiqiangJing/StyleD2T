
from data import read_data
from util import batch_preprocess
import torch
from args import get_args
from transformers import BertTokenizer
config = get_args()
tokenizer = BertTokenizer('../model/bart-baike-daren/vocab-chinese.txt')

train_asr_dataloader, train_weitao_dataloader, test_asr_dataloader, test_weitao_dataloader = read_data(config, tokenizer)

dataloader = zip(test_asr_dataloader, test_weitao_dataloader)

correct = 0
total = 0

for (asr, weitao) in dataloader:
    style_ids, style_mask, data_ids, data_mask, tgt_ids, tgt_mask, rev_style_ids, rev_mask, pos_label, style_labels = batch_preprocess(asr, weitao, config)
    num = int(len(pos_label)/2)
    pos_pre = torch.zeros_like(pos_label)
    for i in range(len(pos_pre)):
        for j in range(len(pos_pre[0])):
            pos_pre[i,j] = j
    
    correct += torch.sum((pos_pre == pos_label) & (pos_label != -1)).item()
    total += torch.sum( pos_label != -1).item()

print((1.0 * correct) / total)

### all 13.30 asr 18.25 10.39