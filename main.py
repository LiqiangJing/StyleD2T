from transformers import BertTokenizer, AutoConfig
from modeling_bart import BartForConditionalGeneration
import torch
import os
from tqdm import tqdm
from args import get_args
import time
from data import read_data
from metric import Evaluator
import numpy as np
from models import MyModel
from util import batch_preprocess, save_model, inference, shift_tokens_right
import jieba
import math
torch.set_printoptions(threshold=np.inf)        

def acc_planning(pre, label):
    '''pre: batch, 30,   label: batch, 30'''
    acc = torch.sum((pre == label) & (label != -1)) * 1.0/ torch.sum(label != -1)
    return acc.item()   

def N_Coverage(refs, hyps):
    coverage = []
    for ref, hyp in zip(refs, hyps):
        cov = Coverage(ref, hyp)
        if np.isnan(cov):
            cov = 0
        coverage.append(cov)
    return np.mean(coverage)
    
def Coverage(ref, hyp, sep='[SEP]'):
    coverage = []
    # hyp = ''.join(hyp.split(' '))
    vals = ref.split(sep)[:-1]
    vals = set([v.partition(':')[2] for v in vals])
    for val in vals:
        if val in hyp:
            coverage.append(1)
        else:
            coverage.append(0)
    return np.mean(coverage) 

def tokenize_output(tokenizer, text, max_len):
    # tokens = [104] + self.tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) >= max_len:
        tokens = tokens[:max_len-1]
    tokens = tokens + [105]
    mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
    tokens = tokens + [0] * (max_len - len(tokens))
    return torch.tensor(tokens), torch.tensor(mask)

def data_augmentation(data_ids, data_mask, rev_style_ids, rev_mask, model, evaluator, tokenizer, pos_label, t1=0.99, t2=0.99, t3=50):

    with torch.no_grad():
        model.eval()
        rev, _ = model.generate_text(
                input_ids=data_ids, attention_mask=data_mask, input_style_ids=rev_style_ids, style_attention_mask=rev_mask, pos_true=pos_label
            )
        rev = [rev[i].replace('<T>', '').replace('<S>', '') for i in range(len(rev))]
    asr_style = 1
    weitao_style= 0
    re = [False] * len(data_ids)
    for i in range(int(len(rev)/2)):
        if len(rev[i]) < 63 or len(rev[i]) > 162:
            continue
        if evaluator.ppl([rev[i]]) > t3:
            continue
        data = tokenizer.decode(data_ids[i]).replace(' ','')
        cover_ = Coverage(data, rev[i])
        cover =  cover_ > t1
        if not cover:
            continue
        # print(str(cover_)+data+'\n'+rev[i])
        text_transfered = ' '.join(jieba.cut(rev[i].strip()))
        label = evaluator.classifier.predict([text_transfered])
        if label[0][0][0] != '__label__weitao' or label[1][0][0] < t2:
            continue
        
        re[i] = True

    for i in range(int(len(rev)/2), len(rev), 1):
        if len(rev[i]) < 48 or len(rev[i]) > 274:
            continue
        if evaluator.ppl([rev[i]]) > t3:
            continue
        data = tokenizer.decode(data_ids[i]).replace(' ','')
        cover = Coverage(data, rev[i]) > t1
        if not cover:
            continue
        text_transfered = ' '.join(jieba.cut(rev[i].strip()))
        label = evaluator.classifier.predict([text_transfered])

        if label[0][0][0] != '__label__asr' or label[1][0][0] < t2:
            continue
        re[i] = True
    
    aug_data_ids = []
    aug_data_mask = []
    aug_rev_ids = []
    aug_rev_mask = []
    aug_tgt_ids = []
    aug_tgt_mask = []
    aug_pos_label = []
    for i in range(len(re)):
        if re[i]:
            aug_data_ids.append(data_ids[i])
            aug_data_mask.append(data_mask[i])
            aug_rev_ids.append(rev_style_ids[i])
            aug_rev_mask.append(rev_mask[i])
            a, b = tokenize_output(tokenizer,rev[i], config.max_length)
            a = a.cuda()
            b = b.cuda()
            aug_tgt_ids.append(a)
            aug_tgt_mask.append(b)
            aug_pos_label.append(pos_label[i])
            # if i < len(re) /2:
            #     print('weitao')
            # else:
            #     print('asr')
            # print(rev[i])

    if len(aug_data_ids) > 1:
        aug_data_ids = torch.stack(aug_data_ids, 0)
        aug_data_mask = torch.stack(aug_data_mask, 0)
        aug_rev_ids = torch.stack(aug_rev_ids, 0)
        aug_rev_mask= torch.stack(aug_rev_mask, 0)
        aug_tgt_ids= torch.stack(aug_tgt_ids, 0)
        aug_tgt_mask= torch.stack(aug_tgt_mask, 0)
        aug_pos_label = torch.stack(aug_pos_label, 0)
    elif len(aug_data_ids) == 1:
        aug_data_ids = aug_data_ids[0].unsqueeze(0)
        aug_data_mask = aug_data_mask[0].unsqueeze(0)
        aug_rev_ids = aug_rev_ids[0].unsqueeze(0)
        aug_rev_mask = aug_rev_mask[0].unsqueeze(0)
        aug_tgt_ids = aug_tgt_ids[0].unsqueeze(0)
        aug_tgt_mask = aug_tgt_mask[0].unsqueeze(0)
        aug_pos_label = aug_pos_label[0].unsqueeze(0)

    model.train()
    
    return aug_data_ids, aug_data_mask, aug_rev_ids, aug_rev_mask, aug_tgt_ids, aug_tgt_mask, aug_pos_label

    
def train_one_epoch(config, asr_dataloader, weitao_dataloader, model, optim, criterion, epoch):
    model.train()
    dataloader = zip(asr_dataloader, weitao_dataloader)
    # pbar = tqdm(total=len(list(dataloader)))
    pbar = tqdm(total=len(asr_dataloader), ncols=150)
    if config.planning:
        plan_fun = torch.nn.CrossEntropyLoss(ignore_index=-1)

    if config.style_disentange:
        style_fun = torch.nn.CrossEntropyLoss()
        mse_fun = torch.nn.MSELoss(reduction='none')

    batch_idx = 0
    
    for (asr, weitao) in dataloader:
        pbar.update(1)
        style_ids, style_mask, data_ids, data_mask, tgt_ids, tgt_mask, rev_style_ids, rev_mask, pos_label, style_labels = batch_preprocess(asr, weitao, config)

        aug_num = 0
        if config.pseudo_sample and epoch >= config.pretrain_step:
            aug_data_ids, aug_data_mask, aug_rev_ids, aug_rev_mask, aug_tgt_ids, aug_tgt_mask, aug_pos_label = data_augmentation(
                data_ids=data_ids, data_mask=data_mask, rev_style_ids=rev_style_ids, rev_mask=rev_mask, model=model, evaluator=evaluator, tokenizer=tokenizer, pos_label=pos_label)
            aug_num = len(aug_data_ids)

        optim.zero_grad()
        decoder_input_ids = shift_tokens_right(tgt_ids, model.config.bos_token_id)
        
        outputs = model(input_ids=data_ids, input_style_ids=style_ids, style_attention_mask=style_mask, attention_mask=data_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=tgt_mask, pos_label=pos_label)
        # outputs = model(input_ids=data_ids, attention_mask=data_mask, decoder_input_ids=decoder_input_ids, input))

        logits = outputs[0]
        labels = tgt_ids
        # tgts = tgt_ids
        rec_loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        total_loss = rec_loss
  
        if config.pseudo_sample and aug_num != 0:
            aug_decoder_input_ids = shift_tokens_right(aug_tgt_ids, model.config.bos_token_id)
            aug_outputs = model(input_ids=aug_data_ids, input_style_ids=aug_rev_ids, style_attention_mask=aug_rev_mask, attention_mask=aug_data_mask, decoder_input_ids=aug_decoder_input_ids, decoder_attention_mask=aug_tgt_mask, pos_label=aug_pos_label)
            # outputs = model(input_ids=data_ids, attention_mask=data_mask, decoder_input_ids=decoder_input_ids, input))

            aug_logits = aug_outputs[0]
            aug_labels = aug_tgt_ids
            aug_loss = config.factor_aug * criterion(aug_logits.view(-1, aug_logits.shape[-1]), aug_labels.view(-1))
            total_loss = total_loss + aug_loss
        else:
            aug_loss = torch.tensor(0)

        if config.planning and outputs[-4] is not None:
            # plan_loss = config.factor_plan * plan_fun(outputs[-3].view(-1, pos_label.shape[-1]), pos_label.view(-1))
            plan_loss = config.factor_plan * plan_fun(outputs[-4].view(-1, outputs[-4].shape[-1]), pos_label.view(-1))
            acc_plan = acc_planning(outputs[-4].argmax(-1), pos_label)
            total_loss = total_loss + plan_loss
        else:
            plan_loss = torch.tensor(0)
            acc_plan = 0
        
        if config.style_disentange:
            batch_size = int(len(style_ids)/2)
            target1 = torch.mean(outputs[-1][: batch_size], 0).clone().detach().unsqueeze(0)
            target1 = target1.repeat(batch_size, 1)
            target2 = torch.mean(outputs[-1][batch_size :], 0).clone().detach().unsqueeze(0)
            target2 = target2.repeat(batch_size, 1)
            target = torch.cat((target1, target2),0)
            target_rev = torch.cat((target2, target1), 0)
            style_loss = config.factor_style * (style_fun(outputs[-2], style_labels.long()) + torch.mean(mse_fun(outputs[-1], target)) + math.exp(-1) * torch.mean(torch.exp(-mse_fun(outputs[-1], target_rev))))
            total_loss = total_loss + style_loss
        else:
            style_loss = torch.tensor(0)
        pbar.set_description(
            "epoch {}, step {}, rec_loss {:.4f}, aug_loss {:.4f} plan_loss {:.4f} style_loss {:.4f} aug_num {} acc_plan {:.4f}".format(
                epoch, batch_idx, rec_loss.item(), aug_loss.item(), plan_loss.item() , style_loss.item(), aug_num, acc_plan)
        )
        batch_idx += 1

        # if batch_idx > 10 :
        #     exit()
        #     break
        total_loss.backward()
        optim.step()
        
    pbar.close()

def evaluation(config, model, asr_dataloader, weitao_dataloader, evaluator, epoch, tokenizer):
    model.eval()

    # correct = 0
    # total = 0
    # for style_ids, style_mask, data_ids, data_mask, tgt, tgt_mask, rev_style_ids, rev_style_mask, pos in weitao_dataloader:
    #     pos_pre = torch.arange(len(pos[0])).repeat(len(pos), 1)
    #     correct += torch.sum((pos_pre == pos) & (pos != -1)).item()
    #     total += torch.sum( pos != -1).item()
    # print(correct/total)
    # correct = 0
    # total = 0
    # for style_ids, style_mask, data_ids, data_mask, tgt, tgt_mask, rev_style_ids, rev_style_mask, pos in asr_dataloader:
    #     pos_pre = torch.arange(len(pos[0])).repeat(len(pos), 1)
    #     correct += torch.sum((pos_pre == pos) & (pos != -1)).item()
    #     total += torch.sum( pos != -1).item()
    # print(correct/total)
    # exit()
    print('generation asr!')
    asr_data, asr_sty, asr_gen, asr_ref, asr_rev_sty, asr_rev_gen, asr_acc_planning, asr_ken = inference(asr_dataloader, model, config, tokenizer)
    print('generation weitao!')
    weitao_data, weitao_sty, weitao_gen, weitao_ref, weitao_rev_sty, weitao_rev_gen, weitao_acc_planning, weitao_ken = inference(weitao_dataloader, model, config, tokenizer)
    # print('convert asr to weitao!')

    # a, b, c = inference(test_asr2weitao_dataset, model, config, tokenizer)
    # print('convert weitao to asr!')

    # a1, b1, c1 = inference(test_weitao2asr_dataloader, model, config, tokenizer)
    bleu_asr = evaluator.ref_bleu_asr(asr_gen)
    bleu_weitao = evaluator.ref_bleu_weitao(weitao_gen)

    rouge_l_asr = evaluator.rouge_l_asr(asr_gen)
    rouge_l_weitao = evaluator.rouge_l_weitao(weitao_gen)

    acc_asr = evaluator.acc_asr(asr_gen)
    acc_weitao = evaluator.acc_weitao(weitao_gen)
    acc_asr2weitao = evaluator.acc_weitao(asr_rev_gen)
    acc_weitao2asr = evaluator.acc_asr(weitao_rev_gen)

    cover_asr = N_Coverage(asr_data, asr_gen)
    cover_weitao = N_Coverage(weitao_data, weitao_gen)
    cover_asr2weitao = N_Coverage(asr_data, asr_rev_gen)
    cover_weitao2asr = N_Coverage(weitao_data, weitao_rev_gen)

    ppl_asr = evaluator.ppl(asr_gen)
    ppl_weitao = evaluator.ppl(weitao_gen)
    ppl_asr2weitao = evaluator.ppl(asr_rev_gen)
    ppl_weitao2asr = evaluator.ppl(weitao_rev_gen)

    for k in range(5):
        idx = np.random.randint(len(asr_gen))
        print('*' * 20, 'sample_asr', '*' * 20)
        print('[data]', asr_data[idx])
        print('[ori_sty]', asr_sty[idx])
        print('[ori_gen]', asr_gen[idx])
        print('[ori_ref]', asr_ref[idx])
        print('[rev_sty]', asr_rev_sty[idx])
        print('[rev_gen]', asr_rev_gen[idx])
    print('*' * 20, '********', '*' * 20)

    for k in range(5):
        idx = np.random.randint(len(weitao_gen))
        print('*' * 20, 'sample_weitao', '*' * 20)
        print('[data]', weitao_data[idx])
        print('[ori_sty]', weitao_sty[idx])
        print('[ori_gen]', weitao_gen[idx])
        print('[ori_ref]', weitao_ref[idx])
        print('[rev_sty]', weitao_rev_sty[idx])
        print('[rev_gen]', weitao_rev_gen[idx])

    print('*' * 20, '********', '*' * 20)

    print(('[auto_eval] bleu_asr: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} acc_asr {:.4f} \n').format(bleu_asr[0], bleu_asr[1], bleu_asr[2], bleu_asr[3], bleu_asr[4], acc_asr))
    print(('[auto_eval] bleu_weitao: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} acc_weitao {:.4f} \n').format(bleu_weitao[0], bleu_weitao[1], bleu_weitao[2], bleu_weitao[3], bleu_weitao[4], acc_weitao))
    print(('acc_asr2weitao: {:.4f} acc_weitao2asr {:.4f} acc {:.4f}\n').format(acc_asr2weitao, acc_weitao2asr, (acc_asr2weitao * len(asr_rev_gen)+ acc_weitao2asr * len(weitao_rev_gen))/(len(asr_rev_gen)+len(weitao_rev_gen))))
    print(('asr_planning: {:.4f} weitao_planning {:.4f} \n').format(asr_acc_planning, weitao_acc_planning))
    print(('rouge_l_asr {:.4f} rouge_l_weitao {:.4f}').format(rouge_l_asr, rouge_l_weitao))
    print(('cover_asr {:.4f} cover_weitao {:.4f} cover_asr2weitao: {:.4f} cover_weitao2asr {:.4f} \n').format(cover_asr, cover_weitao, cover_asr2weitao, cover_weitao2asr))
    print(('ppl_asr {:.4f} ppl_weitao {:.4f} ppl_asr2weitao: {:.4f} ppl_weitao2asr {:.4f} \n').format(ppl_asr, ppl_weitao, ppl_asr2weitao, ppl_weitao2asr))
    print(('asr_ken: {:.4f} weitao_ken {:.4f} \n').format(asr_ken, weitao_ken))

    # save output
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)

    with open(os.path.join(config.save_folder, 'eval_log.txt'), 'a') as fl:
        print(('iter{:5d}: bleu_asr: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} acc_asr {:.4f}' ).format(epoch, bleu_asr[0], bleu_asr[1], bleu_asr[2], bleu_asr[3], bleu_asr[4], acc_asr), file=fl)
        print(('iter{:5d}: bleu_weitao: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} acc_weitao {:.4f}' ).format(epoch, bleu_weitao[0], bleu_weitao[1], bleu_weitao[2], bleu_weitao[3], bleu_weitao[4], acc_weitao), file=fl)
        print(('acc_asr2weitao: {:.4f} acc_weitao2asr {:.4f} acc {:.4f}\n').format(acc_asr2weitao, acc_weitao2asr, (acc_asr2weitao * len(asr_rev_gen)+ acc_weitao2asr * len(weitao_rev_gen))/(len(asr_rev_gen)+len(weitao_rev_gen))), file=fl)
        print(('asr_planning: {:.4f} weitao_planning {:.4f} \n').format(asr_acc_planning, weitao_acc_planning), file=fl)
        print(('rouge_l_asr {:.4f} rouge_l_weitao {:.4f}').format(rouge_l_asr, rouge_l_weitao), file=fl)
        print(('cover_asr {:.4f} cover_weitao {:.4f}  cover_asr2weitao: {:.4f} cover_weitao2asr {:.4f} \n').format(cover_asr, cover_weitao, cover_asr2weitao, cover_weitao2asr), file=fl)
        print(('ppl_asr {:.4f} ppl_weitao {:.4f} ppl_asr2weitao: {:.4f} ppl_weitao2asr {:.4f} \n').format(ppl_asr, ppl_weitao, ppl_asr2weitao, ppl_weitao2asr), file=fl)

        acc_aa = (acc_asr * len(asr_rev_gen)+ acc_weitao * len(weitao_rev_gen)) / (len(asr_gen) + len(weitao_gen))
        acc = (acc_asr2weitao * len(asr_rev_gen)+ acc_weitao2asr * len(weitao_rev_gen)) / (len(asr_gen) + len(weitao_gen))
        acc = (acc_aa + acc) /2
        bleu = (bleu_asr[0] * len(asr_gen) + bleu_weitao[0] * len(weitao_gen)) / (len(asr_gen) + len(weitao_gen))
        rouge_l = (rouge_l_asr * len(asr_gen) + rouge_l_weitao * len(weitao_gen)) / (len(asr_gen) + len(weitao_gen))

        ppl = (ppl_asr + ppl_asr2weitao) * len(asr_gen) / 2 + (ppl_weitao + ppl_weitao2asr) * len(weitao_gen) / 2
        ppl = ppl / (len(asr_gen) + len(weitao_gen))

        cover = (cover_asr + cover_asr2weitao) * len(asr_gen) / 2 + (cover_weitao + cover_weitao2asr) * len(weitao_gen) / 2
        cover = cover / (len(asr_gen) + len(weitao_gen))

        print(('ppl:{:.4f} acc:{:.4f} rouge-l:{:.4f} bleu:{:.4f} cover:{:.4f}').format(ppl, acc, rouge_l, bleu, cover), file=fl)

    with open(os.path.join(config.save_folder,'test'+ str(epoch)+'.txt'), 'w') as fw:
        print(('[auto_eval] bleu_asr: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} acc_asr {:.4f}\n').format(bleu_asr[0], bleu_asr[1], bleu_asr[2], bleu_asr[3], bleu_asr[4], acc_asr), file=fw)
        print(('[auto_eval] bleu_weitao: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} acc_weitao {:.4f}\n').format(bleu_weitao[0], bleu_weitao[1], bleu_weitao[2], bleu_weitao[3], bleu_weitao[4], acc_weitao), file=fw)
        print(('acc_asr2weitao: {:.4f} acc_weitao2asr {:.4f} acc {:.4f}\n').format(acc_asr2weitao, acc_weitao2asr, (acc_asr2weitao * len(asr_rev_gen)+ acc_weitao2asr * len(weitao_rev_gen))/(len(asr_rev_gen)+len(weitao_rev_gen))), file=fw)
        print(('asr_planning: {:.4f} weitao_planning {:.4f} \n').format(asr_acc_planning, weitao_acc_planning), file=fw)
        print(('rouge_l_asr {:.4f} rouge_l_weitao {:.4f}').format(rouge_l_asr, rouge_l_weitao), file=fw)
        print(('cover_asr2weitao: {:.4f} cover_weitao2asr {:.4f} \n').format(cover_asr2weitao, cover_weitao2asr), file=fw)
        print(('ppl_asr {:.4f} ppl_weitao {:.4f} ppl_asr2weitao: {:.4f} ppl_weitao2asr {:.4f} \n').format(ppl_asr, ppl_weitao, ppl_asr2weitao, ppl_weitao2asr), file=fw)

        print('*' * 20, 'sample_asr', '*' * 20, file=fw)

        for idx in range(len(asr_gen)):
            # print('*' * 20, file=fw)
            print('[data]', asr_data[idx], file=fw)
            print('[ori_sty]', asr_sty[idx], file=fw)
            print('[ori_gen]', asr_gen[idx], file=fw)
            print('[ori_ref]', asr_ref[idx], file=fw)
            print('[rev_sty]', asr_rev_sty[idx], file=fw)
            print('[rev_gen]', asr_rev_gen[idx], file=fw)

            print('*' * 20, '********', '*' * 20, file=fw)

        print('*' * 20, 'sample_weitao', '*' * 20, file=fw)

        for idx in range(len(weitao_gen)):
                # print('*' * 20, file=fw)
            print('[data]', weitao_data[idx], file=fw)
            print('[ori_sty]', weitao_sty[idx], file=fw)
            print('[ori_gen]', weitao_gen[idx], file=fw)
            print('[ori_ref]', weitao_ref[idx], file=fw)
            print('[rev_sty]', weitao_rev_sty[idx], file=fw)
            print('[rev_gen]', weitao_rev_gen[idx], file=fw)
            
            print('*' * 20, '********', '*' * 20, file=fw)

    model.train()
    return bleu_asr[0], bleu_asr[1], bleu_asr[2], bleu_asr[3], bleu_asr[4], bleu_weitao[0], bleu_weitao[1], bleu_weitao[2], bleu_weitao[3], bleu_weitao[4]



if __name__ == '__main__':
    results = {'bleu1':[], 
                'bleu2': [],
                'bleu3': [],
                'bleu4': [],
                'bleu_avg': [],
                'epoch': []
    }


    config = get_args()
    print(config)
    ### Load Weitao Dataset.
    tokenizer = BertTokenizer('../model/bart-baike-daren/vocab-chinese.txt')
    bart = BartForConditionalGeneration(AutoConfig.from_pretrained("../model/bart-baike-daren/config-chinese-small.json"))
    model_dict = torch.load(open(os.path.join("../model/bart-baike-daren/model-120000.pt"), 'rb'))
    # model_dict = torch.load(open(os.path.join("../model/bart-baike/model-500000.pt"), 'rb'))

    model_dict['final_logits_bias'] = torch.zeros((1, 21128))
    bart.load_state_dict(model_dict)
    
    train_asr_dataloader, train_weitao_dataloader, test_asr_dataloader, test_weitao_dataloader = read_data(config, tokenizer)
    # print(len(train_asr_dataloader))
    # print(len(train_weitao_dataloader))
    # print(len(test_asr_dataloader))
    # print(len(test_weitao_dataloader))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    config.save_folder = os.path.join(
        os.path.join(config.save_path, config.model_name),
        str(time.strftime('%b%d%H%M%S', time.localtime()))
        )

    # model = MyModel(config, tokenizer, bart)
    # model = bart
    model = MyModel(config, tokenizer, bart)
    # model.load_state_dict(torch.load('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/New/ours/save/test/Mar10155441/0_2.6966010614420135.pkl'))
    # model.load_state_dict(torch.load('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/planning/save/planning_embedding/Nov17201542/15_3.127172874205838.pkl'))
    if config.parallel:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.use_gpu:
        model = model.cuda()
    evaluator = Evaluator()

    for epoch in range(config.Epoch):
        train_one_epoch(config, train_asr_dataloader,train_weitao_dataloader, model, optimizer, criterion, epoch)
        score = evaluation(config, model, test_asr_dataloader, test_weitao_dataloader, evaluator,epoch, tokenizer)
        # score = evaluation(config, model, train_asr_dataloader, train_weitao_dataloader, evaluator,epoch, tokenizer)

        results['bleu_avg'].append(score[0])
        results['bleu1'].append(score[1])
        results['bleu2'].append(score[2])
        results['bleu3'].append(score[3])
        results['bleu4'].append(score[4])
        save_model(epoch, model, max(results['bleu_avg']), config)
        print('best epoch {:4d} avg & 1-4 {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(results['bleu_avg'].index(max(results['bleu_avg'])),max(results['bleu_avg']),max(results['bleu1']), max(results['bleu2']), max(results['bleu3']), max(results['bleu4'])))
        with open(os.path.join(config.save_folder, 'eval_log.txt'), 'a') as fl:
            print(('best epoch {:4d} bleu: {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} \n').format(results['bleu_avg'].index(max(results['bleu_avg'])), max(results['bleu_avg']),max(results['bleu1']), max(results['bleu2']), max(results['bleu3']), max(results['bleu4'])), file=fl)
