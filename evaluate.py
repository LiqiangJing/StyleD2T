from metric import Evaluator
import pandas as pd
import numpy as np
import json
import jieba
from pyossfs.oss_bucket_manager import OSSFileManager

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
    # vals = ref.split(sep)[:-1] #[kv, kv]
    vals = set([v.partition(':')[2] for v in ref])

    for val in vals:
        if val in hyp:
            coverage.append(1)
        else:
            coverage.append(0)
            # print(vals)
            # print(hyp)
            # exit()
    return np.mean(coverage) 
evaluator = Evaluator()

def evaluation( asr_gen, asr_rev_gen, weitao_gen, weitao_rev_gen, asr_data, weitao_data):
    bleu = 0
    rouge_l = 0
    cover = 0
    ppl = 0
    bleu_asr = evaluator.ref_bleu_asr(asr_gen)
    bleu_weitao = evaluator.ref_bleu_weitao(weitao_gen)
 
    rouge_l_asr = evaluator.rouge_l_asr(asr_gen)
    rouge_l_weitao = evaluator.rouge_l_weitao(weitao_gen)

    acc_asr = evaluator.acc_asr(asr_gen)
    acc_weitao = evaluator.acc_weitao(weitao_gen)
    acc_asr2weitao = evaluator.acc_weitao(asr_rev_gen)
    acc_weitao2asr = evaluator.acc_asr(weitao_rev_gen)
    # print(acc_weitao2asr)
    # print(acc_asr2weitao)

    cover_asr = N_Coverage(asr_data, asr_gen)
    cover_weitao = N_Coverage(weitao_data, weitao_gen)
    cover_asr2weitao = N_Coverage(asr_data, asr_rev_gen)
    cover_weitao2asr = N_Coverage(weitao_data, weitao_rev_gen)

    ppl_asr = evaluator.ppl(asr_gen)
    ppl_weitao = evaluator.ppl(weitao_gen)
    ppl_asr2weitao = evaluator.ppl(asr_rev_gen)
    ppl_weitao2asr = evaluator.ppl(weitao_rev_gen)

    acc_aa = (acc_asr * len(asr_rev_gen)+ acc_weitao * len(weitao_rev_gen)) / (len(asr_gen) + len(weitao_gen))
    print('acc_origin',acc_aa) ## 
    acc = (acc_asr2weitao * len(asr_rev_gen)+ acc_weitao2asr * len(weitao_rev_gen)) / (len(asr_gen) + len(weitao_gen))
    print('acc_rev', acc)
    print((acc + acc_aa) /2)
    bleu = (bleu_asr[0] * len(asr_gen) + bleu_weitao[0] * len(weitao_gen)) / (len(asr_gen) + len(weitao_gen))
    print('bleu-4', bleu)
    rouge_l = (rouge_l_asr * len(asr_gen) + rouge_l_weitao * len(weitao_gen)) / (len(asr_gen) + len(weitao_gen))
    print('rouge-l', rouge_l)
    ppl = (ppl_asr + ppl_asr2weitao) * len(asr_gen) / 2 + (ppl_weitao + ppl_weitao2asr) * len(weitao_gen) / 2
    ppl = ppl / (len(asr_gen) + len(weitao_gen))
    print('ppl_origin', (ppl_asr * len(asr_gen) + ppl_weitao * len(weitao_gen))/ (len(asr_gen) + len(weitao_gen)))
    print('ppl_rev', (ppl_asr2weitao * len(asr_gen) + ppl_weitao2asr * len(weitao_gen))/ (len(asr_gen) + len(weitao_gen)))
    cover = (cover_asr + cover_asr2weitao) * len(asr_gen) / 2 + (cover_weitao + cover_weitao2asr) * len(weitao_gen) / 2
    print('cover_origin', (cover_asr * len(asr_gen) + cover_weitao*len(weitao_gen))/ (len(asr_gen)+ len(weitao_gen)))
    print('cover_rev', (cover_asr2weitao * len(asr_gen) + cover_weitao2asr*len(weitao_gen))/ (len(asr_gen)+ len(weitao_gen)))
    cover = cover / (len(asr_gen) + len(weitao_gen))
    return bleu, rouge_l, cover, acc, ppl

def read_data():
    asr_test = pd.read_csv('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/New/data/asr_test.csv', sep='\t').fillna('')

    asr_ = [v for v in asr_test['text']]
    maidian = [v.split() for v in asr_test['maidian']]
    p_v = [v.split() for v in asr_test['p_v']]
    asr_data = [maidian[i]+p_v[i] for i in range(len(maidian))]

    weitao_test = pd.read_csv('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/New/data/weitao_test.csv', sep='\t').fillna('')
    weitao_ = [v for v in weitao_test['text']]

    maidian = [v.split() for v in weitao_test['maidian']]
    p_v = [v.split() for v in weitao_test['p_v']]
    weitao_data = [maidian[i]+p_v[i] for i in range(len(maidian))]
    asr_gen = asr_rev = asr_
    weitao_gen = weitao_rev = weitao_
    return asr_data, weitao_data
    # print(evaluation(asr_, asr_rev, weitao_, weitao_rev, asr_data, weitao_data))

def evaluate_lstm():
    asr_data, weitao_data = read_data()
    texts = []
    # f = open('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/LSTM/LSTM_result.txt', 'r')
    # for line in f:
    #     texts.append(line.strip())
    texts = read_txt_oss('oss://xdp-expriment/liqiang/TaoStyle/lstm/LSTM_result.txt')
    asr_gen = texts[:1000]
    weitao_gen = texts[1000:2000]
    print(len(texts))
    print(evaluation(asr_gen, weitao_gen, weitao_gen, asr_gen, asr_gen, weitao_data))

def evaluate_phvm():
    asr_data, weitao_data = read_data()
    texts = []
    f = open('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/New/ours/results/PHVM.json', 'r')
    for line in f:
        json_data = json.loads(line)[0]['desc'].replace(' ','')
        texts.append(json_data)
    f.close()

    asr_gen = texts[:1000]
    weitao_gen = texts[1000:]

    asr_rev = texts[:1000]
    weitao_rev = texts[1000:]
    print(evaluation(asr_gen, asr_rev, weitao_gen, weitao_rev, asr_data, weitao_data))

def evaluate_plangen():
    texts = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/New/plangen/generator/test_out_3200.txt')
    # texts = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/PlanGen/generator/test_out87.8.txt')
    asr_data, weitao_data = read_data()

    asr_gen = texts[:1000]
    weitao_gen = texts[1000:]
    # f = open('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/nast/data/plangen/test_0.txt', 'w')
    # for t in weitao_gen:
    #     f.write(' '.join(jieba.cut(t))+'\n')
    # f.close()
    # f = open('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/nast/data/plangen/test_1.txt', 'w')
    # for t in asr_gen:
    #     f.write(' '.join(jieba.cut(t))+'\n')
    # f.close()

    print(evaluation(asr_gen, asr_gen, weitao_gen, weitao_gen, asr_data, weitao_data))

def read_txt(path):
    f = open(path, 'r')
    texts = []
    for line in f:
        texts.append(line.strip())
    f.close()
    return texts

def read_txt_oss(path):
    print('Reading.....')
    f = OSSFileManager.open(path, 'r')
    texts = []
    for line in f:
        texts.append(line.strip())
    f.close()
    print('finish')
    return texts

def evaluate_nast():
    num = 59500
    asr_data, weitao_data = read_data()
    # f = open('result.txt', 'w')
    # while True:
    #     if num % 3000 == 0:
    #         print(num)
    #     asr2weitao = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/nast/styletransformer/output/MODELNAME/'+str(num)+'.pos2neg.txt')
    #     weitao2asr = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/nast/styletransformer/output/MODELNAME/'+str(num)+'.neg2pos.txt')

    #     a,b,c,d,e=evaluation(asr2weitao, asr2weitao, weitao2asr, weitao2asr, asr_data, weitao_data)
    #     print(d)
    #     f.write(str(num)+': '+str(d)+'\n')
    #     f.flush()
    #     if d > 0.74:
    #         print(num)
    #         print(d)
    #         exit()
    #     num += 50
    asr2weitao = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/New/nast/styletransformer/nast/run20220308_093743/output/nast/best.pos2neg.txt')
    weitao2asr = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/nast/styletransformer/output/MODELNAME/'+str(num)+'.neg2pos.txt')
    print(evaluation(asr2weitao, asr2weitao, weitao2asr, weitao2asr, asr_data, weitao_data))

def evaluate_styletran():
    texts = read_txt_oss('oss://xdp-expriment/liqiang/TaoStyle/styletransformer/runs//Mar04142351/800.txt')
    asr_data, weitao_data = read_data()
    rev = []
    for text in texts:
        if text[:7] == '[rev ] ':
            rev.append(text[7:])
    asr2weitao = rev[:1000]
    weitao2asr = rev[1000:]
    print(evaluation(asr2weitao, asr2weitao, weitao2asr, weitao2asr, asr_data, weitao_data))

def evaluate_phvm_styletrans():
    asr_data, weitao_data = read_data()
    texts = []
    f = open('../LSTM/PHVM.json', 'r')
    for line in f:
        json_data = json.loads(line)[0]['desc'].replace(' ','')
        # print(json_data)
        texts.append(json_data)
    f.close()
    asr_gen = texts[:253]
    weitao_gen = texts[253:253+322]

    asr_rev = texts[253+322: 253+322+253]
    weitao_rev = texts[253+322+253:]

def evaluate_phvm_nast():
    asr_data, weitao_data = read_data()
    texts = []
    f = open('../LSTM/PHVM.json', 'r')
    for line in f:
        json_data = json.loads(line)[0]['desc'].replace(' ','')
        # print(json_data)
        texts.append(json_data)
    f.close()
    asr_gen = texts[:253]
    weitao_gen = texts[253:253+322]
    asr_rev = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/nast/styletransformer/output/phvm/asr2weitao.txt')
    weitao_rev = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/nast/styletransformer/output/phvm/weitao2asr.txt')
    print(evaluation(asr_gen, asr_rev, weitao_gen, weitao_rev, asr_data, weitao_data))


def evaluate_plangen_styletrans():
    texts = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/PlanGen/generator/test_out.txt')
    texts = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/PlanGen/generator/test_out87.8.txt')
    asr_data, weitao_data = read_data()

    asr_gen = texts[:253]
    weitao_gen = texts[253:]

    print(evaluation(asr_gen, asr_gen, weitao_gen, weitao_gen, asr_data, weitao_data))

def evaluate_plangen_nast():
    texts = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/PlanGen/generator/test_out.txt')
    texts = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/PlanGen/generator/test_out87.8.txt')
    asr_data, weitao_data = read_data()

    asr_gen = texts[:253]
    weitao_gen = texts[253:]

    asr_rev = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/nast/styletransformer/output/plangen/asr2weitao.txt')
    weitao_rev = read_txt('/home/admin/workspace/project/DocumentStyleTransfer/supervised/stylegeneration/TaoBao/nast/styletransformer/output/plangen/weitao2asr.txt')
    print(evaluation(asr_gen, asr_rev, weitao_gen, weitao_rev, asr_data, weitao_data))
if __name__ == "__main__":
    evaluate_phvm()
    # evaluate_nast()
    # evaluate_plangen()
    # evaluate_lstm()
    # read_data()
    # evaluate_phvm_nast()
    # evaluate_plangen_nast()
    # evaluate_styletran()