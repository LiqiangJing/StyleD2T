import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from pyossfs.oss_bucket_manager import OSSFileManager
import numpy as np
import pkg_resources
import fasttext
import pandas as pd
import jieba
import kenlm
import math

def lcs(a, b):
    """
    Compute the length of the longest common subsequence between two sequences.
    Time complexity: O(len(a) * len(b))
    Space complexity: O(min(len(a), len(b)))
    """
    # This is an adaptation of the standard LCS dynamic programming algorithm
    # tweaked for lower memory consumption.
    # Sequence a is laid out along the rows, b along the columns.
    # Minimize number of columns to minimize required memory
    if len(a) < len(b):
        a, b = b, a
    # Sequence b now has the minimum length
    # Quit early if one sequence is empty
    if len(b) == 0:
        return 0
    # Use a single buffer to store the counts for the current row, and
    # overwrite it on each pass
    row = [0] * len(b)
    for ai in a:
        left = 0
        diag = 0
        for j, bj in enumerate(b):
            up = row[j]
            if ai == bj:
                value = diag + 1
            else:
                value = max(left, up)
            row[j] = value
            left = value
            diag = up
    # Return the last cell of the last row
    return left



def rouge_l(peer, model):
    """
    Compute the ROUGE-L score of a peer with respect to one or more models.
    """
    matches = lcs(model, peer)
    recall_total = len(model)
    precision_total = len(peer)
    return _safe_f1(matches, recall_total, precision_total)
def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0
        
def _safe_f1(matches, recall_total, precision_total, alpha=0.5):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0

class Evaluator(object):
    def __init__(self):
        print('load reference!')
        test_asr = pd.read_csv('../data/asr_test.csv', delimiter='\t')
        test_weitao = pd.read_csv('../data/weitao_test.csv', delimiter='\t')
        self.asr_ref = [v for v in test_asr['text']]
        self.weitao_ref = [v for v in test_weitao['text']]

        acc_path = '../data/classifier_taobao.bin'
        self.classifier = fasttext.load_model(acc_path)

        self.ppl_model = kenlm.Model('../data/5_gram_ppl.bin')
        # print(self.classifier.labels)
        # ss = self.classifier.predict(['a book on a desk'])
        # print(ss)

    def nltk_bleu(self, texts_origin, text_transfered):
        # print(' '.join(jieba.cut('你好呀你今天有没有去')))
        texts_origin = [' '.join(jieba.cut(text_origin.strip())).split() for text_origin in texts_origin]
        # print(texts_origin)
        text_transfered = ' '.join(jieba.cut(text_transfered.strip())).split()
        # print(text_transfered)
        # exit()
        return [sentence_bleu(texts_origin, text_transfered) * 100, sentence_bleu(texts_origin, text_transfered, weights=(1,0,0,0)) * 100, \
        sentence_bleu(texts_origin, text_transfered, weights=(0,1,0,0)) * 100, sentence_bleu(texts_origin, text_transfered , weights=(0,0,1,0)) * 100, \
        sentence_bleu(texts_origin, text_transfered , weights=(0,0,0,1)) * 100] 

    def ref_bleu_asr(self, texts_asr):
        # print(len(texts_dii2sis))
        assert len(texts_asr) == len(self.asr_ref), 'Size of input differs from human reference file!'
        n = len(self.asr_ref)

        sum = [0, 0, 0, 0, 0]
        for x, y in zip(self.asr_ref, texts_asr):
            sum =  list(np.array(sum) + np.array(self.nltk_bleu([x], y)))
        return [sum[i] / n for i in range(len(sum))]

    def ref_bleu_weitao(self, texts_weitao):
        assert len(texts_weitao) == len(self.weitao_ref), 'Size of input differs from human reference file!'
        n = len(self.weitao_ref)
        sum = [0, 0, 0, 0, 0]
        
        for x, y in zip(self.weitao_ref, texts_weitao):
            sum = list(np.array(sum)) + np.array(self.nltk_bleu([x], y))   
        return [sum[i] / n for i in range(len(sum))]

    def style_check(self, text_transfered, style):
        text_transfered = ' '.join(jieba.cut(text_transfered.strip()))
        if text_transfered == '':
            return False
        label = self.classifier.predict([text_transfered])
        style_generation = label[0][0][0] == '__label__asr'

        return (style_generation == style)

    def acc_style(self, texts, styles):
        assert len(texts) == len(styles), 'Size of inputs does not match'
        count = 0
        for text, style in zip(texts, styles):
            if self.style_check(text, style):
                count += 1
        return count / len(texts)

    def acc_asr(self, texts):
        styles = [1] * len(texts)
        return self.acc_style(texts, styles)

    def acc_weitao(self, texts):
        styles = [0] * len(texts)
        return self.acc_style(texts, styles)

    def rouge_l_asr(self, models):
        curpus_size = len(self.asr_ref)
        rouge_score = 0
        for (peer, model) in zip(self.asr_ref, models):
            rouge_score += rouge_l(peer, model)
        return rouge_score / curpus_size
    
    def rouge_l_weitao(self, models):
        curpus_size = len(self.weitao_ref)
        rouge_score = 0
        for (peer, model) in zip(self.weitao_ref, models):
            rouge_score += rouge_l(peer, model)
        return rouge_score / curpus_size


    def ppl(self, texts_transfered):
        texts_transfered = [' '.join(jieba.cut(itm.strip())) for itm in texts_transfered]
        sum = 0
        words = []
        length = 0
        for i, line in enumerate(texts_transfered):
            words += [word for word in line.split()]
            length += len(line.split())
            score = self.ppl_model.score(line)
            sum += score
        return math.pow(10, -sum / length)

# if __name__ == "__main__":
    # e = Evaluator()
#     print(e.ppl(['这款粉饼的外观设计非常的简约，但是它的质地轻薄，上脸后不会有厚重感，而且它的外形设计也非常适合夏天使用，它的粉质也非同凡响，不会轻易的被晒伤，而是能够很好的遮盖住脸部的瑕疵，让你的肌肤更加的水润。']))
    # text = '这款粉饼的外观设计非常的简约，但是它的质地轻薄，上脸后不会有厚重感，而且它的外形设计也非常适合夏天使用，它的粉质也非同凡响，不会轻易的被晒伤，而是能够很好的遮盖住脸部的瑕疵，让你的肌肤更加的水润。'
    # text = '这款是纯天然的一个成分，它是可以滋养皮肤的，可以帮助我们去滋养我们的皮肤，让我们皮肤更加的水润，更加水嫩。这款面霜呢是一个纯天天然，不添加香精的。'
    # 2的粉饼，它是一个轻薄的粉底液，它可以帮助你在夏天的时候保湿，而不是说说说话，它就是一款遮瑕的粉扑，它不会让你觉得很闷，而保湿精华液的话呢，它会让人觉得非常轻薄。2这款的粉霜，它里面是有一个小小的一个粉饼。3的粉末，它能够帮助我们在夏季的时间里，让我们的肌肉更加有活力，而这款产品呢，是一种很好地遮瑕，不用担心会有卡粉的问题这款心机粉饼呢，就是为了遮瑕和保湿的效果而设计的。这'
    # print(e.classifier.predict([' '.join(jieba.cut(text))])[1][0][0])
    # print(e.acc_asr(e.asr_ref))
    # print(e.acc_weitao(e.weitao_ref))
    # print(e.ref_bleu_asr(e.asr_ref))
    # print(e.ppl(['这款安心乳是真空密封的，没有任何的化学添加剂、防腐剂、合成酒精、香料、色素、矿物油、荧光剂、苯氧乙醇、尼泊尔防腐乳等成分，非常的水润，质地非常清爽，很安全，非油皮也能用哦']))



