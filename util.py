import torch
from tqdm import tqdm
import copy
import json
import numpy as np
import math
from scipy.stats import kendalltau
is_cuda = torch.cuda.is_available()

def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x

def my_sample_gumbel(shape, eps=1e-20):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
    shape: list of integers
    eps: float, for numerical stability
    Returns:
    A sample of standard Gumbel random variables
    """
    #Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def my_sinkhorn(log_alpha, n_iters = 20):
    # torch version
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the successive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (element wise).
    -However, for stability, sinkhorn works in the log-space. It is only at
    return time that entries are exponentiated.
    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
    log_alpha: a 2D tensor of shape [N, N]
    n_iters: number of sinkhorn iterations (in practice, as little as 20
      iterations are needed to achieve decent convergence for N~100)
    Returns:
    A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
      converted to 3D tensors with batch_size equals to 1)
    """
    n = log_alpha.size()[1]
    log_alpha = log_alpha.view(-1, n, n)

    for i in range(n_iters):
        # torch.logsumexp(input, dim, keepdim, out=None)
        #Returns the log of summed exponentials of each row of the input tensor in the given dimension dim
        #log_alpha -= (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
        #log_alpha -= (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        #avoid in-place
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
    return torch.exp(log_alpha)
def my_gumbel_sinkhorn(log_alpha, temp=1.0, n_samples=1, noise_factor=1.0, n_iters=20, squeeze=True):
    """Random doubly-stochastic matrices via gumbel noise.
    In the zero-temperature limit sinkhorn(log_alpha/temp) approaches
    a permutation matrix. Therefore, for low temperatures this method can be
    seen as an approximate sampling of permutation matrices, where the
    distribution is parameterized by the matrix log_alpha
    The deterministic case (noise_factor=0) is also interesting: it can be
    shown that lim t->0 sinkhorn(log_alpha/t) = M, where M is a
    permutation matrix, the solution of the
    matching problem M=arg max_M sum_i,j log_alpha_i,j M_i,j.
    Therefore, the deterministic limit case of gumbel_sinkhorn can be seen
    as approximate solving of a matching problem, otherwise solved via the
    Hungarian algorithm.
    Warning: the convergence holds true in the limit case n_iters = infty.
    Unfortunately, in practice n_iter is finite which can lead to numerical
    instabilities, mostly if temp is very low. Those manifest as
    pseudo-convergence or some row-columns to fractional entries (e.g.
    a row having two entries with 0.5, instead of a single 1.0)
    To minimize those effects, try increasing n_iter for decreased temp.
    On the other hand, too-low temperature usually lead to high-variance in
    gradients, so better not choose too low temperatures.
    Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
      or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    temp: temperature parameter, a float.
    n_samples: number of samples
    noise_factor: scaling factor for the gumbel samples. Mostly to explore
      different degrees of randomness (and the absence of randomness, with
      noise_factor=0)
    n_iters: number of sinkhorn iterations. Should be chosen carefully, in
      inverse correspondence with temp to avoid numerical instabilities.
    squeeze: a boolean, if True and there is a single sample, the output will
      remain being a 3D tensor.
    Returns:
        sink: a 4D tensor of [batch_size, n_samples, N, N] i.e.
          batch_size *n_samples doubly-stochastic matrices. If n_samples = 1 and
          squeeze = True then the output is 3D.
        log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, N] of
          noisy samples of log_alpha, divided by the temperature parameter. Ifmy_invert_listperm
          n_samples = 1 then the output is 3D.
    """
    n = log_alpha.size()[1]
    log_alpha = log_alpha.view(-1, n, n)
    batch_size = log_alpha.size()[0]

    log_alpha_w_noise = log_alpha.repeat(n_samples, 1, 1)

    if noise_factor == 0:
        noise = 0.0
    else:
        noise = to_var(my_sample_gumbel([n_samples*batch_size, n, n])*noise_factor)

    log_alpha_w_noise = log_alpha_w_noise + noise
    log_alpha_w_noise = log_alpha_w_noise / temp

    my_log_alpha_w_noise = log_alpha_w_noise.clone()

    sink = my_sinkhorn(my_log_alpha_w_noise)
    if n_samples > 1 or squeeze is False:
        sink = sink.view(n_samples, batch_size, n, n)
        sink = torch.transpose(sink, 1, 0)
        log_alpha_w_noise = log_alpha_w_noise.view(n_samples, batch_size, n, n)
        log_alpha_w_noise = torch.transpose(log_alpha_w_noise, 1, 0)
    return sink, log_alpha_w_noise

def inv_soft_pers_flattened(soft_perms_inf):
    inv_soft_perms = torch.transpose(soft_perms_inf, 2, 3)
    inv_soft_perms = torch.transpose(inv_soft_perms, 0, 1)

    inv_soft_perms_flat = inv_soft_perms.view(-1, n_numbers, n_numbers)
    return inv_soft_perms_flat
    
def batch_preprocess(asr, weitao, config):
    asr_style_ids, asr_style_mask, asr_data_ids, asr_data_mask, asr_tgt, asr_tgt_mask, asr_rev_style_ids, asr_rev_style_mask, asr_pos = asr
    weitao_style_ids, weitao_style_mask, weitao_data_ids, weitao_data_mask, weitao_tgt, weitao_tgt_mask, weitao_rev_style_ids, weitao_rev_style_mask, weitao_pos = weitao
    style_labels = torch.cat((torch.ones(len(asr_data_ids)), torch.zeros(len(weitao_data_ids))))
    if config.use_gpu:
        weitao_style_ids, weitao_style_mask, weitao_data_ids, weitao_data_mask, weitao_tgt, weitao_tgt_mask, weitao_rev_style_ids, weitao_rev_style_mask, weitao_pos = \
            weitao_style_ids.cuda(), weitao_style_mask.cuda(), weitao_data_ids.cuda(), weitao_data_mask.cuda(), weitao_tgt.cuda(), weitao_tgt_mask.cuda(), weitao_rev_style_ids.cuda(), weitao_rev_style_mask.cuda(), weitao_pos.cuda()
        asr_style_ids, asr_style_mask, asr_data_ids, asr_data_mask, asr_tgt, asr_tgt_mask, asr_rev_style_ids, asr_rev_style_mask, asr_pos = \
            asr_style_ids.cuda(), asr_style_mask.cuda(), asr_data_ids.cuda(), asr_data_mask.cuda(), asr_tgt.cuda(), asr_tgt_mask.cuda(), asr_rev_style_ids.cuda(), asr_rev_style_mask.cuda(), asr_pos.cuda()
        style_labels = style_labels.cuda()
    # asr_input_ids = torch.cat((asr_style_ids,asr_data_ids), 1)
    # asr_mask = torch.cat((asr_style_mask, asr_data_mask), 1)

    # weitao_input_ids = torch.cat((weitao_style_ids, weitao_data_ids), 1)
    # weitao_mask = torch.cat((weitao_style_mask, weitao_data_mask), 1)

    # input_ids = torch.cat((asr_input_ids, weitao_input_ids), 0)
    # input_mask = torch.cat((asr_mask, weitao_mask), 0)

    style_ids = torch.cat((asr_style_ids, weitao_style_ids), 0)
    style_mask = torch.cat((asr_style_mask, weitao_style_mask), 0)

    rev_style_ids = torch.cat((asr_rev_style_ids, weitao_rev_style_ids), 0)
    rev_mask = torch.cat((asr_rev_style_mask, weitao_rev_style_mask), 0)

    data_ids = torch.cat((asr_data_ids, weitao_data_ids), 0)
    data_mask = torch.cat((asr_data_mask, weitao_data_mask), 0)

    tgt_ids = torch.cat((asr_tgt, weitao_tgt), 0)
    tgt_mask = torch.cat((asr_tgt_mask, weitao_tgt_mask), 0)

    pos_label = torch.cat((asr_pos, weitao_pos), 0)
    # print(style_ids.shape)
    # print(style_mask.shape)
    # print(data_ids.shape)
    # print(data_mask.shape)
    # print(tgt_ids.shape)
    # print(tgt_mask.shape)
    # exit()
    # return input_ids, input_mask, tgt_ids, tgt_mask
    
    return style_ids, style_mask, data_ids, data_mask, tgt_ids, tgt_mask, rev_style_ids, rev_mask, pos_label, style_labels

def shift_tokens_right(input_ids, token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
#   index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = torch.ones_like(prev_output_tokens[:, 0], device=prev_output_tokens.device) * token_id
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens

def tensor2texts(input_ids, tokenizer, skip_special_tokens=True):
    return [tokenizer.decode(w, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True).replace(' ','') for w in input_ids]

def save_model(epoch, model, score, config):
    torch.save(model.state_dict(), config.save_folder+'/'+str(epoch)+'_'+str(score)+'.pkl')
    return

def inference(data_loader, model, config, tokenizer): 
    batch_idx = 0
    data = []
    ori_sty=[]
    ori_gen = []
    ori_ref = []

    rev_sty = []
    rev_gen = []
    correct = 0
    total = 0
    acc = 0
    kender = []
    pbar = tqdm(total=len(data_loader), ncols=150)
    for style_ids, style_mask, data_ids, data_mask, tgt, tgt_mask, rev_style_ids, rev_style_mask, pos in data_loader:
        if config.use_gpu:
            style_ids, style_mask, data_ids, data_mask, tgt, tgt_mask, rev_style_ids, rev_style_mask, pos = \
                style_ids.cuda(), style_mask.cuda(), data_ids.cuda(), data_mask.cuda(), tgt.cuda(), tgt_mask.cuda(), rev_style_ids.cuda(), rev_style_mask.cuda(), pos.cuda()

        # input_ids = torch.cat((style_ids, data_ids), 1)
        # input_mask = torch.cat((style_mask, data_mask), 1)

        with torch.no_grad():
            output, pos_pre  = model.generate_text(
                input_ids=data_ids, attention_mask=data_mask, input_style_ids=style_ids, style_attention_mask=style_mask, pos_true=pos
            )
            rev_output, _ = model.generate_text(
                input_ids=data_ids, attention_mask=data_mask, input_style_ids=rev_style_ids, style_attention_mask=rev_style_mask, pos_true=pos
            )
            if config.planning and pos_pre is not None:
                correct += torch.sum((pos_pre == pos) & (pos != -1)).item()
                total += torch.sum( pos != -1).item()

                for i in range(len(pos_pre)):
                    pad_num = torch.sum(pos[i]!=-1)
                    kk = kendalltau(pos_pre[i][:-pad_num].cpu().numpy(), pos[i][:-pad_num].cpu().numpy())[0]
                    if math.isnan(kk):
                        # print(kk)
                        kk = 0
                    kender.append(kk)
            # output = model.generate_text(
            #     input_ids=data_ids, attention_mask=data_mask)
    
        data += tensor2texts(data_ids, tokenizer,skip_special_tokens=False)
        data = [data[i].replace('[PAD]','') for i in range(len(data))]
        ori_sty += tensor2texts(style_ids, tokenizer)
        ori_gen += output
        ori_ref += tensor2texts(tgt, tokenizer)

        rev_sty += tensor2texts(rev_style_ids, tokenizer)
        rev_gen += rev_output
        pbar.update(1)
        batch_idx += 1
    pbar.close()
    if config.planning and pos_pre is not None:
        acc = correct * 1.0 / total
    return data, ori_sty, ori_gen, ori_ref, rev_sty, rev_gen, acc, np.mean(kender)

def index_of_str(text, keyword):
    lt=text.split(keyword,1)
    if len(lt)==1:
        return -1
    return len(lt[0])
def ainb(v, vs):
    for ss in vs:
        if v == ss:
            return True
        if v in ss:
            return True
        if ss in v:
            return True
    return False

def filter():
    import pandas as pd
    train = pd.read_csv('../data/raw/weitao_test.csv', delimiter='\t')
    train = train.fillna('')

    for i, item in train.iterrows():
        p_v = item['p_v']
        p_v = p_v.split(' ')
        values_pv = [v.partition(':')[2] for v in p_v]
        jilu = []
        for v in values_pv:
            if v != '':
                pos = index_of_str(item['text'], v)
                if pos == -1:
                    jilu.append(v)
        p_v = [p_v[i] for i in range(len(values_pv)) if values_pv[i] not in jilu]
        item['p_v'] = ' '.join(p_v)

        p_v = item['maidian']
        p_v = p_v.split(' ')
        values = [v.partition(':')[2] for v in p_v]
        jilu = []
        for v in values:
            if v != '':
                pos = index_of_str(item['text'], v)
                if pos == -1 or ainb(v, values_pv):
                    jilu.append(v)
        p_v = [p_v[i] for i in range(len(values)) if values[i] not in jilu]
        item['maidian'] = ' '.join(p_v)
    train.to_csv('weitao_test.csv', sep='\t', index=False)

def get_order(ll):
    re = copy.deepcopy(ll)
    ind = sorted(range(len(ll)), key=lambda k: ll[k])
    for i in range(len(ind)):
        re[ind[i]] = i
    return re

def get_position():
    import pandas as pd
    train_asr = pd.read_csv('./weitao_test.csv', delimiter='\t')
    train_asr = train_asr.fillna('')

    # train_weitao = pd.read_csv('../data/weitao_train.csv', delimiter='\t')
    # print(train_asr.head())
    maidian = [v for v in train_asr['maidian']]
    p_v = [v for v in train_asr['p_v']]
    text = [v for v in train_asr['text']]
    num = 0
    total = 0
    positions_pv = []
    
    print(len(p_v))
    for i in range(len(p_v)):
        values = p_v[i].split(' ')
        values = [v.partition(':')[2] for v in values]
        position = []
        
        for v in values:
            if v != '':
                pos = index_of_str(text[i], v)
                position.append(pos)
                total += 1
        # positions_pv.append(' '.join(str(p) for p in position))
        positions_pv.append(position)
        # print(position)
        # print(sorted(position))
        # if i >5:
        #     exit()
    print(len(positions_pv))
    # train_asr['p_v_pos']=[' '.join(str(p))]
    # print(num)
    # print(total)
    num = 0
        # if i>3:
            # break
    total = 0
    positions_maidian = []
    for i in range(len(maidian)):
        values = maidian[i].split(' ')
        values = [v.partition(':')[2] for v in values]
        # print(values)
        position = []
        for v in values:
            if v!='':
                total +=1
                pos = index_of_str(text[i], v)
                position.append(pos)
        # positions_maidian.append(' '.join(str(p) for p in position))
        positions_maidian.append(position)
    positions_all = [positions_maidian[i] + positions_pv[i] for i in range(len(positions_maidian))]
    print(positions_all[:3])
    positions_all = [get_order(ll) for ll in positions_all]
    print(positions_all[:3])
    print('------------------')
    print(len(positions_maidian[1]))

    print(positions_all[1][:len(positions_maidian[1])])
    print(positions_all[1][len(positions_maidian[1]):])
    positions_maidian1 = []
    positions_pv1 = []
    for i in range(len(positions_all)):
        positions_maidian1.append(' '.join(str(p) for p in positions_all[i][:len(positions_maidian[i])]))
        positions_pv1.append (' '.join(str(p) for p in positions_all[i][len(positions_maidian[i]):]))
    print(positions_maidian1[:3])
    print(positions_pv1[:3])
    train_asr['maidian_pos'] = positions_maidian1
    train_asr['p_v_pos'] = positions_pv1
    # print(num)
    # print(total)
    train_asr.drop(train_asr[(train_asr.p_v_pos == '') & (train_asr.maidian_pos == '')].index,axis=0,inplace=True)
    train_asr.to_csv('weitao_test.csv', index=False, sep='\t')
    #     if i > 3:
    #         break
def tongji():
    import pandas as pd
    import numpy as np
    train_asr = pd.read_csv('../data/asr_train.csv', delimiter='\t')
    
    train_asr = train_asr.fillna('')
    
    len_pv = [len(v) for v in train_asr['p_v']]
    len_maidian = [len(v) for v in train_asr['maidian']]
    len_title = [len(v) for v in train_asr['title']]
    len_txt = [len(v) for v in train_asr['text']]
    print(max(len_pv))
    print(max(len_maidian))
    print(max(len_title))
    print(max(len_txt))
    print(sum(len_txt)/len(len_txt))
    print(np.percentile(np.array(len_txt), 95))


def list2num(ll):
    return [int(v) for v in ll]
def sortDictValue(dict, is_reverse=True):
    '''
    将字典按照value排序
    :param dict: 待排序的字典
    :param is_reverse: 是否按照倒序排序
    '''
    # 对字典的值进行倒序排序,items()将字典的每个键值对转化为一个元组,key输入的是函数,item[1]表示元组的第二个元素,reverse为真表示倒序
    tups = sorted(dict.items(), key=lambda item: item[1], reverse=is_reverse)
    s = {}
    for tup in tups:  # 合并成csv需要的逗号分隔格式
        s[tup[0]] = tup[1]
    return s

    
def adj():
    import pandas as pd
    train_asr = pd.read_csv('../data/weitao_train.csv', delimiter='\t')
    train_asr = train_asr.fillna('')

    maidian = [v.split() for v in train_asr['maidian']]
    p_v = [v.split() for v in train_asr['p_v']]
    pos_p_v = [list2num(v.split()) for v in train_asr['p_v_pos']]
    pos_maidian = [list2num(v.split()) for v in train_asr['maidian_pos']]

    kvs = [maidian[i] + p_v[i] for i in range(len(maidian))]
    pos = [pos_maidian[i] + pos_p_v[i] for i in range(len(pos_maidian))]

    train_asr = pd.read_csv('../data/asr_train.csv', delimiter='\t')
    train_asr = train_asr.fillna('')

    maidian = [v.split() for v in train_asr['maidian']]
    p_v = [v.split() for v in train_asr['p_v']]
    pos_p_v = [list2num(v.split()) for v in train_asr['p_v_pos']]
    pos_maidian = [list2num(v.split()) for v in train_asr['maidian_pos']]

    kvs = kvs + [maidian[i] + p_v[i] for i in range(len(maidian))]
    pos = pos + [pos_maidian[i] + pos_p_v[i] for i in range(len(pos_maidian))]
    # print(kvs[:2])
    # print(pos[:2])

    node_dict = {}
    edge_dict = {}

    for i in range(len(kvs)):
        for index, node in enumerate(kvs[i]):
            if node not in node_dict:
                node_dict[node] = 1
            else:
                node_dict[node] += 1

            for j in range(index+1, len(kvs[i]), 1):
                A, B = node, kvs[i][j]
                # print(kvs[i])
                # print(pos[i])
                # print(i)
                weight = 1.0 / (pos[i][j] - pos[i][index])
                if weight > 0:
                    edge = A + ' ' + B
                else:
                    edge = B + ' ' + A
                    weight = -weight
                if edge not in edge_dict:
                    edge_dict[edge] = weight
                else:
                    edge_dict[edge] += weight
    # 对得到的字典按照value进行排序
    node_str = sortDictValue(node_dict)  # 节点
    edge_str = sortDictValue(edge_dict)   # 边


    # print(edge_str[:100])
    # f = open('graph', 'w', encoding='utf8')
    # # json.dump(edge_str, f)
    # # f.close()
    # f.write(str(edge_str))
    # f.close()
    # f = open('nodes', 'w', encoding='utf8')
    # f.write(str(node_str))
    # f.close()

    keys = list(node_str.keys())
    # print(keys[:10])
    # print(node_str)
    # print
    # exit()
    print(len(keys))
    Adj = np.zeros((len(node_str),len(node_str)))
    for i in range(len(Adj)):
        for j in range(len(keys)):
            if i!=j and keys[i]+' '+keys[j] in edge_str:
                Adj[i,j]=edge_str[keys[i]+' '+keys[j]]

            # print(i,j)
    # print(Adj.shape)
    Adj = Adj / Adj.sum(0)
    Adj[np.isnan(Adj)] = 0
    # np.savetxt('adj_normal.txt',Adj)
    print(Adj)
    print(Adj.sum(0))
    print(len(node_str))
    print(len(edge_str))
    return node_str, edge_str

def tongji_testset():
    import json
    import pandas as pd
    f = open('graph', 'r')
    ff = f.readline()
    # print(ff)
    s1 = eval(ff)

    # print(s1.keys())
    test_asr = pd.read_csv('../data/weitao_test.csv', delimiter='\t')
    test_asr = test_asr.fillna('')

    maidian = [v.split() for v in test_asr['maidian']]
    p_v = [v.split() for v in test_asr['p_v']]
    kvs = [maidian[i] + p_v[i] for i in range(len(maidian))]
    number = 0
    for i in range(len(kvs)):
        for j in range(len(kvs[i])):
            for k in range(len(kvs[i])):
                if j==k or kvs[i][j] + ' ' + kvs[i][k] in s1:
                    number += 1
    print(number)
    print(number/len(kvs))
    leng = [len(v) for v in kvs]
    print(sum(leng)/len(leng))    

def tongji_phrase():
    import pandas as pd
    train_asr = pd.read_csv('../data/weitao_test.csv', delimiter='\t')
    train_asr = train_asr.fillna('')
    pos_pv = [v.split() for v in train_asr['p_v_pos']]
    pos_maidian = [v.split() for v in train_asr['maidian_pos']]
    pos = [pos_pv[i]+pos_maidian[i] for i in range(len(pos_pv))]
    lens = [len(pos[i]) for i in range(len(pos))]
    print(max(lens))
# if __name__ == "__main__":

    # filter()
    # get_position()
    # tongji()
    # adj()
    # tongji_testset()
    # tongji_phrase()
    