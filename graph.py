import json
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    print(train_asr.shape)
    train_asr = train_asr.fillna('')

    maidian = [v.split() for v in train_asr['maidian']]
    p_v = [v.split() for v in train_asr['p_v']]
    pos_p_v = [list2num(v.split()) for v in train_asr['p_v_pos']]
    pos_maidian = [list2num(v.split()) for v in train_asr['maidian_pos']]

    kvs = [maidian[i] + p_v[i] for i in range(len(maidian))]
    pos = [pos_maidian[i] + pos_p_v[i] for i in range(len(pos_maidian))]

    train_asr = pd.read_csv('../data/asr_train.csv', delimiter='\t')
    train_asr = train_asr.fillna('')
    print(train_asr.shape)

    maidian = [v.split() for v in train_asr['maidian']]
    p_v = [v.split() for v in train_asr['p_v']]
    pos_p_v = [list2num(v.split()) for v in train_asr['p_v_pos']]
    pos_maidian = [list2num(v.split()) for v in train_asr['maidian_pos']]

    kvs = kvs + [maidian[i] + p_v[i] for i in range(len(maidian))]
    pos = pos + [pos_maidian[i] + pos_p_v[i] for i in range(len(pos_maidian))]

    node_dict = {}
    edge_dict = {}
    pbar = tqdm(total=len(kvs))
    for i in range(len(kvs)):
        pbar.update(1)
        if len(kvs[i]) != len(pos[i]):
            print(i)
            print(kvs[i])
            print(len(kvs[i]))
            print(len(pos[i]))
            print('------------------')
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
    pbar.close()
    node_str = sortDictValue(node_dict)  # 节点
    edge_str = sortDictValue(edge_dict)   # 边

    # print(edge_str[:100])
    f = open('graph', 'w', encoding='utf8')
    json.dump(edge_str, f)
    f.close()
    print('save finished')
    # f.write(str(edge_str))
    # f.close()
    # f = open('nodes', 'w', encoding='utf8')
    # f.write(str(node_str))
    # f.close()

    keys = list(node_str.keys())

    print(len(keys))
    Adj = np.zeros((len(node_str),len(node_str)))
    pbar = tqdm(total=len(Adj))
    for i in range(len(Adj)):
        pbar.update(1)
        for j in range(len(keys)):
            if i!=j and keys[i]+' '+keys[j] in edge_str:
                Adj[i,j]=edge_str[keys[i]+' '+keys[j]]

            # print(i,j)
    pbar.close()
    print(Adj.shape)
    Adj = Adj / Adj.sum(0)
    Adj[np.isnan(Adj)] = 0
    # np.savetxt('adj_normal.txt',Adj)
    print(Adj)
    print(Adj.sum(0))
    print(len(node_str))
    print(len(edge_str))
    return node_str, edge_str

adj()