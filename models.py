import torch
from typing import Dict, List, Optional, Tuple
from transformers.modeling_bart import _prepare_bart_decoder_inputs, PretrainedBartModel, _filter_out_falsey_values, _make_linear_from_emb, _reorder_buffer
import torch.nn.functional as F
from torch import nn
from util import shift_tokens_right, tensor2texts
from transformers.configuration_bart import BartConfig
from copy import deepcopy
from torch.nn.parameter import Parameter
import math
import numpy as np
import scipy.sparse as sp
from util import my_gumbel_sinkhorn, inv_soft_pers_flattened
import random

def adj_normalize(Adj_):
    '''
    Adj batch, num, num
    '''
    Adj = Adj_.cpu().numpy()
    for i in range(len(Adj)):
        Adj[i] = normalize(Adj[i])
    return torch.tensor(Adj, device=Adj_.device, dtype=torch.float32)

def normalize(mx):

    """Row-normalize sparse matrix
    """
    rowsum = mx.sum(1) #行求和
    r_inv = np.power(rowsum, -1).flatten()

    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
 
class GraphNet(torch.nn.Module):
    def __init__(self, d=768, bias=True):
        super().__init__()
        self.W_l = Parameter(torch.FloatTensor(d, d))
        self.W_l1 = Parameter(torch.FloatTensor(d, d))
        self.W_l2 = Parameter(torch.FloatTensor(d, d))
        if bias:
            self.bias = Parameter(torch.FloatTensor(d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # self.W_l1 = torch.nn.Linear(d, d, bias=False)
        # self.W_l2 = torch.nn.Linear(d, d, bias=False)
        # self.W_l = torch.nn.Linear(d, d, bias=True)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W_l.size(1))
        self.W_l.data.uniform_(-stdv, stdv)
        self.W_l1.data.uniform_(-stdv, stdv)
        self.W_l2.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, adj, bidirection=False): 
        '''
            X: batch, node_num, d
            adj: batch, node_num, node_num

            A X nodenum, d
            A' X nodenum, d
            Wx + Ax + A'x
        '''
        e1 = torch.matmul(X, self.W_l1) # batch, node, d
        e11 = torch.bmm(adj_normalize(adj), e1) # batch, node, d
        X1 = torch.matmul(X, self.W_l) # batch, node, d


        if bidirection:
            e2 = torch.matmul(X, self.W_l1)
            e21 = torch.bmm(adj_normalize(adj.transpose(1,2)), e2) # batch, node, d
            h = X1 + e11 + e21
        else:
            h = X1 + e11
        h_l = torch.nn.functional.leaky_relu(h)
        return h_l #batch, node, d
def get_adj(relations, nodes, node_num=30):
    '''
    nodes: batch, node_num
    '''
    adj = np.zeros((len(nodes), node_num, node_num)) # batch, node_num, node_num
    for i in range(len(nodes)):
        for j in range(len(nodes[i])):
            for k in range(len(nodes[i])):
                if k != j and nodes[i][j] +' '+nodes[i][k] in relations:
                    adj[i,j,k] = relations[nodes[i][j] +' '+nodes[i][k]]
    return adj
def get_pos(text_ids, sep_id=102):
    pos = [0]
    for i in range(len(text_ids)):
        if sep_id == text_ids[i]:
            pos.append(i+1)
    return pos
def get_relations():
    f = open('graph', 'r')
    ff = f.readline()
    s1 = eval(ff)
    return s1


class Plan_GRU(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.W = nn.Linear(hidden_size, hidden_size)
        start = torch.rand(hidden_size)
        self.S = torch.nn.Parameter(start)

    def forward(self, input_hidden, tgt, max_length=30, teacher_forcing_ratio=1.0, pos_label=None, generate=False):
        '''
            input_hidden: batch, length 30, hiddensize    
            mask: batch
            pos_label: batch, num. xxx -1
        '''

        input = self.S.expand(input_hidden.shape[0], input_hidden.shape[-1])  # batch, hidden
        hidden = torch.mean(input_hidden, 1) # batch, hidden_size
        hidden = hidden.unsqueeze(0) # 1, batch, hidden_size
        predictions = [] ## prediction results
        infer = []
        flag = torch.zeros(input_hidden.shape[0], input_hidden.shape[1], device=input_hidden.device) #batch, len 记录模型是否的
        for i in range(max_length):
            input = input.unsqueeze(1) # batch, 1, hidden_size
            output, hidden = self.gru(input, hidden) #  output (batch, 1, dimension) hidden (1, batch, dimension)
            pre = torch.bmm(self.W(output), input_hidden.permute(0, 2, 1)) # batch, 1, len
            # print(pre.shape)
            pre = pre.squeeze(1) # batch, len
            predictions.append(pre)

            teacher_force = random.random() < teacher_forcing_ratio

            if teacher_force:
                input = tgt[:, i]
                for ii in range(len(input)):
                    if i < len(pos_label[ii])-(pos_label[ii]==-1).sum():
                        ind = pos_label[ii].tolist().index(i)
                        # flag[ii, ind] = 1

                result = pre.clone().detach()
                result[flag.bool()] = float("-inf")
                for ii in range(len(flag)):
                    ss = len(pos_label[ii])-(pos_label[ii]==-1).sum()
                    result[ii, ss: ] = float("-inf") # batch, num 超出长度的都置为0

                top1 = torch.softmax(result, 1) # batch, num
                infer.append(top1)
                top1 = top1.argmax(1) #batch
                for ii in range(len(flag)):
                    flag[ii, top1[ii]] = 1

            else:
                result = pre.clone()
                # print(flag)
                # print(result)
                result[flag.bool()] = float("-inf")
                for ii in range(len(flag)):
                    ss = len(pos_label[ii])-(pos_label[ii]==-1).sum()
  
                    result[ii, ss: ] = float("-inf") # batch, num
                top1 = torch.softmax(result, 1) # batch, num
                infer.append(top1)
                top1 = top1.argmax(1) #batch
                for ii in range(len(flag)):
                    flag[ii, top1[ii]] = 1
                input = tgt[:, i].clone() #batch, dimension
                for ii in range(len(top1)):
                    input[ii] = input_hidden[ii,top1[ii]].clone()
                # input batch, dimension
                #if teacher forcing, use actual next token as next input
                #if not, use predicted token
        # outputs = torch.stack(outputs, dim=0) # , batch, len, dim
        # outputs = torch.bmm(self.W(outputs), input_hidden.permute(0,2,1)) #batch, len, num
        if not generate:
            outputs = torch.stack(predictions, dim=1) # batch, len, num(logtis)
            pre = torch.stack(infer, 1).clone().detach()
            return outputs, pre
        else:
            outputs = torch.stack(infer, 1)
            return outputs


class BartModel(PretrainedBartModel):
    def __init__(self, bart, args, tokenizer):
        super().__init__(bart.model.config)
        config = bart.config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        
        self.relations = get_relations()

        self.shared = bart.model.shared
        self.tokenizer = tokenizer
        self.encoder = bart.model.encoder
        self.style_encoder = deepcopy(bart.model.encoder)
        self.decoder = bart.model.decoder
        self.args = args
        if args.planning:
            self.Graph = GraphNet()
            self.Sort_Net = torch.nn.Sequential(
                nn.Linear(768, 128),
                nn.Tanh(),
                nn.Linear(128, 30)
            )
            self.planning_net = Plan_GRU(768)
            # self.Plan_Net = torch.nn.Linear(768, 30)

        if args.style_disentange:
            self.Gate = torch.nn.Sequential(
                torch.nn.Linear(768, 768),
                nn.Tanh(),
                nn.Linear(768,768),
                nn.Sigmoid()
            )
            self.Classifier = torch.nn.Sequential(
                nn.Linear(768,768),
                nn.Tanh(),
                nn.Linear(768, 2)
            )
    def forward(
        self,
        input_ids,
        input_style_ids=None,
        attention_mask=None,
        style_attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs = None,
        encoder_outputs_all=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
        style_hidden_states=None,
        nodes=None,
        pos=None,
        pos_label=None,
        relations=None,
        data_ids=None,
        input_hidden_states=None,
        node_mask=None
    ):
        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        pre_pos = None
        out = None
        input_attention_mask = attention_mask

        if self.args.planning and input_hidden_states is None:

            ## select the word
            nodes_emb = torch.zeros((input_ids.shape[0],self.args.num_nodes, 768), device=input_ids.device) #batch,30, dimension
            node_mask = torch.zeros((input_ids.shape[0],self.args.num_nodes), device=input_ids.device)

            inputs_embeds1 = self.shared(input_ids) * self.encoder.embed_scale
            inputs_embeds = inputs_embeds1.clone()

            inputs_embeds_graph = self.shared(input_ids) * self.encoder.embed_scale
            inputs_embeds_graph = inputs_embeds_graph.detach()
            
            kv_start = []
            for i in range(len(input_ids)):
                kv_start.append(get_pos(input_ids[i]))

            for i in range(len(inputs_embeds)):
                leng = 0
                for j in range(len(pos_label[i])):
                    if pos_label[i, j] == -1:
                        break
                    
                    ind = pos_label[i].tolist().index(j)
                    # nodes_emb[i,j] = torch.mean(inputs_embeds1[i, kv_start[i][ind]: kv_start[i][ind+1]].clone(), dim=0)
                    nodes_emb[i,j] = torch.mean(inputs_embeds_graph[i, kv_start[i][j]: kv_start[i][j+1]].clone(), dim=0) ## new

                    node_mask[i,j] = 1

                    # ll = kv_start[i][ind+1] - kv_start[i][ind]
                    # inputs_embeds[i, leng:leng+ll] = inputs_embeds1[i, kv_start[i][ind]: kv_start[i][ind+1]] ## GT
                    # leng += ll

            if self.args.graph:
                adj = get_adj(self.relations, nodes)
                adj = torch.tensor(adj, device=data_ids.device)
                nodes_emb = self.Graph(nodes_emb, adj)
            
            target_emb = nodes_emb.clone().detach()
            for i in range(len(nodes_emb)):
                for j in range(len(pos_label[i])):
                    if pos_label[i, j] == -1:
                        break
                    ind = pos_label[i].tolist().index(j)
                    target_emb[i, j] = nodes_emb[i, ind].clone().detach()

            out, pre = self.planning_net(input_hidden=nodes_emb, tgt=target_emb, teacher_forcing_ratio=self.args.teacher_force, pos_label=pos_label)
            
            pre_pos = pre.argmax(-1) ## batch， len
            # print(pre_pos)
            # print(pos_label)
            # exit()
            for i in range(len(inputs_embeds)):
                leng = 0
                for j in range(len(pre_pos[i])):
                    if pos_label[i, j] == -1:
                        break
                    
                    ind = pre_pos[i].tolist().index(j)

                    ll = kv_start[i][ind+1] - kv_start[i][ind]
                    inputs_embeds[i, leng:leng+ll] = inputs_embeds1[i, kv_start[i][ind]: kv_start[i][ind+1]] 
                    leng += ll



        elif not self.args.planning and input_hidden_states is None:
            inputs_embeds = None

    ############ Add ########################################################################
        # if input_hidden_states is None:
        #     inputs_embeds1 = self.shared(input_ids) * self.encoder.embed_scale
        #     inputs_embeds = inputs_embeds1.clone()

        #     kv_start = []
        #     for i in range(len(input_ids)):
        #         kv_start.append(get_pos(input_ids[i]))

        #     for i in range(len(inputs_embeds)):
        #         leng = 0
        #         for j in range(len(pos_label[i])):
        #             if pos_label[i, j] == -1:
        #                 break
                        
        #             ind = pos_label[i].tolist().index(j)
        #             ll = kv_start[i][ind+1] - kv_start[i][ind]
        #             inputs_embeds[i, leng:leng+ll] = inputs_embeds1[i, kv_start[i][ind]: kv_start[i][ind+1]] ## GT
        #             leng += ll
            
    ############ Add ########################################################################

        assert decoder_input_ids is not None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
            input_hidden_states = encoder_outputs[0]

        assert isinstance(encoder_outputs, tuple)
            # pre_pos = self.Sort_Net(nodes) # batch, N, N
        logits_style = None
        if style_hidden_states is None:
            encoder_style = self.style_encoder(input_ids=input_style_ids, attention_mask=style_attention_mask)
            # style_emb = encoder_style[0]
        hidden_states = None
        if self.args.style_disentange and style_hidden_states is None:
            # input_g = torch.cat((), 1)
            input_g = encoder_style[0]
            g = self.Gate(input_g) # batch, L, dimension
            h = encoder_style[0] * g
            hidden_states = torch.mean(h, 1) # batch, dimension
            logits_style = self.Classifier(hidden_states)
            style_hidden_states = hidden_states.unsqueeze(1) # batch, 1 , dimension
        else:
            if style_hidden_states is None:
                style_hidden_states = torch.mean(encoder_style[0], 1).unsqueeze(1)
            # print(style_emb.shape)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print(input_hidden_states[0, :40, 0])

        decoder_outputs = self.decoder(
                    decoder_input_ids,
                    torch.cat((style_hidden_states, input_hidden_states), 1),
                    # encoder_outputs[0],
                    torch.cat((attention_mask[:,:style_hidden_states.shape[1]], input_attention_mask), 1),
                    # attention_mask,
                    decoder_padding_mask,
                    decoder_causal_mask=causal_mask,
                    decoder_cached_states=decoder_cached_states,
                    use_cache=use_cache
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
 
        # pre_pos = pre_pos if pre_pos is not None else False
        # logits_style = logits_style  if logits_style is not None else False
        return decoder_outputs + encoder_outputs + (out,) + (pre_pos,) + (logits_style,) + (hidden_states,)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly

def index_id(ll, idx):
    # print(ll)
    pos = []
    for i in range(len(ll)):
        if ll[i] == idx:
            pos.append(i)
    # print(pos)
    return pos
class MyModel(PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, args, tokenizer, bart):
        super().__init__(bart.config)
        self.args = args
        base_model = BartModel(bart, args, tokenizer)
        self.model = base_model
        # self.model = bart.model
        self.tokenizer = tokenizer
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def forward(
        self,
        input_ids,
        input_style_ids=None,
        attention_mask=None,
        style_attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs = None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
        style_hidden_states=None,
        pos=None,
        pos_label=None,
        nodes=None,
        relations=None,
        data_ids=None,
        input_hidden_states=None,
        node_mask=None,
    ):  

        if self.args.planning and input_hidden_states is None:
            if data_ids is None:
                data_ids = input_ids
            data = [self.tokenizer.decode(w, skip_special_tokens=False).replace(' ','') for w in data_ids]
            # print(data)
            nodes = [w.split('[SEP]')[:-1] for w in data]
            pos=[]
            for i in range(len(data_ids)):
                pos.append(index_id(data_ids[i], self.tokenizer.sep_token_id))
            # adj = get_adj(relations, nodes, )
            # print(nodes)
            # exit()

        outputs = self.model(
            input_ids,
            input_style_ids=input_style_ids,
            attention_mask=attention_mask,
            style_attention_mask=style_attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            style_hidden_states=style_hidden_states,
            pos=pos,
            nodes=nodes,
            relations=relations,
            pos_label=pos_label,
            data_ids=data_ids,
            input_hidden_states=input_hidden_states,
            node_mask=node_mask
        )
        
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]
        return outputs

    def generate_text(self, input_ids, attention_mask, input_style_ids=None, style_attention_mask=None, eval_beams=1, early_stopping=True, pos_true = None):

        encoder_style = self.model.style_encoder(input_ids=input_style_ids, attention_mask=style_attention_mask)
        node_mask=None
            # style_emb = encoder_style[0]
        if self.args.style_disentange:
            input_g = encoder_style[0]
            g = self.model.Gate(input_g) # batch, L, dimension
            h = encoder_style[0] * g
            hidden_states = torch.mean(h, 1) # batch, dimension
            # logits_style = self.Classifier(hidden_states)
            style_hidden_states = hidden_states.unsqueeze(1) # 
        else:
            style_hidden_states = torch.mean(encoder_style[0], 1).unsqueeze(1)
        if self.args.planning:
            pos_label = pos_true
            nodes_emb = torch.zeros((input_ids.shape[0],self.args.num_nodes, 768), device=input_ids.device) #batch, 30, dimension
            node_mask = torch.zeros((input_ids.shape[0],self.args.num_nodes), device=input_ids.device)

            inputs_embeds1 = self.model.shared(input_ids) * self.model.encoder.embed_scale
            inputs_embeds = inputs_embeds1.clone()
            kv_start = []
            for i in range(len(input_ids)):
                kv_start.append(get_pos(input_ids[i]))

            for i in range(len(inputs_embeds)):
                leng = 0
                for j in range(len(pos_label[i])):
                    if pos_label[i, j] == -1:
                        break
                    # ind = pos_label[i].tolist().index(j)
                    # nodes_emb[i,j] = torch.mean(inputs_embeds1[i, kv_start[i][ind]: kv_start[i][ind+1]].clone(), dim=0)
                    nodes_emb[i,j] = torch.mean(inputs_embeds1[i, kv_start[i][j]: kv_start[i][j+1]].clone(), dim=0) #add

                    node_mask[i,j] = 1

                    # ll = kv_start[i][ind+1] - kv_start[i][ind]
                    # inputs_embeds[i, leng:leng+ll] = inputs_embeds1[i, kv_start[i][ind]: kv_start[i][ind+1]]
                    # leng += ll

            data = [self.tokenizer.decode(w, skip_special_tokens=False).replace(' ','') for w in input_ids]
            nodes = [w.split('[SEP]')[:-1] for w in data]
            # pos=[]
            # for i in range(len(input_ids)):
            #     pos.append(index_id(input_ids[i], self.tokenizer.sep_token_id))
            if self.args.graph:
                adj = get_adj(self.model.relations, nodes)
                adj = torch.tensor(adj, device=input_ids.device)
                nodes_emb = self.model.Graph(nodes_emb, adj)
    
            target_emb = nodes_emb.clone().detach()
            for i in range(len(nodes_emb)):
                for j in range(len(pos_label[i])):
                    if pos_label[i, j] == -1:
                        break
                    ind = pos_label[i].tolist().index(j)
                    target_emb[i, j] = nodes_emb[i, ind].clone().detach()

            # target_emb = None
            out = self.model.planning_net(input_hidden=nodes_emb, tgt=target_emb, teacher_forcing_ratio=0, pos_label=pos_label, generate=True)

            pre_pos = out.argmax(-1) ## batch， len

            pos_label = pre_pos
            
            inputs_embeds1 = self.model.shared(input_ids) * self.model.encoder.embed_scale
            inputs_embeds = inputs_embeds1.clone()
            
            # pos_label = pos_true ## GT
            for i in range(len(inputs_embeds)):
                leng = 0
                # print(pos_label[i])
                for j in range(len(pos_label[i])):
                    if j >= len(nodes[i]):
                        break
                    ind = pos_label[i].tolist().index(j)
                    ll = kv_start[i][ind+1] - kv_start[i][ind]
                    inputs_embeds[i, leng:leng+ll] = inputs_embeds1[i, kv_start[i][ind]: kv_start[i][ind+1]]
                    leng += ll
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
            input_hidden_states = encoder_outputs[0]

        else:
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            input_hidden_states = encoder_outputs[0]
            pos_label = None

        

########################### Add  ###########################
        # kv_start = [] 

        # input_ids_copy =  input_ids.clone()
        # # input_order = input_ids.clone()
        # for i in range(len(input_ids)):
        #     kv_start.append(get_pos(input_ids[i])) 
        # pos_label = pos_true ## GT

        # inputs_embeds1 = self.model.shared(input_ids_copy) * self.model.encoder.embed_scale

        # inputs_embeds = inputs_embeds1.clone()
        # data = [self.tokenizer.decode(w, skip_special_tokens=False).replace(' ','') for w in input_ids]
        # nodes = [w.split('[SEP]')[:-1] for w in data]

        # pos_label = pos_true ## GT
        # for i in range(len(inputs_embeds)):
        #     leng = 0
        #     for j in range(len(kv_start[i])-1):
        #         ind = pos_label[i].tolist().index(j)
        #         ll = kv_start[i][ind+1] - kv_start[i][ind]
        #         inputs_embeds[i, leng:leng+ll, :] = inputs_embeds1[i, kv_start[i][ind]: kv_start[i][ind+1], :]
        #         leng += ll

        # encoder_outputs = self.model.encoder(input_ids, attention_mask, inputs_embeds=inputs_embeds)
        # input_hidden_states = encoder_outputs[0]

########################### Add  ###########################

        generate_ids = self.generate(
            input_ids, 
            attention_mask=attention_mask, 
            input_style_ids=input_style_ids, 
            style_attention_mask=style_attention_mask, 
            use_cache=True,
            decoder_start_token_id=self.config.bos_token_id, 
            num_beams=eval_beams, 
            max_length=self.args.max_length,
            # max_length=2,
            early_stopping=early_stopping,
            encoder_outputs=encoder_outputs,
            style_hidden_states=style_hidden_states,
            data_ids=input_ids,
            input_hidden_states=input_hidden_states,
        )
        # print(input_hidden_states[0, :40, 0])
        # print(generate_ids)
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(' ','') for w in generate_ids], pos_label
        
    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, style_hidden_states, data_ids, input_hidden_states, node_mask=None, input_style_ids=None, style_attention_mask=None, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "input_style_ids": input_style_ids,
            "style_attention_mask": style_attention_mask, 
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "data_ids": data_ids,
            "attention_mask": attention_mask,
            "input_hidden_states": input_hidden_states,
            "style_hidden_states": style_hidden_states,
            "node_mask": node_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # def get_output_embeddings(self):
    #     return _make_linear_from_emb(self.shared)  # make it on the fly


    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def prepare_logits_for_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly

def greedy(out,nodes):
    '''out: batch,  len, len'''
    result = torch.zeros_like(out.argmax(-1))
    for i in range(len(out)):
        listed = []
        for j in range(nodes[i]):
            results[i,j] = out[i,j]
