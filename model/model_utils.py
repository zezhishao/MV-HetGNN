import sys
import random
import numpy as np
import torch
import torch.nn.functional as F

from config import seed, reproducible


def get_input_feature_dim(feature_list):
    input_feature_dim = []
    for input_feature in feature_list:
        input_feature_dim.append(input_feature.shape[1])
    return input_feature_dim

def norm_adj(A, method='add', prefix='none'):
    if prefix == 'none':
        A = A
    elif prefix == 'relu':
        # A = F.relu(A)
        zeros = torch.zeros_like(A)
        A = torch.where(A < 0, zeros, A)
    else:
        raise Exception("error")

    # normalize adjacent matrix to get message passing matrix
    ## add
    if method == 'add':
        A = A
    ## avg(mean by degree)
    elif method == 'avg':
        diag = torch.sum(A, dim=1) + 1e-7
        A = A / diag
    # semi gcn
    elif method == 'semi':
        D = torch.diag(torch.pow(torch.sum(A, dim=1) + 1e-7, -0.5))
        A = adj = torch.mm(torch.mm(D, A), D)
    # eye
    elif method == 'eye':
        A = torch.eye(A.shape[0])
    else:
        raise Exception("error adj norm methods")
    return A

def norm_data(data_vec_tensor, method='sub_mean'):
    # cosine similarity is sensitive to `mean` value
    if method == 'sub_mean':
        data_vec_tensor_mean = torch.mean(data_vec_tensor, dim=0)  # mean of each dimension
        data_vec_tensor = (data_vec_tensor - data_vec_tensor_mean) * 10
        a = 1
    elif method == 'none':
        data_vec_tensor = data_vec_tensor
    else:
        raise Exception("error")
    return data_vec_tensor

def get_similarity_matrix(data_vec_tensor, meta_path_graph=None, norm_data_method='none', prefix='none', norm_adj_method='add'):
    if meta_path_graph is None:
        data_vec_tensor = norm_data(data_vec_tensor, method=norm_data_method)
        # 计算分母
        l2 = torch.norm(data_vec_tensor, dim=1, p=2) + 1e-7  # avoid 0, l2 norm
        l2_1 = torch.mm(l2.unsqueeze(dim=1), l2.unsqueeze(dim=1).T)
        # 计算分子
        l2_2 = torch.mm(data_vec_tensor, data_vec_tensor.T)
        # cos similarity affinity matrix 
        cos_affnity = l2_2 / l2_1
        adj = cos_affnity
    else:
        meta_path_graph = meta_path_graph - torch.diag(meta_path_graph.diag())
        meta_path_graph = F.normalize(meta_path_graph)
        meta_path_graph = meta_path_graph + torch.eye(meta_path_graph.shape[0], device=meta_path_graph.device)
        adj = meta_path_graph
    # adjacent matrix
    adj = norm_adj(adj, method=norm_adj_method, prefix=prefix)
    return adj

def cosine_similarity(x, y, meta_path_graph=None, norm_data_method='none', prefix='none', norm_adj_method='add'):
    # 计算分母
    l2_x = torch.norm(x, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_y = torch.norm(y, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_m = torch.matmul(l2_x.unsqueeze(dim=2), l2_y.unsqueeze(dim=2).transpose(1, 2))
    # 计算分子
    l2_z = torch.matmul(x, y.transpose(1, 2))
    # cos similarity affinity matrix 
    cos_affnity = l2_z / l2_m
    adj = cos_affnity
    return adj

def set_config(seed=seed):
    # print("运行结果是否保证可复现：{0}".format(Reproducible))
    if reproducible:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        pass