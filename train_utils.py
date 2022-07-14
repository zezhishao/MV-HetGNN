import pickle
import random
from dgl import data
import numpy as np
import torch
import os
from dataset.dataset_utils  import  load_LastFM_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from model.model_utils import get_input_feature_dim
from config import home_path, reproducible, seed
import torch as th
import torch
import torch.nn.functional as F


def get_lastfm_node_feat(type_mask, use_node_features, device):
    features_list = []
    input_feature_dims = []
    for i in range(len(use_node_features)):
        dim = (type_mask == i).sum()
        input_feature_dims.append(dim)
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to_dense().to(device))
    return features_list, input_feature_dims

def save_embeddings(test_embeddings, test_labels_list, dataset):
    pickle.dump(test_embeddings.cpu().numpy(), open('checkpoint/test_embeddings_' + dataset + '.pkl', 'wb'))
    pickle.dump(torch.LongTensor(test_labels_list).cpu().numpy(), open('checkpoint/test_label_' + dataset + '.pkl', 'wb'))

def print_test_result(svm_macro_f1_lists, svm_micro_f1_lists, nmi_mean_list, nmi_std_list, ari_mean_list, ari_std_list):
       # print out a summary of the evaluations
    svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
    svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
    nmi_mean_list = np.array(nmi_mean_list)
    nmi_std_list = np.array(nmi_std_list)
    ari_mean_list = np.array(ari_mean_list)
    ari_std_list = np.array(ari_std_list)

    print('----------------------------------------------------------------')
    print('SVM tests summary')
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means tests summary')
    print('NMI: {:.6f}~{:.6f}'.format(
        nmi_mean_list.mean(), nmi_std_list.mean()))
    print('ARI: {:.6f}~{:.6f}'.format(
        ari_mean_list.mean(), ari_std_list.mean()))
    return nmi_mean_list.mean()
    
def cluster_loss(embeddings, labels):
    """
    计算聚类Loss，同类节点之间更加紧密，不同类节点之间更疏远。
    Args:
        - embeddings: tensor size: batch_size x embedding_size
        - labels: tensor size: batch_size x 1
    """
    unique_label = labels.unique()
    inner_class_distances = []
    class_center = []

    for label in unique_label:
        embeddings_label = embeddings[labels==label]
        distance = cal_ed(embeddings_label, embeddings_label)
        distance = torch.sum(distance) / (distance.shape[0] * distance.shape[0])
        if torch.isnan(distance):
            a = 1
        inner_class_distances.append(distance)
        class_center.append(torch.sum(embeddings_label, dim=0)/embeddings_label.shape[0])

    class_center = torch.stack(class_center)
    inter_class_distances = cal_ed(class_center, class_center)

    inner_class_loss = sum(inner_class_distances)
    inter_class_loss = torch.sum(inter_class_distances) / (inter_class_distances.shape[0] * inter_class_distances.shape[0])

    cluster_loss = inner_class_loss / inter_class_loss

    return cluster_loss

def cal_ed(embedding1, embedding2):
    """
    计算embedding1和embedding2中向量之间pair-wise的欧式距离
    Args:
        - embedding1: tensor size m x d
        - embedding1: tensor size n x d
    Returns:
        - distance_matrix: tensor m x n
    """
    m = embedding1.shape[0]
    n = embedding2.shape[0]

    embedding1_l2 = torch.sum(embedding1 * embedding1, dim=1)  # shape: m
    embedding2_l2 = torch.sum(embedding2 * embedding2, dim=1)  # shape: n

    embedding1_l2 = embedding1_l2.unsqueeze(1).expand(m, n)
    embedding2_l2 = embedding2_l2.unsqueeze(0).expand(m, n)

    PCT = torch.mm(embedding1, embedding2.T)

    distance_matrix = embedding1_l2 + embedding2_l2 - 2 * PCT
    distance_matrix = F.relu(distance_matrix)

    return torch.sqrt(distance_matrix)


# ===================================== set input data ================================= #

def set_config(seed=seed):
    # print("运行结果是否保证可复现：{0}".format(Reproducible))
    if reproducible:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_node_feature_type(feature_list, use_node_feature, device):
    for i, is_use in enumerate(use_node_feature):
        if not is_use:
            dim = feature_list[i].shape[0]
            feature_list[i] = torch.FloatTensor(torch.eye(dim)).to(device)
        else:
            feature_list[i] = torch.FloatTensor(feature_list[i]).to(device)
    input_feature_dims = get_input_feature_dim(feature_list)
    return feature_list, input_feature_dims


def get_context_graphs_and_feats(dataset, feature_list, device):
    if dataset == 'DBLP':
        graph_list_file_path = home_path + 'dataset/metapath_based_homogeneous_graph/' + dataset + '/dgl_graph_list.pickle'
        f1 = open(graph_list_file_path, 'rb')
        dgl_graph_list = pickle.load(f1)
        features = {}
        for key in dgl_graph_list.keys():
            middle_node_type = int(key[0])
            features[key] = feature_list[middle_node_type]
        for _ in dgl_graph_list.keys():
            dgl_graph_list[_] = dgl_graph_list[_].to(device)
            features[_] = features[_].to(device)
        return dgl_graph_list, features
    elif dataset == 'IMDb':
        graph_list = []
        _feature_list = []
        for i in range(3):
            graph_list_file_path = home_path + 'dataset/metapath_based_homogeneous_graph/' + dataset + '/dgl_graph_list_' + str(
                i) + '.pickle'
            f1 = open(graph_list_file_path, 'rb')
            dgl_graph_list = pickle.load(f1)
            features = {}
            for key in dgl_graph_list.keys():
                middle_node_type = int(key[0])
                features[key] = feature_list[middle_node_type]
            for _ in dgl_graph_list.keys():
                dgl_graph_list[_] = dgl_graph_list[_].to(device)
                features[_] = features[_].to(device)
            graph_list.append(dgl_graph_list)
            _feature_list.append(features)
        return graph_list, _feature_list
    elif dataset == 'LastFM':
        graph_list = []
        _feature_list = []
        for i in range(2):
            graph_list_file_path = home_path + 'dataset/metapath_based_homogeneous_graph/' + dataset + '/dgl_graph_list_' + str(
                i) + '.pickle'
            f1 = open(graph_list_file_path, 'rb')
            dgl_graph_list = pickle.load(f1)
            features = {}
            for key in dgl_graph_list.keys():
                middle_node_type = int(key[0])
                features[key] = feature_list[middle_node_type]
            for _ in dgl_graph_list.keys():
                dgl_graph_list[_] = dgl_graph_list[_].to(device)
                features[_] = features[_].to(device)
            graph_list.append(dgl_graph_list)
            _feature_list.append(features)
        return graph_list, _feature_list
    else:
        raise Exception("未知数据集")


def get_parameter_number(net):
    a = [p for p in net.parameters()]
    for _ in a:
        print(_.shape)
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total parameters: " + str(total_num))
    print("Trainable parameters: " + str(trainable_num))
    return {'Total': total_num, 'Trainable': trainable_num}


def set_inputs_to_device(input_data, device, dataset, mini_batch=False):
    if dataset == 'DBLP':
        _input_data = []
        # 将input data变成device类型
        for _ in input_data:  # _: tuple
            _input_data.append(([__.to(device) for __ in _[0]], _[1], _[2], _[3].to(device), _[4].to(device)))
    elif dataset == 'IMDb' and not mini_batch:
        target_index = input_data[0][0][1]
        hierarchical_graphs_all = []
        feature_indexes_all = []
        metapath_instances_list_all = []
        # homo_graph_list_all = []
        for node_type, node_type_data in enumerate(input_data):
            hierarchical_graphs = []
            feature_indexes = []
            metapath_instances_list = []
            # homo_graph_list = []
            for metapath_data in node_type_data:
                metapath_hierarchical_graphs = [_.to(device) for _ in metapath_data[0]]
                metapath_feature_indexes = metapath_data[2]
                batch_metapath_instances_list = metapath_data[3]

                hierarchical_graphs.append(metapath_hierarchical_graphs)
                feature_indexes.append(metapath_feature_indexes)
                metapath_instances_list.append(batch_metapath_instances_list)
                # homo_graph_list.append(metapath_data[4].to(device))

            hierarchical_graphs_all.append(hierarchical_graphs)
            metapath_instances_list_all.append(metapath_instances_list)
            feature_indexes_all.append(feature_indexes)
            # homo_graph_list_all.append(homo_graph_list)
        _input_data = [hierarchical_graphs_all, feature_indexes_all, metapath_instances_list_all]
    elif dataset == 'IMDb' and mini_batch:
        _input_data = []
        # 将input data变成device类型
        for _ in input_data:  # _: tuple
            _input_data.append((_[0].to(device), _[1], _[2].to(device)))
    elif dataset == 'LastFM':
        _input_data = []
        for _ in input_data:
            _graphs = []
            graphs = _[0]
            for _g_list in graphs:
                _graphs.append([_1.to(device) for _1 in _g_list])

            _instances = []
            instances = _[1]
            for _instances_list in instances:
                _instances.append([_x.to(device) for _x in _instances_list])

            _indexes = []
            indexes = _[2]

            _hierarchical_data_set = []
            hierarchical_data_set = _[3]
            for mode in hierarchical_data_set:
                mode_data = []
                for metapath_data in mode:
                    hierarchical_graphs = [_.to(device) for _ in metapath_data[0]]
                    _1 = metapath_data[1]
                    hierarchical_indexes = metapath_data[2]
                    for _ in hierarchical_indexes:
                        for k,v in _.items():
                            _[k] = _[k].to(device)
                    _3 = metapath_data[3]
                    mode_data.append([hierarchical_graphs, _1, hierarchical_indexes, _3])
                _hierarchical_data_set.append(mode_data)

            _input_data.append([_graphs, _instances, indexes, _hierarchical_data_set])
    else:
        raise Exception("未知数据集")
    return _input_data


# ============================== evaluate model ================================ #

def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_home_pathd = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_home_pathd, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_home_pathd)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, verbose=False, delta=0, save_path=home_path + 'checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
