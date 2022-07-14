import math
import os
import pickle
from abc import ABC
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
import dgl
from dgl.data import DGLDataset

home_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) + "/"
save_dir = "dataset/data_parser/DBLP/"
sys.path.append(home_dir)

from model.model_utils import set_config
from dataset.dataset_utils import load_DBLP_data, get_meta_path_specific_sub_graph, IndexGenerator
from config import DBLP_batch_size, datasets, dblp_epoch_max


def get_mapped_nodes(original_nodes_id, mapping):
    _ = original_nodes_id.cpu().numpy()
    return list(map(lambda x: mapping[x], _))


class DBLPDataset(DGLDataset, ABC):
    def __init__(self, name, mode, url=None, raw_dir=None, save_dir=None, force_reload=False, verbose=True,
                 device=torch.device('cpu'), full_batch=False):
        super(DBLPDataset, self).__init__(name=name,
                                          url=url,
                                          raw_dir=raw_dir,
                                          save_dir=save_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)
        assert mode == 'train' or mode == 'validate' or mode == 'test'
        assert name in datasets
        adj_lists, edge_metapath_indices_list, _, _, _, labels, train_val_test_idx = load_DBLP_data(
            prefix=home_dir)
        self.graph = adj_lists  # adjacent matrix in math-path based subgraph
        # metapath instances list for all metapath based subgraphs
        self.meta_path_instances_list = edge_metapath_indices_list
        self.device = device
        self.labels = labels
        self.edge_type_list, self.num_edge_type = [
            [0, 1], [0, 2, 3, 1], [0, 4, 5, 1]], 6

        if mode == 'train':
            self.shuffle = True
            self.idx_generator = IndexGenerator(batch_size=DBLP_batch_size,
                                                indices=np.sort(
                                                    train_val_test_idx['train_idx']),
                                                shuffle=self.shuffle)
        elif mode == 'validate':
            self.shuffle = False
            self.idx_generator = IndexGenerator(batch_size=DBLP_batch_size,
                                                indices=np.sort(
                                                    train_val_test_idx['val_idx']),
                                                shuffle=self.shuffle)
        elif mode == 'test':
            self.shuffle = False
            self.idx_generator = IndexGenerator(batch_size=DBLP_batch_size,
                                                indices=np.sort(
                                                    train_val_test_idx['test_idx']),
                                                shuffle=self.shuffle)
        else:
            print(str(mode) + '不再train、validate、test中。')

    def download(self):
        # download raw data to local disk
        raise Exception('不应该运行到这里')
        pass

    def process(self):
        raise Exception('不应该运行到这里')
        pass

    def __getitem__(self, idx):
        print("共{0}轮，正在加载第{1}轮的数据。".format(self.__len__(), idx))
        # get one example by index
        idx_list = self.idx_generator.next()
        idx_list.sort()
        het_train_data_list = []
        magnn_train_data_list = []
        for index, (adjacent_list, meta_path_instances) in enumerate(zip(self.graph, self.meta_path_instances_list)):
            dgl_graph, batch_target_index, batch_meta_path_instances, original_mapping = get_meta_path_specific_sub_graph(
                adjacent_list,
                meta_path_instances,
                idx_list,
                self.device)
            homo_graph = dgl_graph.adj().to_dense()
            index_on_homo_graph = batch_target_index[0]
            batch_adj = homo_graph[index_on_homo_graph][:, index_on_homo_graph]

            edge_type = self.edge_type_list[index]
            mapping = {}
            hierarchical_graphs = []
            feature_indexes = []

            target_node_original = list(
                map(lambda x: {v: k for k, v in original_mapping.items()}[x], batch_target_index[0]))
            for i, relation in enumerate(edge_type):
                original_id = batch_meta_path_instances[:,
                                                        i:i + 2].reshape(-1).unique().sort()[0]
                new_id = list(range(original_id.shape[0]))
                for ori, new in zip(original_id, new_id):
                    mapping[ori.item()] = new
                # 建图
                edges = batch_meta_path_instances[:, i:i + 2]
                edges = torch.tensor(
                    np.unique(edges.cpu().numpy(), axis=0), device=edges.device)

                original_src = edges[:, 0]
                original_dst = edges[:, 1]
                src = get_mapped_nodes(original_src, mapping)
                dst = get_mapped_nodes(original_dst, mapping)
                graph = dgl.graph((src, dst))
                hierarchical_graphs.append(graph)

                _feature_dict = {'original_id': original_id,
                                 'src_ids_on_graph': torch.tensor(sorted(list(set(src)))).to(torch.device('cuda:0')),
                                 'dst_ids_on_graph': torch.tensor(sorted(list(set(dst)))).to(torch.device('cuda:0')),
                                 'src_ids_original': original_src.unique().to(
                                     torch.device('cuda:0')),
                                 'dst_ids_original': original_dst.unique().to(
                                     torch.device('cuda:0'))}
                feature_indexes.append(_feature_dict)
            magnn_train_data_list.append(
                (dgl_graph, batch_target_index, batch_meta_path_instances, mapping))
            # input_data is consist of dgl graph, batch target index batch metapath instances and mapping dict.
            # the batch target index is the target index in dgl graph,
            # while the index in batch metapath instances is the original index in heterogeneous graph.
            # Meanwhile, the mapping function connects the indexes between dgl graph and original heterogeneous graph.
            het_train_data_list.append(
                (hierarchical_graphs, target_node_original, feature_indexes, batch_meta_path_instances, batch_adj))
        _label = self.labels[idx_list]
        # the mapping dict and idx list will only be used in experiments such as checking bad cases,
        # It will not be used in training process.
        # input_data labels indexes
        return [het_train_data_list, _label, idx_list, magnn_train_data_list]

    def __len__(self):
        # number of data examples
        return self.idx_generator.num_iterations()

    def save(self):
        raise Exception('不应该运行到这里')
        pass

    def has_cache(self):
        return True

for mode in ['train', 'validate', 'test']:
    # mode = 'train'

    set_config()
    train_num = 400
    val_num = 400
    num_all = 4057

    dataset_name = 'DBLP'
    batch_size_dict = {'train': int(math.ceil(train_num / DBLP_batch_size)), 'validate': int(math.ceil(val_num / DBLP_batch_size)),
                    'test': int(math.ceil((num_all - train_num - val_num) / DBLP_batch_size))}

    dataset = DBLPDataset(dataset_name, mode=mode,
                        force_reload=False, verbose=True)

    def _collate_fn(batch):
        return batch

    dataloader = DataLoader(
        dataset, batch_size=batch_size_dict[mode], num_workers=0, shuffle=False, collate_fn=_collate_fn)

    f = open(home_dir + save_dir + 'processed_' + mode + '_data.pickle', 'wb')
    for _ in dataloader:
        pickle.dump(_, f)
        f.close()
