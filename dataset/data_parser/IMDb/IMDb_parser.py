import os
import sys
import torch
import pickle
from abc import ABC
import dgl
import numpy as np
from dgl.data import DGLDataset
from torch.utils.data import DataLoader

home_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) + "/"
save_dir = "dataset/data_parser/IMDb/"
sys.path.append(home_dir)

from model.model_utils import set_config
from dataset.dataset_utils import load_IMDB_data
from config import home_path, datasets


def get_mapped_nodes(original_nodes_id, mapping):
    _ = original_nodes_id.cpu().numpy()
    return list(map(lambda x: mapping[x], _))


class IMDbDataset(DGLDataset, ABC):
    def __init__(self, name, mode, url=None, raw_dir=None, save_dir=None, force_reload=False, verbose=True,
                 device=torch.device('cpu'), full_batch=True):
        super(IMDbDataset, self).__init__(name=name,
                                          url=url,
                                          raw_dir=raw_dir,
                                          save_dir=save_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)
        assert mode == 'train' or mode == 'validate' or mode == 'test'
        assert name in datasets

        nx_G_lists, edge_metapath_indices_lists, _, _, _, labels, train_val_test_idx = load_IMDB_data(
            prefix=home_path)
        self.g_lists = []
        for nx_G_list in nx_G_lists:
            self.g_lists.append([])
            for nx_G in nx_G_list:
                g = dgl.graph(data=([], [])).to(device)
                g.add_nodes(nx_G.number_of_nodes())
                g.add_edges(
                    *list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
                self.g_lists[-1].append(g)

        self.meta_path_instances_list = edge_metapath_indices_lists
        self.device = device
        self.labels = labels
        self.full_batch = full_batch

        self.edge_type_list_full, self.num_edge_type = [[[0, 1], [2, 3]],
                                                        [[1, 0], [1, 2, 3, 0]],
                                                        [[3, 2], [3, 0, 1, 2]]], 6

        if mode == 'train':
            self.target_idx = train_val_test_idx['train_idx']

        elif mode == 'validate':
            self.target_idx = train_val_test_idx['val_idx']

        elif mode == 'test':
            self.target_idx = train_val_test_idx['test_idx']

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
        # TODO add hierarchical graph
        het_train_data_list_all = []
        for _, (metapath_all_graph, metapath_instances_all, edge_type_list) in enumerate(zip(self.g_lists, self.meta_path_instances_list, self.edge_type_list_full)):
            # metapath based subgraphs for node type _.
            # for example, _ = 0, metapath_all_graph contains metapath based subgraphs corrseponding to MDM MAM。
            # M:0 D:1 A:2
            het_train_data_list = []
            for index, (graph, metapath_instances) in enumerate(zip(metapath_all_graph, metapath_instances_all)):
                # homo_graph = graph.adj().to_dense()
                metapath_instances = torch.from_numpy(metapath_instances.copy())

                edge_type = edge_type_list[index]
                mapping = {}
                hierarchical_graphs = []
                feature_indexes = []

                target_node_original = self.target_idx.tolist()

                for i, relation in enumerate(edge_type):
                    original_id = metapath_instances[:,
                                                     i:i + 2].reshape(-1).unique().sort()[0]
                    new_id = list(range(original_id.shape[0]))
                    for ori, new in zip(original_id, new_id):
                        mapping[ori.item()] = new
                    # 建图
                    edges = metapath_instances[:, i:i + 2]
                    edges = torch.tensor(
                        np.unique(edges.cpu().numpy(), axis=0), device=edges.device)

                    original_src = edges[:, 0]
                    original_dst = edges[:, 1]
                    src = get_mapped_nodes(original_src, mapping)
                    dst = get_mapped_nodes(original_dst, mapping)
                    graph = dgl.graph((src, dst))
                    hierarchical_graphs.append(graph)

                    _feature_dict = {'original_id': original_id,
                                     'src_ids_on_graph': torch.LongTensor(sorted(list(set(src)))).to(
                                         torch.device('cuda:0')),
                                     'dst_ids_on_graph': torch.LongTensor(sorted(list(set(dst)))).to(
                                         torch.device('cuda:0')),
                                     'src_ids_original': original_src.unique().to(
                                         torch.device('cuda:0')),
                                     'dst_ids_original': original_dst.unique().to(
                                         torch.device('cuda:0'))}
                    feature_indexes.append(_feature_dict)
                het_train_data_list.append(
                    (hierarchical_graphs, target_node_original, feature_indexes, metapath_instances))
            het_train_data_list_all.append(het_train_data_list)
        # 想要利用其他类别节点的MetaPath，必须是两层，要不然的话无法加入到loss里面去，相当于无效。
        return [het_train_data_list_all,
                self.labels[self.target_idx],
                self.target_idx]

    def __len__(self):
        # number of data examples
        return 1

    def save(self):
        raise Exception('不应该运行到这里')
        pass

    def has_cache(self):
        return True

for mode in ['train', 'validate', 'test']:
    # mode = 'train'

    set_config()
    print(mode)
    dataset_name = 'IMDb'

    dataset = IMDbDataset(dataset_name, mode=mode, force_reload=False, verbose=True)


    # create collate_fn
    def _collate_fn(batch):
        return batch


    dataloader = DataLoader(dataset, batch_size=1, num_workers=0,
                            shuffle=False, collate_fn=_collate_fn)
    f = open(home_dir + save_dir + 'processed_' + mode + '_data.pickle', 'wb')

    for _ in dataloader:
        pickle.dump(_, f)
