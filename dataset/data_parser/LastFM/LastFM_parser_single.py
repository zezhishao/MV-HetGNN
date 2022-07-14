import math
import os
import pickle
from abc import ABC
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
from dgl.data import DGLDataset

home_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) + "/"
save_dir = "dataset/data_parser/LastFM/single/"
sys.path.append(home_dir)

from model.model_utils import set_config
from dataset.dataset_utils import load_LastFM_data, parse_minibatch_LastFM, IndexGenerator
from config import LastFM_batch_size


def get_mapped_nodes(original_nodes_id, mapping):
    _ = original_nodes_id.cpu().numpy()
    return list(map(lambda x: mapping[x], _))


class LastFMDataset(DGLDataset, ABC):
    def __init__(self, name, mode, url=None, raw_dir=None, save_dir=None, force_reload=False, verbose=True,
                 device=torch.device('cpu'), full_batch=False):
        super(LastFMDataset, self).__init__(name=name,
                                          url=url,
                                          raw_dir=raw_dir,
                                          save_dir=save_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)
        assert mode == 'train' or mode == 'validate' or mode == 'test'

        adjlists_ua, edge_metapath_indices_list_ua, _, _, train_val_test_pos_user_artist, train_val_test_neg_user_artist = load_LastFM_data(prefix=home_dir)

        self.adjlists_ua = adjlists_ua  # adjacent matrix in math-path based subgraph
        # metapath instances list for all metapath based subgraphs
        self.edge_type_list = [[[0, 1], [0, 2, 3, 1], [None]],
                               [[1, 0], [2, 3], [1, None, 0]]]
        self.num_edge_type = 4
        self.meta_path_instances_list = [
            [(0, 1, 0), (0, 1, 2, 1, 0), (0, 0)],
            [(1, 0, 1), (1, 2, 1), (1, 0, 0, 1)]
        ]
        self.edge_metapath_indices_list_ua = edge_metapath_indices_list_ua
        self.device = device
        self.mode = mode

        self.num_user = 1892
        self.num_artist = 17632
        self.neighbor_samples = 100


        self.no_mask = [[False] * 3, [False] * 3]
        if mode == 'train':
            self.shuffle = True
            self.use_mask = [[True, True, False],
                             [True, False, True]]
            self.pos_user_artist = train_val_test_pos_user_artist['train_pos_user_artist']  # 
            self.neg_user_artist = train_val_test_neg_user_artist['train_neg_user_artist']  #
            self.idx_generator = IndexGenerator(batch_size=LastFM_batch_size, num_data=len(self.pos_user_artist),
                                                shuffle=self.shuffle)

        elif mode == 'validate':
            self.shuffle = False
            self.use_mask = [[False] * 3, [False] * 3]
            self.pos_user_artist = train_val_test_pos_user_artist['val_pos_user_artist']  # 
            self.neg_user_artist = train_val_test_neg_user_artist['val_neg_user_artist']
            self.idx_generator = IndexGenerator(batch_size=LastFM_batch_size, num_data=len(self.pos_user_artist),
                                                shuffle=self.shuffle)

        elif mode == 'test':
            self.shuffle = False
            self.use_mask = [[False] * 3, [False] * 3]
            self.pos_user_artist = train_val_test_pos_user_artist['test_pos_user_artist']  # 
            self.neg_user_artist = train_val_test_neg_user_artist['test_neg_user_artist']
            self.idx_generator = IndexGenerator(batch_size=LastFM_batch_size, num_data=len(self.pos_user_artist),
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
        # idx_list = self.idx_generator.next()
        # idx_list.sort()
        # het_train_data_list = []
        # magnn_train_data_list = []
        
        if self.mode == 'train':
            idx_batch = self.idx_generator.next()   # idx 不是user或者artist的序号
            idx_batch.sort()
            pos_user_artist_batch = self.pos_user_artist[idx_batch].tolist()    # 

            neg_idx_batch = np.random.choice(len(self.neg_user_artist), len(idx_batch))
            neg_idx_batch.sort()
            neg_user_artist_batch = self.neg_user_artist[neg_idx_batch].tolist()    #
        else:
            idx_batch = self.idx_generator.next()
            pos_user_artist_batch = self.pos_user_artist[idx_batch].tolist()   

            neg_user_artist_batch = self.neg_user_artist[idx_batch].tolist()    # 
        print("# ========== processing mode: {0} batch: {1} ============== #\n idx batch: {2}".format(self.mode, idx, idx_batch))

        pos_g_lists, pos_indices_lists, pos_idx_batch_mapped_lists, pos_hierarchical_graph_all = parse_minibatch_LastFM(
                self.adjlists_ua, self.edge_metapath_indices_list_ua, pos_user_artist_batch, self.device,
                self.neighbor_samples, self.use_mask, self.num_user, edge_type_list=self.edge_type_list)
        neg_g_lists, neg_indices_lists, neg_idx_batch_mapped_lists, neg_hierarchical_graph_all = parse_minibatch_LastFM(
                self.adjlists_ua, self.edge_metapath_indices_list_ua, neg_user_artist_batch, self.device,
                self.neighbor_samples, self.no_mask, self.num_user, edge_type_list=self.edge_type_list)

        return [[pos_g_lists, pos_indices_lists, pos_idx_batch_mapped_lists, pos_hierarchical_graph_all],
                [neg_g_lists, neg_indices_lists, neg_idx_batch_mapped_lists, neg_hierarchical_graph_all]]

    def __len__(self):
        # number of data examples
        return self.idx_generator.num_iterations()

    def save(self):
        raise Exception('不应该运行到这里')
        pass

    def has_cache(self):
        return True

set_config()
train_num = 9283
val_num = 9283
num_all = 92834

for mode in ['train', 'validate', 'test']:
    # mode = 'train'
    # mode = 'validate'
    # mode = 'test'

    print(mode)

    dataset_name = 'LastFM'
    batch_size_dict = {'train': int(math.ceil(train_num / LastFM_batch_size)), 'validate': int(math.ceil(val_num / LastFM_batch_size)),
                    'test': int(math.ceil((num_all - train_num - val_num) / LastFM_batch_size))}

    dataset = LastFMDataset(dataset_name, mode=mode,
                        force_reload=False, verbose=True)


    def _collate_fn(batch):
        return batch


    dataloader = DataLoader(
        dataset, batch_size=batch_size_dict[mode], num_workers=0, shuffle=False, collate_fn=_collate_fn)

    # one epoch
    for _ in dataloader:
        f = open(home_dir + save_dir + 'processed_' + mode + '_data.pickle', 'wb')
        pickle.dump(_, f)
        f.close()
        exit(0)
