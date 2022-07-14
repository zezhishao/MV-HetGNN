# -*- coding: utf-8 -*-
# @Time        : 2020/9/15 下午3:17
# @Author      : Zezhi Shao
# @File        : model_data_loader.py
# @Software    : PyCharm
# @Description : 为训练、验证、测试环节建立data loader和dataset
from abc import ABC
import pickle
from dgl.data import DGLDataset
from config import home_path, datasets


class ModelDataset(DGLDataset, ABC):
    def __init__(self, name, mode, epoch=None, url=None, raw_dir=None, save_dir=None, force_reload=False, verbose=False,
                 full_batch=True, method=None):
        super(ModelDataset, self).__init__(name=name,
                                           url=url,
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)
        self._name = name
        assert mode == 'train' or mode == 'validate' or mode == 'test'
        assert name in datasets
        if name == 'IMDb' and full_batch == False:
            # IMDb数据集也是用minibatch训练
            file_path = home_path + 'dataset/data_parser/' + name + '/processed_' + str(mode) + '_mini_batch_data.pickle'
        elif name == 'IMDb' and full_batch == True:
            file_path = home_path + 'dataset/data_parser/' + name + '/processed_' + str(mode) + '_data.pickle'
        elif name == 'DBLP':
            if epoch is not None:
                # train mode:
                # DBLP只能使用minibatch
                file_path = home_path + 'dataset/data_parser/' + name + '/processed_' + str(mode) + '_data_epoch_' + str(epoch) + '.pickle'
            else:
                file_path = home_path + 'dataset/data_parser/' + name + '/processed_' + str(mode) + '_data.pickle'

        elif name == 'LastFM':
            if method == 'ours':
                mode = 'dump_' + mode
            file_path = home_path + 'dataset/data_parser/' + name + '/processed_' + str(mode) + '_data.pickle'
        else:
            raise Exception("为识别的数据集")
        self.data = pickle.load(open(file_path, 'rb'))

    def process(self):
        raise Exception('不应该运行到这里')
        pass

    def __getitem__(self, idx):
        if self._name == 'LastFM':
            return self.data[idx]
        if self._name == 'IMDb':
            return self.data[idx][0], self.data[idx][1], self.data[idx][2]  # graphs labels indexes
        return self.data[idx][0], self.data[idx][1]

    def __len__(self):
        # number of data examples
        return len(self.data)

    def has_cache(self):
        return True


if __name__ == '__main__':
    pass