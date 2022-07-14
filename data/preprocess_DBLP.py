prefix = '/home/S22/workspace/MVHetGNN/'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data.preprocess_data_utils import *
import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
# TODO 未整理完成

prefix = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/"

save_prefix = prefix + 'data/preprocessed/DBLP/'

print("start")

"""
预处理DBLP数据集：
目的是分类Author。
产出数据：
1. adjM:
ndarray, 大小为|V| X |V| = 26128 x 26128，其中|V|是预处理后的Node的数量。
adhM_{ij} = Node_i与Node_j之间的联通性(0不连通，1连通）
其中，
authors:4057
papers:14328
terms:7723
confs:20
以scipy的csr_matrix的形式存储，读取的方法为：
scipy.sparse.load_npz("./data/DBLP/adjM.npz")
2. features_0: 
ndarray, 大小为|V_{author}| X |feature size| = 4057 x 334，是预处理后所有Author节点的特征矩阵。
以scipy的csr_matrix的形式存储，读取的方法为：
scipy.sparse.load_npz("./data/DBLP/features_0.npz")
3. features_1: 
ndarray, 大小为|V_{paper}| X |feature size| = 14328 x 4233，是预处理后所有Paeper节点的特征矩阵。
以scipy的csr_matrix的形式存储，读取的方法为：
scipy.sparse.load_npz("./data/DBLP/features_0.npz")
3. features_2: 
ndarray, 大小为|V_{term}| X |feature size| = 7723 x 50，是预处理后所有Term节点的特征矩阵。
4. labels:
ndarray: 大小为 1 X |V_{Author}| = 1x4057，是每一个节点的label。
5. node_types:
ndarray, 大小为 1 X |V| = 1 x 26128， 是一个Type mask。
其中，
node_types[0:4057] = 0,代表Author
node_types[4057: 4057+14328] = 1, 代表Paper
node_types[4057+14328: 4057+14328+7723] = 代表Term
node_types[4057+14328+7723:] = 代表Conference
6.train_val_test_idx:
训练、验证、测试节点的idx。
其中，train、val的节点数量都是400，test数量为3257。
7. 0/0-1-0:
MetaPath:0-1-0(A-P-A)对应的子图的邻接矩阵。共4057个节点。
8. 0/0-1-0_idx:
ndarray, 大小为|E_{0-1-0}| X len(metapath 0-1-0)。
其中|E_{0-1-0}|是0-1-0这个MetaPath对应的子图中的边的数量，
    len(metapath 0-1-0)是metapath 0-1-0的长度（即3）。
这个矩阵记录这每个MetaPath实例详细路径。
数据总长度26128
authors:4057
terms:7723
confs:20
adjM:(26128, 26128)
features author:(4057, 334)
/home/shao/anaconda3/envs/HetG/lib/python3.6/site-packages/sklearn/feature_extraction/text.py:386: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ["'d", "'ll", "'re", "'s", "'ve", 'doe', 'ha', 'le', "n't", 'need', 'sha', 'u', 'wa', 'wo'] not in stop_words.
  'stop_words.' % sorted(inconsistent))
features paper:(14328, 4233)
features term:(7723, 50)
"""

# %%

# 读取所有的文件
print("1. 读取所有文件")
author_label = pd.read_csv(prefix + 'data/raw/DBLP/author_label.txt', sep='\t', header=None,
                           names=['author_id', 'label', 'author_name'], keep_default_na=False, encoding='utf-8')
paper_author = pd.read_csv(prefix + 'data/raw/DBLP/paper_author.txt', sep='\t', header=None,
                           names=['paper_id', 'author_id'],
                           keep_default_na=False, encoding='utf-8')
paper_conf = pd.read_csv(prefix + 'data/raw/DBLP/paper_conf.txt', sep='\t', header=None, names=['paper_id', 'conf_id'],
                         keep_default_na=False, encoding='utf-8')
paper_term = pd.read_csv(prefix + 'data/raw/DBLP/paper_term.txt', sep='\t', header=None, names=['paper_id', 'term_id'],
                         keep_default_na=False, encoding='utf-8')
papers = pd.read_csv(prefix + 'data/raw/DBLP/paper.txt', sep='\t', header=None, names=['paper_id', 'paper_title'],
                     keep_default_na=False, encoding='cp1252')
terms = pd.read_csv(prefix + 'data/raw/DBLP/term.txt', sep='\t', header=None, names=['term_id', 'term'],
                    keep_default_na=False,
                    encoding='utf-8')
confs = pd.read_csv(prefix + 'data/raw/DBLP/conf.txt', sep='\t', header=None, names=['conf_id', 'conf'],
                    keep_default_na=False,
                    encoding='utf-8')

glove_dim = 50
glove_vectors = load_glove_vectors(prefix+"data/", dim=glove_dim)

# %%

## 一共有四类节点：Paper、Conference、Author、Term
## 几类文件：
## 第一大类：某种节点的标签文件：1. 作者-作者的标签 2. 会议-会议的标签 3. 论文-论文的标签
## 第二大类：某种节点的所有成员：1. 作者列表 2. 会议列表 3. 论文列表 4. 关键词（term）列表
## 第三大类：节点之间的关系文件：1. 论文-作者关系 2. 论文-会议关系 3. 论文-关键词（term）关系
## 一共十大类文件，这个项目中主要目的是对作者进行分类。作者主要分为四类：Database, Data Mining, Artificial Intelligence, Information Retrieval
## 显然，在不同节点之间的关系中，论文是“中心节点”，由他连接其他两类节点。这个特点也导致下一步的term清洗还需要且只需要更改paper-termid的关系。

# %%

# filter out all nodes which does not associated with labeled authors
# 过滤三种节点、三种关系（以paper为中心：paper-author、paper-conference、paper-term）
print("2. 过滤三种节点、三种关系（以paper为中心：paper-author、paper-conference、paper-term）")
labeled_authors = author_label['author_id'].to_list()  # 过滤节点：有label的作者
paper_author = paper_author[paper_author['author_id'].isin(labeled_authors)].reset_index(
    drop=True)  # 过滤关系：论文-作者关系中，作者有label的那些
valid_papers = paper_author['paper_id'].unique()
papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)  # 过滤节点：和有label作者相关联的文章
paper_conf = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)  # 过滤关系：和有label作者相关联的论文的相关联的会议
paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(
    drop=True)  # 过滤关系：和有label作者相关联的论文的相关联的Term
valid_terms = paper_term['term_id'].unique()
terms = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)  # 过滤节点：和有label作者相关联的term

# %%
print("3. 用NLP的工具处理Term")

## terms: 8898 rows × 2 columns
## term中的词是有词形变化的，因此需要先将其变化会原来的词性。
## 由于original word和变化之后的单词的一对多关系，会有许多term变重复。
## 因此需要进行重新映射id。
## 此外，由于termid的变化，和term有关的节点的映射关系也有变化一下——有且只有paper节点。
## paperid-termid中的映射关系，要进行更新。
## 更新完之后不要忘记去重。

# %%

# 词形还原和termid重新映射
lemmatizer = WordNetLemmatizer()  # 词形还原
lemma_id_mapping = {}
lemma_list = []
lemma_id_list = []
# 词形还原后，将原term对应到新的term（有一些term会因为词形还原而重复了）
i = 0
for _, row in terms.iterrows():
    i += 1
    lemma = lemmatizer.lemmatize(row['term'])
    lemma_list.append(lemma)
    if lemma not in lemma_id_mapping:
        lemma_id_mapping[lemma] = row['term_id']
    lemma_id_list.append(lemma_id_mapping[lemma])

# %%

# 先更改term节点的id
# 添加词形还原厚的单词（lemma）和其新的id（lemma_list）
terms['lemma'] = lemma_list
terms['lemma_id'] = lemma_id_list
# 将原来的termid和新的lemmaid进行映射。
term_lemma_mapping = {row['term_id']: row['lemma_id'] for _, row in terms.iterrows()}

# 再更改paper-term中的term（id）：更改方式为添加一行lemmaid，使用上文得到的termid-lemmaid映射，输入termid得到。
lemma_id_list = []
for _, row in paper_term.iterrows():
    lemma_id_list.append(term_lemma_mapping[row['term_id']])
paper_term['lemma_id'] = lemma_id_list  # paper_term:原来包含paper和term的关系，现在再由term映射到lemma的关系

# 更改完term节点和paper-term关系后，原来的termid就多余了：
# 清洗paper-term
paper_term = paper_term[['paper_id', 'lemma_id']]  # 只提取lemma id
paper_term.columns = ['paper_id', 'term_id']
paper_term = paper_term.drop_duplicates()
# 清洗term节点
terms = terms[['lemma_id', 'lemma']]
terms.columns = ['term_id', 'term']
terms = terms.drop_duplicates()

# %%


# %%

## 词形处理完后的：terms，7924 * 2
## term 还未处理完毕！term中还有许多是无意义词：停用词。需要将他们过滤干净
## 同理，过滤完之后也要更新termid和paper-term对应关系
## 需要注意的是，这次清洗比上一步更简单，因此这一步只涉及可能的删除操作。即假如是停用词，删除相挂内容即可。
## 而上一步，由于词形变化的一对多的性质，因此更为复杂，需要重新映射id。

# %%

# 依据nltk中的停用词，把term中不是停用词的提取出来
stopwords = sklearn_stopwords.union(set(nltk_stopwords.words('english')))
stopword_id_list = terms[terms['term'].isin(stopwords)]['term_id'].to_list()  # 选择不是停用词的那些
# 把paper-term中，term是停用词的那些拿掉，只剩下不是停用词的。～是dataframe取反。
paper_term = paper_term[~(paper_term['term_id'].isin(stopword_id_list))].reset_index(drop=True)
# 同理，把term中的那些也去掉
terms = terms[~(terms['term'].isin(stopwords))].reset_index(drop=True)

# %%

print("4. 整理初步处理完的几类节点并构造特征")

## 这样的抽取处理完成后，
## paper、conference、author、term以及它们之间的关系
## paper-conference、paper-author、paper-term，
## 就只剩下了和author中有label的那些节点之间的节点和关系。
## 并且term节点及其相关关系已经清理完成

# %%

author_label = author_label.sort_values('author_id').reset_index(drop=True)
papers = papers.sort_values('paper_id').reset_index(drop=True)
terms = terms.sort_values('term_id').reset_index(drop=True)
confs = confs.sort_values('conf_id').reset_index(drop=True)
author_label.to_csv("./preprocessed/DBLP/Author_label.csv", index=False)
papers.to_csv("./preprocessed/DBLP/papers.csv", index=False)
terms.to_csv("./preprocessed/DBLP/terms.csv", index=False)
confs.to_csv("./preprocessed/DBLP/confs.csv", index=False)

# %%

# extract labels of authors
labels = author_label['label'].to_numpy()

# %%

# print(len(author_label))
# print(len(papers))
# print(len(terms))
# print(len(confs))
dim = len(author_label) + len(papers) + len(terms) + len(confs)
type_mask = np.zeros((dim), dtype=int)
type_mask[len(author_label):len(author_label) + len(papers)] = 1
type_mask[len(author_label) + len(papers):len(author_label) + len(papers) + len(terms)] = 2
type_mask[len(author_label) + len(papers) + len(terms):] = 3
print("数据总长度" + str(len(type_mask)))
print("authors:" + str(len(author_label)))
print("terms:" + str(len(terms)))
print("confs:" + str(len(confs)))

# %%

# mapping之后的id是从0开始的，这么做的目的，是为了给邻接矩阵赋值
author_id_mapping = {row['author_id']: i for i, row in author_label.iterrows()}
paper_id_mapping = {row['paper_id']: i + len(author_label) for i, row in papers.iterrows()}
term_id_mapping = {row['term_id']: i + len(author_label) + len(papers) for i, row in terms.iterrows()}
conf_id_mapping = {row['conf_id']: i + len(author_label) + len(papers) + len(terms) for i, row in confs.iterrows()}

adjM = np.zeros((dim, dim), dtype=int)
for _, row in paper_author.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = author_id_mapping[row['author_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1    # note that the graph is always bi-directional
for _, row in paper_term.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = term_id_mapping[row['term_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in paper_conf.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = conf_id_mapping[row['conf_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

print("adjM:" + str(adjM.shape))
# %%


# 构造Author的特征
mat = scipy.io.loadmat(prefix + 'data/raw/DBLP/DBLP4057_GAT_with_idx.mat')
features_author = np.array(list(zip(*sorted(zip(labeled_authors, mat['features']), key=lambda tup: tup[0])))[1])
features_author = scipy.sparse.csr_matrix(features_author)
print("features author:" + str(features_author.shape))


# %%

# 构造Paper的特征
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


vectorizer = CountVectorizer(min_df=2, stop_words=stopwords, tokenizer=LemmaTokenizer())
features_paper = vectorizer.fit_transform(papers['paper_title'].values)
print("features paper:" + str(features_paper.shape))  # 14328 4233

# %%

# 构造term的特征
# use pretrained GloVe vectors as the features of terms
features_term = np.zeros((len(terms), glove_dim))
for i, row in terms.iterrows():
    features_term[i] = glove_vectors.get(row['term'], glove_vectors['the'])

print("features term:" + str(features_term.shape))

# %%
print("5. 计算MetaPath相关的图数据并保存")
# A:0 P:1 T:2 C:3
# # A-P-A A-P-T-P-A A-P-C-P-A
# edge_type_num = A-P P-A P-T T-P P-C C-P
expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)],
    # [(1, 0, 1), (2, 1, 2), (3, 1, 3)]   # reverse meta path
    [(1, 0, 1)],  # reverse meta path
    [(2, 1, 2)],
    [(3, 1, 3)]

]

# %%

import pickle

# neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[0])

# %%

# data = neighbor_pairs  # Some Python object
# f = open('data_records/pickle/neighbor_pairs', 'wb')
# pickle.dump(data, f)
# f.close()
# exit(0)
# G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, type_mask, 0)

# data = G_list  # Some Python object
# f = open('pickle/G_list', 'wb')
# pickle.dump(data, f)
# f.close()
#
# all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)

# %%

# create the directories if they do not exist
# for i in range(4):
#     if i == 0 or i == 1 or i == 3:
#         continue  # 跳过0
#     pass
    # pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)
for i in range(4):  # meta path and reverse meta path
    if i == 0 or i == 1 or i == 3:
        continue  # 跳过0
    # Get MetaPath graph stored in adjlist format
    neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])
    # construct and save metapath-based networks
    G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, type_mask, i)

    # save data
    # networkx graph (metapath specific)
    for G, metapath in zip(G_list, expected_metapaths[i]):
        nx.write_adjlist(G, save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')
    # node indices of edge metapaths
    all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)
    for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):
        np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

# save data
# all nodes adjacency matrix
scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
# all nodes (authors, papers, terms and conferences) features
# currently only have features of authors, papers and terms
scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(0), features_author)
scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(1), features_paper)
np.save(save_prefix + 'features_{}.npy'.format(2), features_term)
# all nodes (authors, papers, terms and conferences) type labels
np.save(save_prefix + 'node_types.npy', type_mask)
# author labels
np.save(save_prefix + 'labels.npy', labels)
# author train/validation/test splits
rand_seed = 1566911444  # 确定种子以确保足够随机但每次分出来的都一样。
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=400, random_state=rand_seed)
train_idx, test_idx = train_test_split(train_idx, test_size=3257, random_state=rand_seed)
train_idx.sort()
val_idx.sort()
test_idx.sort()
np.savez(save_prefix + 'train_val_test_idx.npz',
         val_idx=val_idx,
         train_idx=train_idx,
         test_idx=test_idx)

# %%

# post-processing for mini-batched training
target_idx_list = np.arange(4057)
for metapath in [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)]:
    edge_metapath_idx_array = np.load(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.npy')
    target_metapaths_mapping = {}
    for target_idx in target_idx_list:
        target_metapaths_mapping[target_idx] = edge_metapath_idx_array[edge_metapath_idx_array[:, 0] == target_idx][:,
                                               ::-1]
    out_file = open(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb')
    pickle.dump(target_metapaths_mapping, out_file)
    out_file.close()