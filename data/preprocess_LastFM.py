from preprocess_data_utils import multidim_intersect
import pathlib
import os
import pickle
import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
from tqdm import tqdm
import sys

train_p = 0.2
val_p = 0.1
test_p = 0.7

home_dir = "/home/S22/workspace/MVHetGNN/data"
sys.path.append(home_dir)

np.random.seed(453289)
num_all_links = 92834

num_user = 1892
num_artist = 17632
num_tag = 11945 # not the final number

# ============================= 通用参数 ============================= #
prefix = '/home/S22/workspace/MVHetGNN/'
raw_dir = prefix + 'data/raw/LastFM/'
save_prefix = prefix + 'data/preprocessed/LastFM/'

print(save_prefix)
if not os.path.exists(save_prefix):
    os.makedirs(save_prefix)

# ============================= 读取所有的关系 ============================= #
user_artist = pd.read_csv(raw_dir + 'user_artist.dat', encoding='utf-8',
                          delimiter='\t', names=['userID', 'artistID', 'weight'])
user_friend = pd.read_csv(raw_dir + 'user_user(original).dat',
                          encoding='utf-8', delimiter='\t', names=['userID', 'friendID'])
artist_tag = pd.read_csv(raw_dir + 'artist_tag.dat', encoding='utf-8',
                         delimiter='\t', names=['artistID', 'tagID'])

# ======================= 读取数据集的index并重采样 ========================= #
# 归并所有的index
train_val_test_idx = np.load(raw_dir + 'train_val_test_idx.npz')
train_idx = train_val_test_idx['train_idx']
val_idx = train_val_test_idx['val_idx']
test_idx = train_val_test_idx['test_idx']
num_of_idx = len(np.concatenate((train_idx, val_idx, test_idx)))

print("train length: {0}".format(len(train_idx)))
print("val length: {0}".format(len(val_idx)))
print("test length: {0}".format(len(test_idx)))

# ===== test unit =====#
# 0. u-a长度正常
# 1. train、val、test不重复
# 2. train、val、test对应的u-a关系不重复
# 3. train、val、test的比例正常
assert len(user_artist) == num_all_links
diff12 = np.intersect1d(train_idx, val_idx)    # 获取剩余样本
diff13 = np.intersect1d(train_idx, test_idx)    # 获取剩余样本
diff23 = np.intersect1d(val_idx, test_idx)    # 获取剩余样本
assert len(diff12) == 0 and len(diff13) == 0 and len(diff23) == 0
tmp_user_artist = user_artist[['userID', 'artistID']].to_numpy()
tmp_user_artist = tmp_user_artist - 1
tmp_train_pairs = tmp_user_artist[train_idx]
tmp_val_pairs = tmp_user_artist[val_idx]
tmp_test_pairs = tmp_user_artist[test_idx]
diff12 = multidim_intersect(tmp_train_pairs, tmp_val_pairs)
diff13 = multidim_intersect(tmp_train_pairs, tmp_test_pairs)
diff23 = multidim_intersect(tmp_val_pairs, tmp_test_pairs)
assert len(diff12) == 0 and len(diff13) == 0 and len(diff23) == 0
assert len(train_idx) == round(num_all_links * train_p) and len(val_idx) == round(
    num_all_links * val_p) and len(test_idx) == (num_of_idx - len(train_idx)- len(val_idx))
del tmp_user_artist, tmp_train_pairs, tmp_val_pairs, tmp_test_pairs, diff12, diff13, diff23

# ======================= 处理所有的关系并产生邻接矩阵和Type Mask ========================= #
# 由于是link prediction的任务，所有u-a关系只能使用train index内的那些u-a来产生邻接矩阵
user_artist = user_artist.loc[train_idx].reset_index(drop=True)
# ============= 产生未经过滤的邻接矩阵和type mask ============= #
# build type mask
# 0 for user, 1 for artist, 2 for tag
dim = num_user + num_artist + num_tag
type_mask = np.zeros((dim), dtype=int)
type_mask[num_user:num_user+num_artist] = 1
type_mask[num_user+num_artist:] = 2

# build the adjacency matrix
adjM = np.zeros((dim, dim), dtype=int)
# user - artist(always bi-directional)
for _, row in user_artist.iterrows():
    uid = row['userID'] - 1
    aid = num_user + row['artistID'] - 1
    adjM[uid, aid] = max(1, row['weight'])
    adjM[aid, uid] = max(1, row['weight'])
# user - user(not always bi-directional, wait for post process)
for _, row in user_friend.iterrows():
    uid = row['userID'] - 1
    fid = row['friendID'] - 1
    adjM[uid, fid] = 1

# artist tag(always bi-directional)
for _, row in artist_tag.iterrows():
    aid = num_user + row['artistID'] - 1
    tid = num_user + num_artist + row['tagID'] - 1
    adjM[aid, tid] += 1
    adjM[tid, aid] += 1
# ============= 过滤掉不符合要求的artist-tags ============= #
# post process a-t, and make it bi-directional
print("filter out artist-tag links with counts less than 2")
adjM[num_user:num_user+num_artist, num_user+num_artist:] = adjM[num_user:num_user+num_artist,
                                                                num_user+num_artist:] * (adjM[num_user:num_user+num_artist, num_user+num_artist:] > 1)
adjM[num_user+num_artist:, num_user:num_user +
     num_artist] = np.transpose(adjM[num_user:num_user+num_artist, num_user+num_artist:])

# delete tags that do not connect to any artist
valid_tag_idx = adjM[num_user:num_user+num_artist,
                     num_user+num_artist:].sum(axis=0).nonzero()[0]
num_tag = len(valid_tag_idx)
dim = num_user + num_artist + num_tag
type_mask = np.zeros((dim), dtype=int)
type_mask[num_user:num_user+num_artist] = 1
type_mask[num_user+num_artist:] = 2

adjM_reduced = np.zeros((dim, dim), dtype=int)
adjM_reduced[:num_user+num_artist, :num_user +
             num_artist] = adjM[:num_user+num_artist, :num_user+num_artist]
adjM_reduced[num_user:num_user+num_artist, num_user +
             num_artist:] = adjM[num_user:num_user+num_artist, num_user+num_artist:][:, valid_tag_idx]
adjM_reduced[num_user+num_artist:, num_user:num_user+num_artist] = np.transpose(
    adjM_reduced[num_user:num_user+num_artist, num_user+num_artist:])

adjM = adjM_reduced

# ======================= 产生u-a/u-u/a-t对应的所有连接并按照node id 组建字典 ======================= #
user_artist_list = {i: adjM[i, num_user:num_user +
                            num_artist].nonzero()[0] for i in range(num_user)}
artist_user_list = {i: adjM[num_user + i, :num_user].nonzero()[0]
                    for i in range(num_artist)}
user_user_list = {i: adjM[i, :num_user].nonzero()[0] for i in range(num_user)}
artist_tag_list = {i: adjM[num_user + i, num_user +
                           num_artist:].nonzero()[0] for i in range(num_artist)}
tag_artist_list = {i: adjM[num_user + num_artist + i,
                           num_user:num_user+num_artist].nonzero()[0] for i in range(num_tag)}


# ======================= 抽取所有的metapath相关的内容 ========================== #
print("# ===================== all metapath instances =========================== #")
print("0-1-0")
# 0-1-0
u_a_u = []
for a, u_list in tqdm(artist_user_list.items()):
    u_a_u.extend([(u1, a, u2) for u1 in u_list for u2 in u_list])
u_a_u = np.array(u_a_u)
u_a_u[:, 1] += num_user
sorted_index = sorted(list(range(len(u_a_u))),
                      key=lambda i: u_a_u[i, [0, 2, 1]].tolist())
u_a_u = u_a_u[sorted_index]
print("1-2-1")

# 1-2-1
a_t_a = []
for t, a_list in tqdm(tag_artist_list.items()):
    a_t_a.extend([(a1, t, a2) for a1 in a_list for a2 in a_list])
a_t_a = np.array(a_t_a)
a_t_a += num_user
a_t_a[:, 1] += num_artist
sorted_index = sorted(list(range(len(a_t_a))),
                      key=lambda i: a_t_a[i, [0, 2, 1]].tolist())
a_t_a = a_t_a[sorted_index]
print("0-1-2-1-0")

# 0-1-2-1-0
u_a_t_a_u = []
for a1, t, a2 in tqdm(a_t_a):
    if len(artist_user_list[a1 - num_user]) == 0 or len(artist_user_list[a2 - num_user]) == 0:
        continue
    candidate_u1_list = np.random.choice(len(artist_user_list[a1 - num_user]), int(
        0.2 * len(artist_user_list[a1 - num_user])), replace=False)
    candidate_u1_list = artist_user_list[a1 - num_user][candidate_u1_list]
    candidate_u2_list = np.random.choice(len(artist_user_list[a2 - num_user]), int(
        0.2 * len(artist_user_list[a2 - num_user])), replace=False)
    candidate_u2_list = artist_user_list[a2 - num_user][candidate_u2_list]
    u_a_t_a_u.extend([(u1, a1, t, a2, u2)
                      for u1 in candidate_u1_list for u2 in candidate_u2_list])
u_a_t_a_u = np.array(u_a_t_a_u)
sorted_index = sorted(list(range(len(u_a_t_a_u))),
                      key=lambda i: u_a_t_a_u[i, [0, 4, 1, 2, 3]].tolist())
u_a_t_a_u = u_a_t_a_u[sorted_index]
print("0-0")
# 0-0
u_u = user_friend.to_numpy(dtype=np.int32) - 1
sorted_index = sorted(list(range(len(u_u))), key=lambda i: u_u[i].tolist())
u_u = u_u[sorted_index]
print("1-0-1")

# 1-0-1
a_u_a = []
for u, a_list in tqdm(user_artist_list.items()):
    a_u_a.extend([(a1, u, a2) for a1 in a_list for a2 in a_list])
a_u_a = np.array(a_u_a)
a_u_a[:, [0, 2]] += num_user
sorted_index = sorted(list(range(len(a_u_a))),
                      key=lambda i: a_u_a[i, [0, 2, 1]].tolist())
a_u_a = a_u_a[sorted_index]
print("1-0-0-1")

# 1-0-0-1
a_u_u_a = []
for u1, u2 in tqdm(u_u):
    a_u_u_a.extend([(a1, u1, u2, a2) for a1 in user_artist_list[u1]
                    for a2 in user_artist_list[u2]])
a_u_u_a = np.array(a_u_u_a)
a_u_u_a[:, [0, 3]] += num_user
sorted_index = sorted(list(range(len(a_u_u_a))),
                      key=lambda i: a_u_u_a[i, [0, 3, 1, 2]].tolist())
a_u_u_a = a_u_u_a[sorted_index]
expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 0, 0, 1)]
]
# create the directories if they do not exist
for i in range(len(expected_metapaths)):
    pathlib.Path(save_prefix + '{}'.format(i)
                 ).mkdir(parents=True, exist_ok=True)

metapath_indices_mapping = {(0, 1, 0): u_a_u,
                            (0, 1, 2, 1, 0): u_a_t_a_u,
                            (0, 0): u_u,
                            (1, 0, 1): a_u_a,
                            (1, 2, 1): a_t_a,
                            (1, 0, 0, 1): a_u_u_a}

print("write all things")

# ======================== 存储所有的metapath based neighbors（adjm的形式）以及所有的metapath instances ======================== #
# write all things
target_idx_lists = [np.arange(num_user), np.arange(num_artist)]
offset_list = [0, num_user]
for i, metapaths in enumerate(expected_metapaths):
    for metapath in metapaths:
        print(str(metapath))
        edge_metapath_idx_array = metapath_indices_mapping[metapath]

        # extract metapath instances
        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        # extract metapath based neighbors
        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                    right += 1
                neighbors = edge_metapath_idx_array[left:right, -
                                                    1] - offset_list[i]
                neighbors = list(map(str, neighbors))
                if len(neighbors) > 0:
                    out_file.write('{} '.format(target_idx) +
                                   ' '.join(neighbors) + '\n')
                else:
                    out_file.write('{}\n'.format(target_idx))
                left = right

# ================== 写入adjm 和 type mask ===================== #
scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
np.save(save_prefix + 'node_types.npy', type_mask)
print("output user_artist.npy")

# ================= 再重新读取u-a，前面为了生成adjm以及对应的metapath数据，删除了val和test的ua =============== # 
# output user_artist.npy
user_artist = pd.read_csv(raw_dir + 'user_artist.dat', encoding='utf-8',
                          delimiter='\t', names=['userID', 'artistID', 'weight'])
user_artist = user_artist[['userID', 'artistID']].to_numpy()
user_artist = user_artist - 1
np.save(save_prefix + 'user_artist.npy', user_artist)

print("# ========== output positive and negative samples for training, validation and testing ============== #")
# output positive and negative samples for training, validation and testing
user_artist = np.load(save_prefix + 'user_artist.npy')
print("产生所有的负样本：...")
neg_candidates = []
counter = 0
for i in tqdm(range(num_user)):
    for j in range(num_artist):
        if counter < len(user_artist):
            if i == user_artist[counter, 0] and j == user_artist[counter, 1]:
                counter += 1
            else:
                neg_candidates.append([i, j])
        else:
            neg_candidates.append([i, j])
neg_candidates = np.array(neg_candidates)
print("产生验证、测试负样本：...")
idx = np.random.choice(len(neg_candidates), len(
    val_idx) + len(test_idx), replace=False)
val_neg_candidates = neg_candidates[sorted(idx[:len(val_idx)])]
test_neg_candidates = neg_candidates[sorted(idx[len(val_idx):])]

train_user_artist = user_artist[train_idx]
train_neg_candidates = []
counter = 0
print("产生训练负样本：...")
for i in range(num_user):
    for j in range(num_artist):
        if counter < len(train_user_artist):
            if i == train_user_artist[counter, 0] and j == train_user_artist[counter, 1]:
                counter += 1
            else:
                train_neg_candidates.append([i, j])
        else:
            train_neg_candidates.append([i, j])
train_neg_candidates = np.array(train_neg_candidates)

np.savez(save_prefix + 'train_val_test_neg_user_artist.npz',
         train_neg_user_artist=train_neg_candidates,
         val_neg_user_artist=val_neg_candidates,
         test_neg_user_artist=test_neg_candidates)
np.savez(save_prefix + 'train_val_test_pos_user_artist.npz',
         train_pos_user_artist=user_artist[train_idx],
         val_pos_user_artist=user_artist[val_idx],
         test_pos_user_artist=user_artist[test_idx])
