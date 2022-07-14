import sys
import numpy as np
import scipy
import pickle
import dgl
import torch
import networkx as nx

preprocessed_data_dir = "data/preprocessed"

# ============================ load pre-processed data ========================== #
def load_LastFM_data(prefix, local=False, only_type_mask=False):
    prefix = prefix + preprocessed_data_dir +'/LastFM'
    if only_type_mask:
        type_mask = np.load(prefix + '/node_types.npy')
        return None, None, None, type_mask, None, None
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/1/1-2-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-2-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_user_artist = np.load(prefix + '/train_val_test_pos_user_artist.npz')
    train_val_test_neg_user_artist = np.load(prefix + '/train_val_test_neg_user_artist.npz')

    return [[adjlist00, adjlist01, adjlist02], [adjlist10, adjlist11, adjlist12]], \
           [[idx00, idx01, idx02], [idx10, idx11, idx12]], \
           adjM, type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist


def load_DBLP_data(prefix, local=False):
    prefix = prefix + preprocessed_data_dir +"/DBLP"
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adj_list00 = [line.strip() for line in in_file]
    adj_list00 = adj_list00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adj_list01 = [line.strip() for line in in_file]
    adj_list01 = adj_list01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adj_list02 = [line.strip() for line in in_file]
    adj_list02 = adj_list02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    adj_m = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adj_list00, adj_list01, adj_list02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3], \
           adj_m, \
           type_mask, \
           labels, \
           train_val_test_idx


def load_ACM_data(prefix, local=False):
    prefix = prefix + preprocessed_data_dir + "/ACM_processed"
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adj_list00 = [line.strip() for line in in_file]
    adj_list00 = adj_list00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adj_list01 = [line.strip() for line in in_file]
    adj_list01 = adj_list01[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()

    features_0 = np.load(prefix + '/features_0.npy')
    features_1 = np.load(prefix + '/features_1.npy')
    features_2 = np.load(prefix + '/features_2.npy')

    adj_m = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adj_list00, adj_list01], \
           [idx00, idx01], \
           [features_0, features_1, features_2], \
           adj_m, \
           type_mask, \
           labels, \
           train_val_test_idx


def load_IMDB_data(prefix, only_load_feature=False):
    prefix = prefix + preprocessed_data_dir +'/IMDb'
    if only_load_feature:
        G00 = None
        G01 = None
        G10 = None
        G11 = None
        G20 = None
        G21 = None
        idx00 = None
        idx01 = None
        idx10 = None
        idx11 = None
        idx20 = None
        idx21 = None
        adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
        labels = np.load(prefix + '/labels.npy')
        train_val_test_idx = None
    else:
        G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
        G01 = nx.read_adjlist(prefix + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
        G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
        G11 = nx.read_adjlist(prefix + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
        G20 = nx.read_adjlist(prefix + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
        G21 = nx.read_adjlist(prefix + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
        idx00 = np.load(prefix + '/0/0-1-0_idx.npy')[:, ::-1]
        idx01 = np.load(prefix + '/0/0-2-0_idx.npy')[:, ::-1]
        idx10 = np.load(prefix + '/1/1-0-1_idx.npy')[:, ::-1]
        idx11 = np.load(prefix + '/1/1-0-2-0-1_idx.npy')[:, ::-1]
        idx20 = np.load(prefix + '/2/2-0-2_idx.npy')[:, ::-1]
        idx21 = np.load(prefix + '/2/2-0-1-0-2_idx.npy')[:, ::-1]
        adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
        labels = np.load(prefix + '/labels.npy')
        train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_0 = features_0.todense()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_1 = features_1.todense()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    features_2 = features_2.todense()

    type_mask = np.load(prefix + '/node_types.npy')

    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2], \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx


# =============================================== data parser helper ============================================== #

class IndexGenerator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0


def get_mapped_nodes(original_nodes_id, mapping):
    _ = original_nodes_id.cpu().numpy()
    return list(map(lambda x: mapping[x], _))


def get_meta_path_specific_sub_graphs(g, meta_path_instances_list, target_idx_list, device, sample=100, **kw):
    """
    根据输入的异构图和meta path以及meta path实例以及target node，构造训练用的异构子图。
    Args:
        g: 原始的异构图。list of list，g[i]代表meta path所对应的子图。
                        例如g[0]代表meta_path_list[0]的子图，g[0][i]代表节点i在meta_path_list[0]下的所有邻居。
                        g[0][4]='4 4 2602 2814'的代表节点4可以通过meta_path_list[0]连接到节点4、2602和2814。
        meta_path_instances_list: list of dict，meta_path_list[i]meta_path_instances_list[i]中查到。
                                例如，meta_path_instances_list[0]["0"]对应于meta_path_list[0]=[0,1,0]=A-P-A中A是"0"号的Author节点对应的所有MetaPath实例。
                                meta_path_instances_list[0]["0"]是一个Nx3的ndarray。
                                N是节点"0"对应的MetaPath实例数量，
                                每一行三维都是一个MetaPath最后一维代表节点"0"，
                                该行的意思是某节点通过该meta path连接到节点"0".
                                Notes：输入数据的各个节点的Index都是指原始图上的Index。
        target_idx_list: 原始图上的目标节点，用来构造训练/验证/测试子图
        device:
        sample: 负采样的数量。为None时不进行负采样
        **kw:

    Returns:
        g_list: 根据target_idx_list和meta_path_list构造出的子图，其index是重新安排的。
        meta_path_instances_list: 某meta path下对应的meta path完整的实例的列表，与g_list一一对应。
                                    要注意的是，这里的index是按照原来的图进行安排的。
                                    要确保meta path instances的顺序和g_list中边的添加顺序一致，既meta_path_instance_list[i]的第k条meta_path_instance对应到g_list[i]中第k条边。
                                    这么设定是为了保证后面在取feature的时候，是正确的。
        sub_graph_train_idx_list: 目前的target_idx_list映射到g_list中的训练index。
        het_train_data_list: 用于后文的训练数据，het_train_data_list[i]对应于meta_path_list[i]。当前状态下仍然是异质图，还未经过Meta Path Encoder。
    """
    # 图采用dgl.Graph的形式来计算
    # 每一个Meta Path会对应一个子图
    # 每一个子图需要构造其（新的）节点和边

    het_train_data_list = []
    # 每个Meta Path 对应一个子图
    for adjacent_list, meta_path_instances in zip(g, meta_path_instances_list):
        # 产生该异质子图的新的nodes和edges
        # 由于新异质子图的Index都是被重新安排的，所以还需要知道新的图和老的图之间的Node Mapping Dict。
        ## 只有知道了这个Mapping Dict才能正确的取出feature。
        # 还需要知道整个MetaPath路径（MetaPathInstanceEncode时要用）

        # 输出一个dgl.Graph、输出该图上的训练节点（新的Index）、输出该图对应的MetaPathInstances。
        # MetaPathInstances的顺序，要确保和dgl.Graph上的edata相对应，即dgl.Graph添加一条边的时候，MetaPathInstances就得添加一个实例。
        dgl_graph, batch_target_index, batch_meta_path_instances = get_meta_path_specific_sub_graph(adjacent_list,
                                                                                                    meta_path_instances,
                                                                                                    target_idx_list,
                                                                                                    device,
                                                                                                    sample)
        het_train_data_list.append((dgl_graph, batch_target_index, batch_meta_path_instances))

    return het_train_data_list


def get_meta_path_specific_sub_graph(adjacent_list, meta_path_instances_dict, target_idx_list, device, sample=100):
    """
    输入某个meta path下的adjacent(list形式)和所有的meta path instances，以及训练节点的index，构造对应的dgl.Graph、dgl.Graph上的训练节点的index和输出该图对应的MetaPathInstances。
    Notes：MetaPathInstances的顺序，要确保和dgl.Graph上的edata相对应，即dgl.Graph添加一条边的时候，MetaPathInstances就得添加一个实例。


    Args:
        adjacent_list: list of str, adjacent_list[0]代表节点0的邻接性，例如adjacent_list[0] = '0 1 5'代表节点0-->节点1，节点0-->节点5。
        meta_path_instances_dict: dict of ndarray, meta_path_instances[i]存储着节点i的所有meta path instance。例如meta_path_instances[0] = [[3, 4595,3],[339,6388,3]]   # TODO，评估一下3-i-3这类路径的意义。因为对于节点3来说，它有一个i类型的邻居就会有一个3-i-3，太多这种收尾是同一个节点的了。这种是不是类似于了随机游走，也包含着一些Structure信息？可以做一下实验
        target_idx_list: 训练节点的index
        device:
        sample: 负采样节点的数量，默认为None。

    Returns:
        dgl_graph
        batch_target_index: 该图中训练节点的index
        batch_meta_path_instances: 该图中edge对应的原始meta path实例。
    """
    edges = []
    nodes = set()
    batch_meta_path_instances = []
    batch_adjacent_list = [adjacent_list[i] for i in target_idx_list]
    batch_meta_path_instances_list = [meta_path_instances_dict[i] for i in target_idx_list]

    sample = None

    for row, meta_path_instances in zip(batch_adjacent_list,
                                        batch_meta_path_instances_list):
        # row: 节点i的邻接性
        # indices: 节点i的所有metapath实例
        row_parsed = list(map(int, row.split(' ')))  # 获取节点i的所有邻居，其自己也算邻居
        nodes.add(row_parsed[0])  # 先添加节点i
        # Sample节点i的邻居，然后写入nodes中
        if len(row_parsed) > 1:
            if sample is None:
                neighbors = row_parsed[1:]  # 获取邻居
                batch_meta_path_instances.append(meta_path_instances)  # 写入Meta Path
                # 这里可以直接这么些，是因为在预处理的时候确保了adjacent list中边的顺序和meta path instances ndarray是一样的。
            else:
                # TODO: here is negative sampling
                # 获取所有的邻居
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                # 标准的negative sampling
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                sample = min(sample, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, sample, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]  # 获取采样后的邻居
                batch_meta_path_instances.append(meta_path_instances[sampled_idx])  # 写入Meta Path
        else:
            neighbors = []
            batch_meta_path_instances.append(meta_path_instances)
        for neighbor in neighbors:
            nodes.add(neighbor)
            edges.append((row_parsed[0], neighbor))  # Notes：edges中的元素是(src, dst)
    batch_meta_path_instances = np.vstack(batch_meta_path_instances)
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}  # 获取新的nodes到老得nodes的index的映射。
    batch_target_index = [np.array([mapping[idx] for idx in target_idx_list])]

    # Notes：nodes和现在的edges中还是老的index
    # 构造dgl图
    ## 获取节点
    num_nodes = len(nodes)
    # edges换成新的index
    ## 获取边
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))  # edges中的node index转变成新的图上的节点
    dgl_graph = dgl.graph(data=([], [])).to(device)
    dgl_graph.add_nodes(num_nodes)
    if len(edges) > 0:
        # 确保dgl.graph的edge的顺序和meta path的顺序是一一对应的
        sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
        dgl_graph.add_edges(*zip(*[(edges[i][1], edges[i][0]) for i in sorted_index]))
        # dgl_graph.add_edges(*zip(*[(edges[i][0], edges[i][1]) for i in sorted_index]))
        batch_meta_path_instances = torch.LongTensor(batch_meta_path_instances[sorted_index]).to(device)
    else:
        batch_meta_path_instances = torch.LongTensor(batch_meta_path_instances).to(device)

    return dgl_graph, batch_target_index, batch_meta_path_instances, mapping


def get_mapped_nodes(original_nodes_id, mapping):
    _ = original_nodes_id.cpu().numpy()
    return list(map(lambda x: mapping[x], _))


def parse_adjlist_LastFM(adjlist, edge_metapath_indices, samples=None, exclude=None, offset=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]
                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed[1:]
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch_LastFM_ori(adjlists_ua, edge_metapath_indices_list_ua, user_artist_batch, device, samples=None,
                               use_masks=None, offset=None):
    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(
            zip(adjlists_ua, edge_metapath_indices_list_ua)):  # mode = node type
        for adjlist, indices, use_mask in zip(adjlists, edge_metapath_indices_list, use_masks[mode]):
            if use_mask:
                edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch],
                    [indices[row[mode]] for row in user_artist_batch], samples, user_artist_batch, offset, mode)
            else:
                edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch],
                    [indices[row[mode]] for row in user_artist_batch], samples, offset=offset, mode=mode)
            g = dgl.graph(([], []), device=device)
            # g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(num_nodes)
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in user_artist_batch]))

    return g_lists, result_indices_lists, idx_batch_mapped_lists


def parse_minibatch_LastFM(adjlists_ua, edge_metapath_indices_list_ua, user_artist_batch, device, samples=None,
                           use_masks=None, offset=None, **kw):
    # edge_metapath_indices_list_ua: U、A所有节点的MetaPath List
    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    hierarchical_graph_all = [[], []]
    mode_dict = {0:'User', 1:"Artist"}
    meta_path_instances_list = [
            [(0, 1, 0), (0, 1, 2, 1, 0), (0, 0)],
            [(1, 0, 1), (1, 2, 1), (1, 0, 0, 1)]
        ]

    for mode, (adjlists, edge_metapath_indices_list) in enumerate(
            zip(adjlists_ua, edge_metapath_indices_list_ua)):  # mode = node type
        for index, (adjlist, indices, use_mask) in enumerate(
                zip(adjlists, edge_metapath_indices_list, use_masks[mode])):
            if use_mask:
                _edges, result_indices, num_nodes, _mapping = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch],
                    [indices[row[mode]] for row in user_artist_batch], samples, user_artist_batch, offset, mode)
            else:
                _edges, result_indices, num_nodes, _mapping = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch],
                    [indices[row[mode]] for row in user_artist_batch], samples, offset=offset, mode=mode)

            hierarchical_graphs = []
            feature_indexes = []
            hierarchical_mapping = {}

            target_node_original = [pair[mode] + mode * 1892 for pair in user_artist_batch]
            target_node_original_unique = sorted(list(set(target_node_original)))
            target_node_original_unique_mapping = {value:key for key, value in enumerate(target_node_original_unique)}

            target_index = target_node_original
            new_idx_arrangement = [target_node_original_unique_mapping[_] for _ in target_node_original]   # 保持target_node_original的顺序

            batch_meta_path_instances = None

            # print("# ================= extracting for {0} ======================== #".format(mode_dict[mode]))
            # print("MetaPath:{0}".format(meta_path_instances_list[mode][index]))
            # print("target_node_original_unique length:{0}".format(len(target_node_original_unique)))

            # print("target_node_original_unique detail:{0}".format(target_node_original_unique))
            # print("extracted metapath instances num:{0}".format(torch.FloatTensor(result_indices)[:, -1].unique().shape[0]))
            # print("extracted metapath instances detail:{0}".format(torch.FloatTensor(result_indices)[:, -1].unique().numpy()))

            edge_type = kw['edge_type_list'][mode][index]
            result_indices_tensor = torch.from_numpy(result_indices)
            for i, relation in enumerate(edge_type):
                edges = result_indices_tensor[:, i:i + 2]
                # edges = torch.tensor(np.unique(edges.cpu().numpy(), axis=0), device=edges.device)
                missed_points = []
                if i == (len(edge_type) - 1):
                    # 最后一层
                    # 找到没有对应的metapath instances的节点
                    # 将其添加一个self loop
                    original_dst = edges[:, 1]
                    dsts = original_dst.unique().numpy()
                    targets = np.array(target_node_original_unique)
                    missed_points = list(np.setdiff1d(targets, dsts))
                missed_points = torch.tensor(missed_points, dtype=torch.int64)  # dst的一种
                edges_points = result_indices_tensor[:, i:i + 2].reshape(-1).unique()
                original_id = torch.cat([missed_points, edges_points]).sort()[0]
                
                new_id = list(range(original_id.shape[0]))
                for ori, new in zip(original_id, new_id):
                    hierarchical_mapping[ori.item()] = new
                
                graph = dgl.graph(data=[])
                graph.add_nodes(len(new_id))

                # 建图
                
                edges_numpy = edges.cpu().numpy()
                src = get_mapped_nodes(edges[:, 0], hierarchical_mapping)
                dst = get_mapped_nodes(edges[:, 1], hierarchical_mapping)
                graph.add_edges(src, dst)

                src_ids_original = edges[:, 0].unique().sort()[0]   # tensor
                dst_ids_original = torch.cat([edges[:, 1], missed_points]).unique().sort()[0]    # 这个顺序必须和target完全一致
                if i == (len(edge_type) - 1):
                    dst_ids_original_list = list(dst_ids_original.cpu().numpy())
                    flag = dst_ids_original_list == target_node_original_unique
                    if not flag:
                        raise Exception("error")
                src_ids_on_graph = get_mapped_nodes(src_ids_original, hierarchical_mapping) # list
                dst_ids_on_graph = get_mapped_nodes(dst_ids_original, hierarchical_mapping) # list


                hierarchical_graphs.append(graph)

                _feature_dict = {'original_id': original_id,
                                 'src_ids_on_graph': torch.tensor(src_ids_on_graph).to(torch.device('cuda:0')),
                                 'dst_ids_on_graph': torch.tensor(dst_ids_on_graph).to(torch.device('cuda:0')),
                                 'src_ids_original': src_ids_original.to(torch.device('cuda:0')),
                                 'dst_ids_original': dst_ids_original.to(torch.device('cuda:0'))}
                feature_indexes.append(_feature_dict)
            hierarchical_graph_all[mode].append(
                (hierarchical_graphs, [target_index, new_idx_arrangement], feature_indexes, batch_meta_path_instances))
            g = dgl.graph(([], []), device=device)
            # g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(num_nodes)
            if len(_edges) > 0:
                sorted_index = sorted(range(len(_edges)), key=lambda i: _edges[i])
                g.add_edges(*list(zip(*[(_edges[i][1], _edges[i][0]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(np.array([_mapping[row[mode]] for row in user_artist_batch]))

    return g_lists, result_indices_lists, idx_batch_mapped_lists, hierarchical_graph_all
