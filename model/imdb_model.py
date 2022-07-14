from abc import ABC
import torch
import torch.nn as nn
import dgl
import numpy as np
from model.base_models.NodeTransform import NodeTransform
from model.base_models.EgoGraphEncoding import EgoGraphEncoding, CompGCNEncoder
from model.base_models.AutoMvFusion import AutoMvFusion
import torch.nn as nn

class BaseLayer(nn.Module, ABC):
    """
    HetG for represent a kind of node
    """
    def __init__(self, args, metapath_list, edge_type_list, num_edge, relational_encoders, layer_num=-1):
        super(BaseLayer, self).__init__()
        # args
        if layer_num == 0:
            self.hidden_dim, self.intra_dim, self.inter_dim = args.hidden_dim, args.intra_dim, args.inter_dim
        else:
            self.hidden_dim, self.intra_dim, self.inter_dim = args.inter_dim, args.inter_dim, args.inter_dim

        # ============= metapath ================= #
        self.metapath_list = metapath_list
        self.edge_type_list, self.num_edge_type = edge_type_list, num_edge
        
        # ============= public paramerters for ego graph encoding
        self.relational_encoders = relational_encoders

        # ================ key component ===================== #
        # =============== 2. metapath encoder ================== #
        self.ego_graph_encoder = nn.ModuleList([EgoGraphEncoding(metapath=self.metapath_list[_]) for _ in range(len(metapath_list))])
        # ================= 3. semantic fusion ================== #
        self.auto_mv_fusion = AutoMvFusion(self.hidden_dim, self.intra_dim, self.inter_dim, self.metapath_list, mse=args.mse)


    def forward(self, input_data, transformed_features):
        metapath_outs = {'-'.join(map(str, metapath)): None for metapath in self.metapath_list}

        for index, metapath_layer_data in enumerate(zip(input_data[0], input_data[1], input_data[2])):
            # metapath_layer_data[0] = hierarchical relational graphs
            # metapath_layer_data[1] = target index
            # metapath_layer_data[2] = feature index. node index of each hierarchical relational graph
            # metapath_layer_data[3] = batch metapath instances lists corresponding to hierarchical relational graphs
            hierarchical_relational_graphs = metapath_layer_data[0]
            edge_types = self.edge_type_list[index]
            transformed_features = transformed_features
            feature_index = metapath_layer_data[1]

            metapath_outs['-'.join(map(str, self.metapath_list[index]))] = \
                self.ego_graph_encoder[index](hierarchical_relational_graphs, edge_types,
                                                       self.relational_encoders,
                                                       transformed_features, feature_index)
        
        # ================== metapath fusion ====================#
        fused_features, re_and_ortho_loss = self.auto_mv_fusion(metapath_outs)

        return fused_features, re_and_ortho_loss


class IMDBModelLayer(nn.Module, ABC):
    def __init__(self, type_mask, args, edge_type_list_full, num_edge_type, metapath_list_full, layer_num=-1):
        super(IMDBModelLayer, self).__init__()
        """
        comp vec and message passing matrix are shared ** in this layer **, but auto mv fusion is set for every node type. 
        """
        # in this layer, we change the dimension
        if layer_num == 0:
            self.hidden_dim = args.hidden_dim
        else:
            self.hidden_dim = args.inter_dim
        self.inter_dim  = args.inter_dim
        self.type_mask  = type_mask
        # ================ metapaths ===================== #
        self.metapath_list_full = metapath_list_full
        self.edge_type_list_full, self.num_edge_type = edge_type_list_full, num_edge_type

        # =============================== gcn encoder and compositional vec =================================== #
        self.comp_vec               = nn.ParameterList([nn.Parameter(torch.empty(1, self.hidden_dim)) for _ in range(self.num_edge_type)])
        self.gcn_encoders           = nn.ModuleList([dgl.nn.GraphConv(self.hidden_dim, self.hidden_dim, norm='right') for _ in range(2)])
        self.relational_encoders    = torch.nn.ModuleList([CompGCNEncoder(self.comp_vec[_], self.get_gcn_encoder(_)) for _ in range(self.num_edge_type)])

        self.IMDB_layers = nn.ModuleList([BaseLayer(args, metapath_list, edge_type_list, self.num_edge_type, relational_encoders=self.relational_encoders, layer_num=layer_num)
             for metapath_list, edge_type_list in zip(metapath_list_full, edge_type_list_full)])

    def get_gcn_encoder(self, e_type):
        if e_type in [1, 2]:
            return self.gcn_encoders[0]
        elif e_type in [0, 3]:
            return self.gcn_encoders[1]

    def forward(self, input_data, transformed_feature_list):
        # very relational encoder contains a comp_vec and a gcn_encoder.
        h = torch.zeros(self.type_mask.shape[0], self.inter_dim, device=torch.device('cuda:0'))
        # # first layer will receive features of context node mapping
        transformed_feature = transformed_feature_list
        re_and_ortho_loss_list = []
        for i, (hierarchical_graphs, feature_index, metapath_instances_list, layer) in enumerate(zip(input_data[0], input_data[1], input_data[2], self.IMDB_layers)):
            out, re_and_ortho_loss = layer((hierarchical_graphs, feature_index, metapath_instances_list), transformed_feature)
            node_indices = np.where(self.type_mask == i)[0]
            h[node_indices] = out
            re_and_ortho_loss_list.append(re_and_ortho_loss)
        re_and_ortho_loss = sum(re_and_ortho_loss_list)
        return h, re_and_ortho_loss

class IMDBModel(nn.Module, ABC):
    def __init__(self, type_mask, input_feature_dims, args, num_layers=2):
        super(IMDBModel, self).__init__()
        # agrs
        self.gain,      self.use_activation,    self.dropout_rate   = args.gain,        True,               args.dropout_rate
        self.inter_dim, self.hidden_dim,        self.output_dim     = args.inter_dim,   args.hidden_dim,    args.output_dim
        
        # ===================== metapath ======================= #
        self.metapath_list_full = (  # M:0 D:1 A:2
            [(0, 1, 0), (0, 2, 0)],  # MDM MAM
            [(1, 0, 1), (1, 0, 2, 0, 1)],  # DMD DMAMD
            [(2, 0, 2), (2, 0, 1, 0, 2)]  # AMA AMDMA
        )
        self.edge_type_list_full, self.num_edge_type = [[[0, 1], [2, 3]],
                                                        [[1, 0], [1, 2, 3, 0]],
                                                        [[3, 2], [3, 0, 1, 2]]], 4
        # =============================== key component ================================= #
        # 1. node transformation #
        self.node_transformation = NodeTransform(input_feature_dims, self.hidden_dim, type_mask, 'IMDb', self.dropout_rate, self.use_activation)
        # 2. layers. 
        self.IMDBModelLayers = nn.ModuleList([IMDBModelLayer(type_mask, args, self.edge_type_list_full, self.num_edge_type, self.metapath_list_full, _) for _ in range(num_layers)])

        # ================= output layer =============== #
        self.output_layer = nn.Linear(self.inter_dim, self.output_dim, bias=False)
        self.init_parameters()

    def init_parameters(self):
        # comp vec
        for layer in self.IMDBModelLayers:
            for _ in layer.comp_vec:
                nn.init.xavier_normal_(_.data, gain=self.gain)
        
        # node transformation
        pass
        
        # gcn conv
        pass
        
        # fusion
        for layer in self.IMDBModelLayers:
            for base_layer in layer.IMDB_layers:
                nn.init.orthogonal_(base_layer.auto_mv_fusion.V, gain=self.gain)
                nn.init.orthogonal_(base_layer.auto_mv_fusion.V_reverse, gain=self.gain)

                for fc in base_layer.auto_mv_fusion.fc_list:
                    nn.init.orthogonal_(fc, gain=self.gain)
                for fc in base_layer.auto_mv_fusion.fc_list_reverse:
                    nn.init.orthogonal_(fc, gain=self.gain)

        # output layer
        nn.init.xavier_normal_(self.output_layer.weight, gain=self.gain)
        # for _ in self.comp_vec:
            # nn.init.xavier_normal_(_.data, gain=1)

    def forward(self, input_data, features, target_index):
        # =============== node mapping =================== #
        transformed_feature = self.node_transformation(features)
        # layers
        re_and_ortho_loss_list = []
        h = transformed_feature
        i = 0
        for layer in self.IMDBModelLayers:
            i+=1
            h, re_and_ortho_loss = layer(input_data, h)
            re_and_ortho_loss_list.append(re_and_ortho_loss)

        logits = self.output_layer(h)
        re_and_ortho_loss = sum(re_and_ortho_loss_list)
        return logits[target_index], h[target_index], re_and_ortho_loss
