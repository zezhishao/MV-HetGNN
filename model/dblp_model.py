from abc import ABC
import math
import torch
import torch.nn as nn
import dgl
from model.base_models.NodeTransform import NodeTransform
from model.base_models.EgoGraphEncoding import EgoGraphEncoding, CompGCNEncoder
from model.base_models.AutoMvFusion import AutoMvFusion

class DBLPModel(nn.Module, ABC):
    def __init__(self, type_mask, input_feature_dims, args):
        super(DBLPModel, self).__init__()
        # args
        self.hidden_dim, self.intra_dim,     self.output_dim        = args.hidden_dim,    args.intra_dim,    args.output_dim
        self.gain,       self.use_activation, self.dropout_rate     = args.gain,          True,               args.dropout_rate
        self.inter_dim  = args.inter_dim
        # =============== metapath ================= #
        self.metapath_list                      = ([0, 1, 0], [0, 1, 2, 1, 0], [0, 1, 3, 1, 0])
        self.edge_type_list, self.num_edge_type = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]],           6       # six kind of edges：0AP 1PA 2PT 3TP 4PC 5CP

        # =============== public parameters for ego graph encoding =============== #
        self.comp_vec               = nn.ParameterList([nn.Parameter(torch.empty(1, self.hidden_dim)) for _ in range(self.num_edge_type)])
        self.agg_convs              = nn.ModuleList([dgl.nn.GraphConv(self.hidden_dim, self.hidden_dim, norm='right') for _ in range(2)])
        self.relational_encoders    = torch.nn.ModuleList([CompGCNEncoder(self.comp_vec[_], self.get_agg_conv(_)) for _ in range(self.num_edge_type)])        # each relational encoder contains a comp_vec and a gcn_encoder to perform message passing

        # =============== Three Key Components ================= #
        # 1. node transformation #
        self.node_transformation    = NodeTransform(input_feature_dims, self.hidden_dim, type_mask, 'DBLP', self.dropout_rate, self.use_activation)
        # 2. ego graph encoder   #
        self.ego_graph_encoder      = nn.ModuleList([EgoGraphEncoding(metapath=self.metapath_list[_]) for _ in range(len(self.metapath_list))])
        # 3. auto mv fusion      #
        print("==================== mse: {0} ======================".format(args.mse))
        self.auto_mv_fusion         = AutoMvFusion(self.hidden_dim, self.intra_dim, self.inter_dim, self.metapath_list, args.mse)
        # self.auto_mv_fusion = AutoMvFusion(self.hidden_dim, self.fusion_dim, self.fusion_dim, self.metapath_list, Linear=False)
        
        # =================== output layer ====================== #
        self.output_layer           = nn.Linear(self.inter_dim, self.output_dim)   # classification layer
        self.init_parameter()

    def init_parameter(self, simple=True):
        # comp vec
        for _ in self.comp_vec:
            nn.init.xavier_normal_(_.data, gain=self.gain)
        
        # node transformation
        pass
    
        # gcn conv
        pass

        # fusion
        nn.init.orthogonal_(self.auto_mv_fusion.V, gain=self.gain)
        nn.init.orthogonal_(self.auto_mv_fusion.V_reverse, gain=self.gain)

        for fc in self.auto_mv_fusion.fc_list:
            nn.init.orthogonal_(fc, gain=self.gain)
        for fc in self.auto_mv_fusion.fc_list_reverse:
            nn.init.orthogonal_(fc, gain=self.gain)

        # output layer
        nn.init.xavier_normal_(self.output_layer.weight, gain=self.gain)

    def get_agg_conv(self, e_type):
        if e_type in [1, 2, 4]:
            return self.agg_convs[0]
        elif e_type in [0, 3, 5]:
            return self.agg_convs[1]
        
    def forward(self, input_data, features):
        # ================= node mapping =================== #
        transformed_features = self.node_transformation(features)

        # ============== ego graph encoding ================= #
        metapath_outs = {'-'.join(map(str, metapath)): None for metapath in self.metapath_list}
        for index, metapath_layer_data in enumerate(input_data):
            # metapath_layer_data[0] = hierarchical relational graphs
            # metapath_layer_data[1] = target index
            # metapath_layer_data[2] = feature index. node index of each hierarchical relational graph
            # metapath_layer_data[3] = batch metapath instances lists corresponding to hierarchical relational graphs

            hierarchical_relational_graphs  = metapath_layer_data[0]
            edge_types = self.edge_type_list[index]
            transformed_features = transformed_features
            feature_index = metapath_layer_data[2]

            metapath_outs['-'.join(map(str, self.metapath_list[index]))] = \
                self.ego_graph_encoder[index](hierarchical_relational_graphs, edge_types,
                                                       self.relational_encoders,
                                                       transformed_features, feature_index)

        # ================== metapath fusion ====================#
        fused_features, re_and_ortho_loss = self.auto_mv_fusion(metapath_outs)

        # =================== output layer =====================#
        logits = self.output_layer(fused_features)

        return logits, fused_features, re_and_ortho_loss
