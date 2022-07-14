from abc import ABC
import torch
import torch.nn as nn
import dgl
from model.base_models.NodeTransform import NodeTransform
from model.base_models.EgoGraphEncoding import EgoGraphEncoding, CompGCNEncoder
from model.base_models.AutoMvFusion import AutoMvFusion

class BaseLayer(nn.Module, ABC):
    """
    HetG for represent a kind of node
    """
    def __init__(self, args, metapath_list, edge_type_list, num_edge, relational_encoders):
        super(BaseLayer, self).__init__()
        # args
        self.hidden_dim, self.intra_dim, self.inter_dim = args.hidden_dim, args.intra_dim, args.inter_dim

        # ============= metapath ================= #
        self.metapath_list = metapath_list
        self.edge_type_list, self.num_edge_type = edge_type_list, num_edge
        
        # ============= public paramerters for ego graph encoding
        self.relational_encoders = relational_encoders

        # ================ key component ===================== #
        # =============== 2. metapath encoder ================== #
        self.ego_graph_encoder = nn.ModuleList([EgoGraphEncoding(metapath=self.metapath_list[_]) for _ in range(len(metapath_list))])
        # ================= 3. semantic fusion ================== #
        self.auto_mv_fusion = AutoMvFusion(self.hidden_dim, self.intra_dim, self.inter_dim, self.metapath_list)

    def forward(self, input_data, transformed_features):
        metapath_outs = {'-'.join(map(str, metapath)): None for metapath in self.metapath_list}

        for index, metapath_layer_data in enumerate(input_data):
            # metapath_layer_data[0] = hierarchical relational graphs
            # metapath_layer_data[1] = target index
            # metapath_layer_data[2] = feature index. node index of each hierarchical relational graph
            # metapath_layer_data[3] = batch metapath instances lists corresponding to hierarchical relational graphs
            hierarchical_relational_graphs  = metapath_layer_data[0]
            edge_types                      = self.edge_type_list[index]
            transformed_features            = transformed_features
            _, new_index_arrangement        = metapath_layer_data[1]
            feature_index                   = metapath_layer_data[2]

            metapath_outs['-'.join(map(str, self.metapath_list[index]))] = \
                self.ego_graph_encoder[index](hierarchical_relational_graphs, edge_types,
                                                       self.relational_encoders,
                                                       transformed_features, feature_index)[new_index_arrangement]
        
        # ================== metapath fusion ====================#
        fused_features, re_and_ortho_loss = self.auto_mv_fusion(metapath_outs)

        return fused_features, re_and_ortho_loss

class LinkPredictionLayer(nn.Module):
    def __init__(self, type_mask, args, edge_type_list_full, num_edge_type, metapath_list_full, relational_encoders):
        super(LinkPredictionLayer, self).__init__()
        # args
        self.hidden_dim = args.hidden_dim
        self.type_mask  = type_mask

        # ================ metapaths ===================== #
        self.metapath_list_full = metapath_list_full
        self.edge_type_list_full, self.num_edge_type = edge_type_list_full, num_edge_type

        self.metapath_list_user = self.metapath_list_full[0]
        self.metapath_list_item = self.metapath_list_full[1]

        self.edge_type_user = self.edge_type_list_full[0]
        self.edge_type_item = self.edge_type_list_full[1]

       # =============================== gcn encoder and compositional vec =================================== #
        self.relational_encoders = relational_encoders
        
        # ============ two based layer for node representation ========== #
        # user base layer in imdb model
        self.user_layer = BaseLayer(args, self.metapath_list_user, self.edge_type_user,
                                    self.num_edge_type, self.relational_encoders)
        self.item_layer = BaseLayer(args, self.metapath_list_item, self.edge_type_item,
                                    self.num_edge_type, self.relational_encoders)

        # ================= output layer ===================== #
        self.a = torch.nn.Sigmoid()

    def forward(self, input_data, transformed_features, **kw):
        h_user, re_and_ortho_loss_user = self.user_layer(input_data[0], transformed_features)
        h_item, re_and_ortho_loss_item = self.item_layer(input_data[1], transformed_features)
        return [h_user, h_item], [h_user, h_item], re_and_ortho_loss_user + re_and_ortho_loss_item


class LastFMModel(nn.Module):
    def __init__(self, type_mask, input_feature_dims, args):
        super(LastFMModel, self).__init__()
        # args
        self.hidden_dim = args.hidden_dim
        self.gain,  self.use_activation, self.dropout_rate  = args.gain,    True,   args.dropout_rate

        # metapath
        self.metapath_list_full = [[(0, 1, 0), (0, 1, 2, 1, 0), (0, 0)],
                                   [(1, 0, 1), (1, 2, 1), (1, 0, 0, 1)]]
        self.edge_type_list_full, self.num_edge_type = [[[0, 1], [0, 2, 3, 1], [4]],
                                                        [[1, 0], [2, 3], [1, 4, 0]]], 5

        # =============== public parameters for ego graph encoding =============== #
        self.comp_vec               = nn.ParameterList([nn.Parameter(torch.empty(1, self.hidden_dim)) for _ in range(self.num_edge_type)])
        self.gcn_encoders           = nn.ModuleList([dgl.nn.GraphConv(self.hidden_dim, self.hidden_dim, norm='right') for _ in range(3)])
        self.relational_encoders    = torch.nn.ModuleList([CompGCNEncoder(self.comp_vec[_], self.get_gcn_encoder(_)) for _ in range(self.num_edge_type)])

        # =============================== key component ================================= #
        # node transformation
        self.node_transformation    = NodeTransform(input_feature_dims, self.hidden_dim, type_mask, 'LastFM', self.dropout_rate, self.use_activation)
        # layers
        # =============== link prediction layer =================#
        self.link_predict           = LinkPredictionLayer(type_mask, args, self.edge_type_list_full, self.num_edge_type, self.metapath_list_full, self.relational_encoders)

        self.init_parameters()

    def get_gcn_encoder(self, e_type):
        if e_type in [1, 2, 4]:
            return self.gcn_encoders[1]
        elif e_type in [0, 3]:
            return self.gcn_encoders[2]

    def init_parameters(self):
        # comp vec
        for _ in self.comp_vec:
            nn.init.xavier_normal_(_.data, gain=self.gain)
        
        # node transformation
        pass

        # gcn conv
        pass
        
        # fusion
        nn.init.orthogonal_(self.link_predict.user_layer.auto_mv_fusion.V, gain=self.gain)
        nn.init.orthogonal_(self.link_predict.user_layer.auto_mv_fusion.V_reverse, gain=self.gain)

        nn.init.orthogonal_(self.link_predict.item_layer.auto_mv_fusion.V, gain=self.gain)
        nn.init.orthogonal_(self.link_predict.item_layer.auto_mv_fusion.V_reverse, gain=self.gain)

        for fc in self.link_predict.user_layer.auto_mv_fusion.fc_list:
            nn.init.orthogonal_(fc, gain=self.gain)
            
        for fc in self.link_predict.user_layer.auto_mv_fusion.fc_list_reverse:
            nn.init.orthogonal_(fc, gain=self.gain)

        for fc in self.link_predict.item_layer.auto_mv_fusion.fc_list:
            nn.init.orthogonal_(fc, gain=self.gain)
            
        for fc in self.link_predict.item_layer.auto_mv_fusion.fc_list_reverse:
            nn.init.orthogonal_(fc, gain=self.gain)

    def forward(self, input_data, features):
        # node transformation
        transformed_features = self.node_transformation(features)

        [fc_user, fc_item], [h_user, h_item], loss = self.link_predict(
            input_data, transformed_features)

        return [fc_user, fc_item], [h_user, h_item], loss
