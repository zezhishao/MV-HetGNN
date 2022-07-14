from abc import ABC
import numpy as np
import torch
import torch.nn as nn

class NodeTransform(nn.Module, ABC):
    def __init__(self,          input_feature_dims, hidden_dim, type_mask,  dataset, dropout_rate,  use_activation):
        super(NodeTransform, self).__init__()
        # =================== parameters ===================== #
        self.hidden_dim         =   hidden_dim
        self.type_mask          =   type_mask
        self.dataset            =   dataset
        self.input_feature_dim  =   input_feature_dims

        self.use_activation     =   use_activation    # default is ReLU()
        self.feat_drop          =   nn.Dropout(dropout_rate)   # dropout layer
        self.fc_list            =   get_transform_matrix_list(input_feature_dims, hidden_dim, self.dataset)     # bias = True
        
        if self.use_activation:
            self._activation    =   nn.ReLU()

    def forward(self, features):
        device = features[0].device

        num_nodes               = self.type_mask.shape[0]
        transformed_features    = torch.zeros((num_nodes, self.hidden_dim), device=device)

        # ================= context node feature mapping ========== #
        for i, (feature, fc) in enumerate(zip(features, self.fc_list)):
            node_indices                        =   np.where(self.type_mask == i)[0]
            transformed_features[node_indices]  =   fc(feature)
            
        transformed_features        =   self.feat_drop(transformed_features)
        if self.use_activation:
            transformed_features    =   self._activation(transformed_features)
        return transformed_features

def get_transform_matrix_list(input_feature_dims, hidden_dim, dataset):
    gnn_list = nn.ModuleList()
    for node_type in range(len(input_feature_dims)):
        gnn_list.append(nn.Linear(input_feature_dims[node_type], hidden_dim))
    return gnn_list
