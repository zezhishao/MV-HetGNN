from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoMvFusion(nn.Module, ABC):
    # MAGNN_with_self_fusion_and_concat
    def __init__(self, hidden_dim, intra_dim, inter_dim, metapath_list, mse):
        """
        将不同的metapath融合到一起,但是要注意的是,这些metapath的目标类型节点应当是相同的(例如A-P-A, A-P-T-P-A,A-P-C-P-A)
        Args:
            hidden_dim: meta path outs的hidden dim
            num_heads: 多头注意力数量
            activation: 所有MetaPathLayer的输出是否经过激活函数
        """
        super(AutoMvFusion, self).__init__()
        self.hidden_dim     = hidden_dim
        self.intra_dim      = intra_dim
        self.inter_dim      = inter_dim
        self.metapath_list  = metapath_list
        self.num_metapath   = len(self.metapath_list)
        self.mse            = mse

        self.fc_list    = nn.ParameterList([nn.Parameter(torch.empty(self.hidden_dim, self.intra_dim)) for _ in self.metapath_list])  # encoder
        self.V          = nn.Parameter(torch.empty(self.intra_dim * self.num_metapath, self.inter_dim))

        self.fc_list_reverse    = nn.ParameterList([nn.Parameter(torch.empty(self.intra_dim, self.hidden_dim)) for _ in self.metapath_list])
        self.V_reverse          = nn.Parameter(torch.empty(self.inter_dim, (self.intra_dim * self.num_metapath)))

        self.beta = 100

    def orthogonal_regularization_l1(self, weight):
        weight_T    = weight.T # 64 x 192
        wTw         = torch.mm(weight_T, weight)    # 64 x 64
        assert wTw.shape[0] == weight.shape[1]
        or_loss     = self.beta * torch.sum(torch.abs(wTw * (1-torch.eye(wTw.shape[0])).to(weight.device))) # Freer regularization
        # or_loss     = self.beta * torch.sqrt(torch.sum(torch.square(wTw * (1-torch.eye(wTw.shape[0])).to(weight.device)))) # Freer regularization
        return or_loss

    def forward(self, metapath_outs):
        """
        Args:
            metapath_outs: meta path outs of same type of nodes
        Returns:
        """
        metapath_outs   = [_.view(-1, self.hidden_dim) for _ in list(metapath_outs.values())]
        # =========================== feed forward ============================= #
        # inner view
        metapath_out_inner = []
        for index, metapath_out in enumerate(metapath_outs):
            metapath_out_inner.append(F.elu(torch.mm(metapath_out, self.fc_list[index])))

        # intra view
        data_full   = torch.cat(metapath_out_inner, dim=1)
        data        = torch.mm(data_full, self.V)

        # =========================== orthognal loss =========================== #
        # inner view
        or_loss_view    = []
        for fc in self.fc_list:
            or_loss_view.append(self.orthogonal_regularization_l1(fc))
        or_loss_view    = sum(or_loss_view)
        # intra view
        or_loss     = self.orthogonal_regularization_l1(self.V)

        # =========================== reconstruction loss ========================== #
        # inner view
        metapath_outs_re = []
        for index, metapath_hidden in enumerate(metapath_out_inner):
            metapath_outs_re.append(F.relu(torch.mm(metapath_hidden, self.fc_list_reverse[index])))
        mse_losses = []
        for metapath_out, metapath_out_re in zip(metapath_outs, metapath_outs_re):
            mse_losses.append(0.5 * F.mse_loss(metapath_out, metapath_out_re))
        mse_loss_inner = sum(mse_losses) / len(mse_losses)
        # intra view
        re = torch.mm(data, self.V_reverse)
        mse_loss_intra = 0.5 * F.mse_loss(data_full, re)
        mse_loss = mse_loss_inner + mse_loss_intra

        return data, or_loss + or_loss_view + mse_loss * self.mse
