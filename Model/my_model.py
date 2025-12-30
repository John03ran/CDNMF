'''
CDNMF
'''

import os
import pickle
import random
import torch
import gc
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel

class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.device = config['device']
        self.net_shape = config['net_shape']
        self.att_shape = config['att_shape']
        self.net_input_dim = config['net_input_dim']
        self.att_input_dim = config['att_input_dim']
        self.is_init = config['is_init']
        self.pretrain_params_path = config['pretrain_params_path']
        self.tau = config['tau']
        self.conc = config['conc']
        self.negc = config['negc']
        self.rec = config['rec']
        self.r = config['r']
        self.model_path = config['model_path']

        # Cache for optimization
        self.A_sparse = None
        self.norm_A_sq = None
        self.norm_X_sq = None
        self.D_vec = None

        # Optimization: Add Dropout and Fusion Layer
        dropout_rate = config.get('dropout', 0.5)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fusion_layer = torch.nn.Linear(self.net_shape[-1] * 2, 1)

        self.fc1 = torch.nn.Linear(self.net_shape[-1], self.net_shape[1])
        self.fc2 = torch.nn.Linear(self.net_shape[1], self.net_shape[0])

        self.fc3 = torch.nn.Linear(self.att_shape[-1], self.net_shape[1])
        self.fc4 = torch.nn.Linear(self.net_shape[1], self.net_shape[0])

        self.U = torch.nn.ParameterDict({})
        self.V = torch.nn.ParameterDict({})

        if os.path.isfile(self.pretrain_params_path):
            with open(self.pretrain_params_path, 'rb') as handle:
                self.U_init, self.V_init = pickle.load(handle)

        if self.is_init:
            # Define trainable parameters
            module = 'net'
            # print(len(self.net_shape))
            for i in range(len(self.net_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))

            module = 'att'
            # print(len(self.att_shape))
            for i in range(len(self.att_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))
        else:
            module = 'net'
            # print(len(self.net_shape))
            for i in range(len(self.net_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.U_init[name], dtype=torch.float32)))
            self.V[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.V_init[name], dtype=torch.float32)))

            module = 'att'
            # print(len(self.att_shape_shape))
            for i in range(len(self.att_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.U_init[name]), dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.V_init[name]), dtype=torch.float32))

    def projection1(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z.t()))
        z = self.dropout(z)
        return self.fc2(z)

    def projection2(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc3(z.t()))
        z = self.dropout(z)
        return self.fc4(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        self.index_net = torch.argmax(self.V1, dim=0).long()
        O_net = F.one_hot(self.index_net, self.net_shape[-1]).float()
        
        # Optimization: Avoid constructing N x N S_net matrix
        # S_net = torch.mm(O_net, O_net.t())
        # refl_pos = refl_sim * S_net
        
        # Calculate row sums of (refl_sim * S_net) efficiently
        # sum_j (R_ij * sum_k O_ik O_jk) = sum_k O_ik * (R @ O)_ik
        refl_pos_sum = torch.sum(O_net * torch.mm(refl_sim, O_net), dim=1)

        return -torch.log((between_sim.diag()) / (refl_sim.sum(1) - refl_pos_sum + between_sim.diag()))



    def contra_loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True):
        h1 = self.projection1(z1)
        h2 = self.projection2(z2)

        ret = self.semi_loss(h1, h2)
        # l2 = self.semi_loss(h2, h1)

        # ret = (l1 + l2) * 0.5
        # ret = l1
        ret = ret.mean() if mean else ret.sum()

        return ret

    def forward(self):
        # self.V1 = F.normalize(self.V['net' + str(len(self.net_shape) - 1)], p=2, dim=0)
        # self.V2 = F.normalize(self.V['att' + str(len(self.att_shape) - 1)], p=2, dim=0)
        self.V1 = self.V['net' + str(len(self.net_shape) - 1)]
        self.V2 = self.V['att' + str(len(self.att_shape) - 1)]
        
        # Optimization: Attention-based Fusion
        V1_t = self.V1.t()
        V2_t = self.V2.t()
        cat = torch.cat([V1_t, V2_t], dim=1)
        weights = torch.sigmoid(self.fusion_layer(cat))
        V_fused = weights * V1_t + (1 - weights) * V2_t
        return V_fused.t()

    def loss(self, graph):
        # Optimization: Precompute constants and use sparse operations
        if self.A_sparse is None:
            self.A_sparse = graph.A.to_sparse()
            self.norm_A_sq = torch.square(torch.norm(graph.A))
            self.norm_X_sq = torch.square(torch.norm(graph.X)) # graph.X is N x F
            self.D_vec = torch.sum(graph.A, dim=1)

        # reconstruction A
        # P1 = U0 @ U1 ... @ V
        P1_left = self.U['net0']
        for i in range(1, len(self.net_shape)):
            P1_left = torch.mm(P1_left, self.U['net' + str(i)])
        
        i = len(self.net_shape) - 1
        V_net = self.V['net' + str(i)]
        
        # ||A - UV||^2 = ||A||^2 + ||UV||^2 - 2 Tr(V A U)
        # ||UV||^2 = Tr(V^T U^T U V) = Tr(V V^T U^T U)
        norm_P1_sq = torch.trace(torch.mm(torch.mm(V_net, V_net.t()), torch.mm(P1_left.t(), P1_left)))
        
        # Tr(V A U) = Tr(V (A U))
        # A is sparse. A @ U is dense (N x k)
        AU = torch.sparse.mm(self.A_sparse, P1_left)
        tr_VAU = torch.sum(V_net * AU.t())
        
        loss1 = self.norm_A_sq + norm_P1_sq - 2 * tr_VAU

        # reconstruction X
        # P2 = U0 @ U1 ... @ V
        P2_left = self.U['att0']
        for i in range(1, len(self.att_shape)):
            P2_left = torch.mm(P2_left, self.U['att' + str(i)])
            
        i = len(self.att_shape) - 1
        V_att = self.V['att' + str(i)]
        
        # ||X - P2||^2. X is F x N (graph.X.T). P2 is F x N.
        # P2 = P2_left @ V_att. P2_left (F x k), V_att (k x N).
        norm_P2_sq = torch.trace(torch.mm(torch.mm(V_att, V_att.t()), torch.mm(P2_left.t(), P2_left)))
        
        # Tr(P2^T X) = Tr(V^T U^T X) = Tr(X V^T U^T) ? No.
        # <X, P2> = Tr(X^T P2) = Tr(X^T U V) = Tr(V X^T U)
        # X^T is N x F (graph.X). U is F x k.
        # X^T @ U is N x k.
        XTU = torch.mm(graph.X, P2_left)
        tr_XP2 = torch.sum(V_att * XTU.t())
        
        loss2 = self.norm_X_sq + norm_P2_sq - 2 * tr_XP2

        # contrastive loss
        loss3 = self.contra_loss(self.V1, self.V2)

        # regularization loss
        # Trace(V L V^T) = Trace(V D V^T) - Trace(V A V^T)
        # Trace(V D V^T) = sum_k D_k * ||V_k||^2
        tr_VDVT_net = torch.sum(self.D_vec * torch.sum(V_net**2, dim=0))
        # Trace(V A V^T) = Trace(V (A V^T))
        AVT_net = torch.sparse.mm(self.A_sparse, V_net.t())
        tr_VAVT_net = torch.sum(V_net * AVT_net.t())
        loss4_net = tr_VDVT_net - tr_VAVT_net
        
        tr_VDVT_att = torch.sum(self.D_vec * torch.sum(V_att**2, dim=0))
        AVT_att = torch.sparse.mm(self.A_sparse, V_att.t())
        tr_VAVT_att = torch.sum(V_att * AVT_att.t())
        loss4_att = tr_VDVT_att - tr_VAVT_att
        
        loss4 = loss4_net + loss4_att

        # nonnegative loss item
        loss5 = 0
        for i in range(len(self.net_shape)):
            loss5 += torch.sum(torch.relu(-self.U['net' + str(i)])**2)
        loss5 += torch.sum(torch.relu(-self.V['net' + str(i)])**2)

        for i in range(len(self.att_shape)):
            loss5 += torch.sum(torch.relu(-self.U['att' + str(i)])**2)
        i = len(self.att_shape) - 1
        loss5 += torch.sum(torch.relu(-self.V['att' + str(i)])**2)

        loss = self.rec*(loss1 + loss2) + self.conc*loss3 + self.r*loss4 + self.negc*loss5

        return loss, loss1, loss2, loss3, loss4, loss5











