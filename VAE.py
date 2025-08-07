import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Arguments import argparser
# from egnn_pytorch import EGNN_Network
import scipy.optimize
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import math
from collections import OrderedDict

from torch_geometric.nn import MessagePassing, GCNConv, GraphNorm, BatchNorm, InstanceNorm, LayerNorm, GATConv, PairNorm, GINEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GraphMultisetTransformer

ARGS = argparser()

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 # including aromatic and self-loop edge
num_bond_direction = 3

class TriScaleFusion(nn.Module):
    def __init__(self, in_dim_s, in_dim_m, in_dim_l, hidden_dim, num_heads=4, dropout=0.1):
        """
        xs: [B, F_s], xm: [B, F_m], xl: [B, F_l]
        hidden_dim: 融合后的内部维度 D
        """
        super().__init__()
        # 投影到同一隐藏空间
        self.proj_s = nn.Linear(in_dim_s, hidden_dim)
        self.proj_m = nn.Linear(in_dim_m, hidden_dim)
        self.proj_l = nn.Linear(in_dim_l, hidden_dim)
        
        # 融合子模块：可以复用上面的 AttentionBlock（Query from one, KV from another）
        self.att_sm = CrossAttBlock(hidden_dim, num_heads, dropout)
        self.att_sl = CrossAttBlock(hidden_dim, num_heads, dropout)
        self.att_ml = CrossAttBlock(hidden_dim, num_heads, dropout)
        
        # 最终融合：拼接后降维 + 门控
        self.final_fc = nn.Linear(3 * hidden_dim, hidden_dim)
        self.gate_fc  = nn.Linear(3 * hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, xm, xl):
        # 1) 投影
        xs_p = self.proj_s(xs)  # [B, D]
        xm_p = self.proj_m(xm)  # [B, D]
        xl_p = self.proj_l(xl)  # [B, D]
        
        # 如果这里是节点级 [B,N,D]，同理亦可
        # 2) 并行两两融合
        #    Query always take the “higher” scale to preserve hierarchy:
        sm = self.att_sm(x=xs_p, y=xm_p)  # xm 接受 xs 信息
        sl = self.att_sl(x=xs_p, y=xl_p)  # xl 接受 xs 信息
        ml = self.att_ml(x=xm_p, y=xl_p)  # xl 接受 xm 信息
        
        # 3) 拼接三路结果
        concat = torch.cat([sm, sl, ml], dim=-1)  # [B, 3D]
        
        # 4) 门控加权
        gate = torch.sigmoid(self.gate_fc(concat))  # [B, D]
        fused = self.final_fc(concat)               # [B, D]
        fused = gate * fused + (1 - gate) * xm_p   # 以 xm 作为主干残差
        
        # 5) 残差 + 归一化 + 激活
        out = self.norm(fused + xm_p)
        out = self.act(out)
        out = self.dropout(out)
        return out

class CrossAttBlock(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        """
        多头通道注意力模块，输入 x (Key/Value) 与 y (Query) 都是 [B, F]。
        feature_dim: F，必须能被 num_heads 整除
        num_heads: 注意力头数
        """
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Q/K/V 各自的通道映射
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        
        # 输出线性映射 + dropout
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        
        # 残差归一化与激活
        self.norm = nn.LayerNorm(feature_dim)
        self.act  = nn.ReLU()

    def forward(self, x, y):
        """
        x: [B, F]  # Key & Value
        y: [B, F]  # Query
        return: [B, F]
        """
        B, F = x.shape
        
        # 1) 生成 Q, K, V
        q = self.q_proj(y)  # [B, F]
        k = self.k_proj(x)  # [B, F]
        v = self.v_proj(x)  # [B, F]
        
        # 2) 拆成多头: [B, heads, head_dim]
        def split_heads(t):
            return t.view(B, self.num_heads, self.head_dim)
        
        qh = split_heads(q)
        kh = split_heads(k)
        vh = split_heads(v)
        
        # 3) element-wise 缩放点乘 + sigmoid 得到注意力权重
        #    这样每个 Head 都会对自己那部分通道做 gate
        scores = torch.sigmoid(qh * kh)  # [B, heads, head_dim]
        
        # 4) 加权 V
        attn_out = scores * vh           # [B, heads, head_dim]
        
        # 5) 拼回通道维度 & 输出映射
        attn_out = attn_out.contiguous().view(B, F)  # [B, F]
        out = self.out_proj(attn_out)               # [B, F]
        out = self.dropout(out)
        
        # 6) 残差 + LayerNorm + 激活
        out = self.norm(out + y)
        out = self.act(out)
        return out

class MultiScaleEncoder(nn.Module):
    def __init__(self, inFeatures):
        super(MultiScaleEncoder, self).__init__()
        self.nodeEmbedding = nn.Embedding(num_atom_type, inFeatures // 2)
        self.charityEmbedding = nn.Embedding(num_chirality_tag, inFeatures // 2)

        nn.init.xavier_uniform_(self.nodeEmbedding.weight.data)
        nn.init.xavier_uniform_(self.charityEmbedding.weight.data)

        self.gin1 = GCNConv(inFeatures // 2, inFeatures)
        self.bn1 = GraphNorm(inFeatures)
        self.pn1  = PairNorm()
        self.gin2 = GCNConv(inFeatures, inFeatures)
        self.bn2 = GraphNorm(inFeatures)
        self.pn2  = PairNorm()
        self.gin3 = GCNConv(inFeatures, inFeatures)
        self.bn3 = GraphNorm(inFeatures)
        self.pn3  = PairNorm()
        self.gin4 = GCNConv(inFeatures, inFeatures)
        self.bn4 = BatchNorm(inFeatures)
        self.pn4  = PairNorm()
        self.gin5 = GCNConv(inFeatures, inFeatures)
        self.bn5 = BatchNorm(inFeatures)
        self.pn5  = PairNorm()

        self.attFuse = TriScaleFusion(inFeatures, inFeatures, inFeatures, inFeatures)

        self.attPool = GraphMultisetTransformer(channels=inFeatures, k=1, heads=8)

        self.projectMLP = nn.Sequential(
            nn.Linear(inFeatures, inFeatures // 2),
            nn.ReLU(),
            nn.Linear(inFeatures // 2, inFeatures // 2)
        )

        self.muFC = nn.Sequential(
            nn.Linear(inFeatures // 2, inFeatures // 2),
            BatchNorm(inFeatures // 2),
            # nn.ReLU()
        )
        self.varFC = nn.Sequential(
            nn.Linear(inFeatures // 2, inFeatures // 2),
            BatchNorm(inFeatures // 2),
            # nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.FloatTensor(std.size()).normal_(0, 0.1).to(mean.device))
        return eps.mul(std).add_(mean)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = self.nodeEmbedding(x[:,0]) + self.charityEmbedding(x[:,1])

        x1 = self.gin1(x, edge_index)
        x1 = self.bn1(x1, data.batch)
        x1 = self.pn1(x1, data.batch)
        x1 = F.dropout(F.relu(x1), 0.0, training=self.training)

        x2 = self.gin2(x1, edge_index)
        x2 = self.bn2(x2, data.batch)
        x2 = self.pn2(x2, data.batch)
        x2 = F.dropout(F.relu(x2), 0.0, training=self.training)

        x3 = self.gin3(x2, edge_index)
        x3 = self.bn3(x3, data.batch)
        x3 = self.pn3(x3, data.batch)
        x3 = F.dropout(F.relu(x3), 0.0, training=self.training)

        x4 = self.gin4(x3, edge_index)
        x4 = self.bn4(x4)
        x4 = self.pn4(x4, data.batch)
        x4 = F.dropout(F.relu(x4), 0.0, training=self.training)

        x5 = self.gin5(x4, edge_index)
        x5 = self.bn5(x5)
        x5 = self.pn5(x5, data.batch)
        x5 = F.dropout(F.relu(x5), 0.0, training=self.training)


        xfuse = self.attFuse(x1, x3, x5)
        xfuse = self.attPool(xfuse, data.batch)
        xfuse = self.projectMLP(xfuse)

        mu = self.muFC(xfuse)
        var = self.varFC(xfuse)
        latentFeature = self.reparametrize(mu, var)
        return latentFeature, mu, var

class DynamicVAEDecoder(nn.Module):
    def __init__(self, featureNums, nodeType = 119):
        super(DynamicVAEDecoder, self).__init__()
        self.nodeType = nodeType
        self.recovNodeLayer = nn.Sequential(
            nn.Linear(featureNums // 2 + 8, featureNums // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(featureNums // 2, featureNums // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(featureNums // 4, self.nodeType)
        )
        self.recovBondTypeLayer = nn.Sequential(
            nn.Linear(featureNums + 16, featureNums // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(featureNums // 2, featureNums // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(featureNums // 4, 4)
        )
        self.recovBondDirLayer = nn.Sequential(
            nn.Linear(featureNums + 16, featureNums // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(featureNums // 2, featureNums // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(featureNums // 4, 3)
        )

    def forward(self, latentFeature, data):
        nodeLevelFeatures = torch.cat([latentFeature[data.batch], data.pos], dim = -1) # [B, F] to [N, F + 8]
        reconNode = self.recovNodeLayer(nodeLevelFeatures) # [B, AtomType]

        src, dst = data.edge_index_full
        edgeLevelFeature = torch.cat([nodeLevelFeatures[src], nodeLevelFeatures[dst]], dim = -1)
        reconEdgeType = self.recovBondTypeLayer(edgeLevelFeature)
        reconEdgeDir = self.recovBondDirLayer(edgeLevelFeature)

        return reconNode, reconEdgeType, reconEdgeDir
    
class VAE(nn.Module):
    def __init__(self, tuningFlag):
        super(VAE, self).__init__()
        self.tuningFlag = tuningFlag
        self.vaeEncoder = MultiScaleEncoder(ARGS.FeatureNums)
        self.vaeDecoder = DynamicVAEDecoder(ARGS.FeatureNums)
        self.struMLP = nn.Sequential(
            nn.Linear(ARGS.FeatureNums // 2, ARGS.FeatureNums // 4),
            nn.ReLU(),
            nn.Linear(ARGS.FeatureNums // 4, ARGS.FeatureNums // 4),
            nn.ReLU(),
            nn.Linear(ARGS.FeatureNums // 4, 167)
        )
        
    def forward(self, data):
        latentFeature, mu, var = self.vaeEncoder(data)
        if self.tuningFlag == False:
            reconNode, reconBondType, reconEdgeDir = self.vaeDecoder(latentFeature, data)
            preStru = self.struMLP(latentFeature)
            return latentFeature, mu, var, reconNode, reconBondType, reconEdgeDir, preStru
        else:
            return latentFeature, mu, var
