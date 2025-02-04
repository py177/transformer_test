import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(GraphAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # 初始化线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape  # 获取批次大小和序列长度

        # 计算邻接矩阵（用于图结构信息，shape: [seq_len, seq_len]）
        adj_matrix = torch.randn(seq_len, seq_len, device=query.device)  # 动态生成邻接矩阵

        # 将查询、键、值进行线性变换并分配到多个头
        Q = self.query_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 加入邻接矩阵影响
        attention_scores = attention_scores + adj_matrix.unsqueeze(0).unsqueeze(0)  # 维度匹配

        # 如果有mask，屏蔽某些位置
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 使用softmax计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权后的值
        attention_output = torch.matmul(attention_weights, V)

        # 还原形状 (batch_size, seq_len, embed_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 最后通过线性变换层得到输出
        output = self.out_linear(attention_output)

        return output  # 返回最终的图注意力输出
