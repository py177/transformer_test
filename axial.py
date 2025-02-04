import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialAttention(nn.Module):
    def __init__(self, embed_size, num_heads, axis=0):
        super(AxialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.axis = axis

        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, mask=None):
        batch_size, seq_len, _ = query.shape

        Q = self.query_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 轴向注意力机制
        if self.axis == 0:
            attention_scores = attention_scores.mean(dim=1)  # 沿头维度平均
        elif self.axis == 1:
            attention_scores = attention_scores.mean(dim=2)  # 沿序列维度平均

        # 应用 mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权后的 V
        attention_output = torch.matmul(attention_weights.unsqueeze(1), V).squeeze(1)

        # 确保形状正确 (batch_size, seq_len, embed_size)
        attention_output = attention_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # 线性变换到原始维度
        output = self.out_linear(attention_output)

        return output
