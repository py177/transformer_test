import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryAugmentedAttention(nn.Module):
    def __init__(self, embed_size, num_heads, memory_size=128):
        super(MemoryAugmentedAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.memory_size = memory_size

        # 线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

        # 记忆矩阵 (memory_size, head_dim)，确保 head_dim 对齐
        self.memory = nn.Parameter(torch.randn(memory_size, self.head_dim))

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        # 线性变换
        Q = self.query_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算标准注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 计算记忆增强部分，并进行广播适配
        memory_scores = torch.matmul(Q, self.memory.T.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1))

        # **修正 memory_scores 形状，使其匹配 attention_scores**
        memory_scores = memory_scores[:, :, :, :seq_len]  # 截断 memory_size 以匹配 seq_len

        # 融合注意力得分
        attention_scores = attention_scores + memory_scores  

        # Mask 处理
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算最终注意力输出
        attention_output = torch.matmul(attention_weights, V)

        # 还原形状
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 输出层
        output = self.out_linear(attention_output)

        return output
