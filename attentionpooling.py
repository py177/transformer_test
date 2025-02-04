import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # 初始化线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        # 添加一个输出线性层，将拼接后的各头输出映射回 embed_size
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # 对查询、键、值进行线性变换并分配到多个头
        Q = self.query_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分，形状：(batch, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 如果有mask，则屏蔽掉无关位置
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 使用 softmax 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权后的值，形状：(batch, num_heads, seq_len, head_dim)
        attention_output = torch.matmul(attention_weights, V)

        # 将各头拼接起来，形状：(batch, seq_len, num_heads * head_dim)
        concat = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        # 通过输出线性层映射回 embed_size
        output = self.out_linear(concat)
        return output
