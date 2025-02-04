import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AdversarialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # 初始化线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

        # 使用1×1卷积来生成对抗噪声，输入形状为 (batch, num_heads, L, L)
        self.adversarial_layer = nn.Conv2d(num_heads, num_heads, kernel_size=1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # 计算 Q, K, V，并 reshape 成 (batch, num_heads, seq_len, head_dim)
        Q = self.query_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分，形状为 (batch, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 生成对抗噪声，保持形状一致
        adversarial_noise = self.adversarial_layer(attention_scores)
        attention_scores = attention_scores + adversarial_noise

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)
        # 计算加权后的值，形状为 (batch, num_heads, seq_len, head_dim)
        attention_output = torch.matmul(attention_weights, V)

        # 拼接所有注意力头：转换形状为 (batch, seq_len, num_heads * head_dim) = (batch, seq_len, embed_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        output = self.out_linear(attention_output)
        return output
