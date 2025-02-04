import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入功能模块，包含常用的数学函数

class ConvolutionalAttention(nn.Module):  # 定义卷积注意力类，继承 nn.Module
    def __init__(self, embed_size, num_heads, kernel_size=3):  # 构造函数，初始化嵌入维度、头数和卷积核大小
        super(ConvolutionalAttention, self).__init__()  # 调用父类构造函数
        self.num_heads = num_heads  # 记录头数
        self.head_dim = embed_size // num_heads  # 每个头的维度
        self.kernel_size = kernel_size  # 卷积核大小

        # 初始化线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

        # 初始化卷积层
        self.conv = nn.Conv1d(in_channels=num_heads, out_channels=num_heads, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, query, mask=None):  # 前向传播，计算卷积注意力
        batch_size, seq_len, _ = query.shape

        # 线性变换得到 Q, K, V
        Q = self.query_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 调整维度以适应 Conv1d
        attention_scores = attention_scores.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, seq_len)
        attention_scores = attention_scores.reshape(batch_size * seq_len, self.num_heads, seq_len)

        # 通过 Conv1d 处理注意力分数
        attention_scores = self.conv(attention_scores)

        # 还原形状
        attention_scores = attention_scores.reshape(batch_size, seq_len, self.num_heads, seq_len).permute(0, 2, 1, 3)

        # 应用 Mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权后的 V
        attention_output = torch.matmul(attention_weights, V)

        # 变换回 (batch_size, seq_len, embed_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 线性变换恢复原始维度
        output = self.out_linear(attention_output)

        return output  # 返回最终的卷积注意力输出
