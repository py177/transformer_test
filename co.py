import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入功能模块，包含常用的数学函数

class CoAttention(nn.Module):  # 定义协同注意力类，继承 nn.Module
    def __init__(self, embed_size, num_heads):  # 初始化嵌入维度和头数
        super(CoAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # 初始化线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query1, key1, value1, query2, key2, value2, mask=None):  # 前向传播，计算协同注意力
        batch_size, seq_len, _ = query1.shape

        # 将查询、键、值进行线性变换并分配到多个头
        Q1 = self.query_linear(query1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K1 = self.key_linear(key1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V1 = self.value_linear(value1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q2 = self.query_linear(query2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K2 = self.key_linear(key2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V2 = self.value_linear(value2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算协同注意力得分
        attention_scores1 = torch.matmul(Q1, K2.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_scores2 = torch.matmul(Q2, K1.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 使用mask，屏蔽掉无关位置
        if mask is not None:
            attention_scores1 = attention_scores1.masked_fill(mask == 0, float('-inf'))
            attention_scores2 = attention_scores2.masked_fill(mask == 0, float('-inf'))

        # 使用softmax计算注意力权重
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        attention_weights2 = F.softmax(attention_scores2, dim=-1)

        # 计算加权后的值
        attention_output1 = torch.matmul(attention_weights1, V2)
        attention_output2 = torch.matmul(attention_weights2, V1)

        # 重新排列形状，使其符合 out_linear 输入
        attention_output = (attention_output1 + attention_output2).transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        attention_output = attention_output.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, embed_size)

        # 通过线性变换层得到最终输出
        output = self.out_linear(attention_output)

        return output  # 返回最终的注意力输出
