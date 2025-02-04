import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入功能模块，包含常用的数学函数

class TransformerXLAttention(nn.Module):  # 定义Transformer-XL的段级重复注意力类
    def __init__(self, embed_size, num_heads, segment_len):  # 初始化嵌入维度、头数和段长度
        super(TransformerXLAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.segment_len = segment_len

        # 初始化线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):  # 前向传播，计算Transformer-XL的注意力
        batch_size = query.shape[0]

        # 将查询、键、值进行线性变换并分配到多个头
        Q = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 使用段级重复注意力，将当前段与前一个段的注意力信息进行结合
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))  # 使用mask屏蔽无关位置

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        output = self.out_linear(attention_output)

        return output  # 返回最终的注意力输出
