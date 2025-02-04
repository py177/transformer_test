import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入功能模块，包含常用的数学函数

class HierarchicalAttention(nn.Module):  # 定义层次注意力类，继承 nn.Module
    def __init__(self, embed_size, num_heads):  # 构造函数，初始化嵌入维度和头数
        super(HierarchicalAttention, self).__init__()  # 调用父类构造函数
        self.num_heads = num_heads  # 记录头数
        self.head_dim = embed_size // num_heads  # 每个头的维度

        # 初始化线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):  # 前向传播，计算注意力
        batch_size = query.shape[0]  # 获取批次大小

        # 将查询、键、值进行线性变换并分配到多个头
        Q = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分，点积除以头的维度的平方根进行缩放
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 如果有mask，屏蔽某些位置
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))  # 使用mask屏蔽无关位置

        # 使用softmax计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权后的值
        attention_output = torch.matmul(attention_weights, V)

        # 将所有头的输出拼接起来
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # 最后通过线性变换层得到输出
        output = self.out_linear(attention_output)
        
        return output  # 返回最终的注意力输出
