import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入功能模块，包含常用的数学函数

class SparseAttention(nn.Module):  # 定义稀疏注意力类，继承 nn.Module
    def __init__(self, embed_size, num_heads, sparsity=0.1):  # 构造函数，初始化嵌入维度、头数和稀疏度
        super(SparseAttention, self).__init__()  # 调用父类构造函数
        self.num_heads = num_heads  # 记录头数
        self.head_dim = embed_size // num_heads  # 每个头的维度
        self.sparsity = sparsity  # 稀疏度，控制保留的注意力得分的比例

        # 初始化线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):  # 前向传播，计算稀疏注意力
        batch_size = query.shape[0]  # 获取批次大小

        # 将查询、键、值进行线性变换并分配到多个头
        Q = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分，点积除以头的维度的平方根进行缩放
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 进行稀疏化处理，保留最高的几个得分
        top_k_scores, _ = torch.topk(attention_scores, int(self.sparsity * attention_scores.size(-1)), dim=-1)
        attention_scores = attention_scores.masked_fill(attention_scores < top_k_scores[:, :, :, -1:], float('-inf'))

        # 使用softmax计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权后的值
        attention_output = torch.matmul(attention_weights, V)

        # 将所有头的输出拼接起来
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # 最后通过线性变换层得到输出
        output = self.out_linear(attention_output)
        
        return output  # 返回最终的稀疏注意力输出
