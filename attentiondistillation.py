import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入功能模块，包含常用的数学函数

class AttentionDistillation(nn.Module):  # 定义注意力蒸馏类，继承 nn.Module
    def __init__(self, embed_size, num_heads):  # 初始化嵌入维度和头数
        super(AttentionDistillation, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # 初始化线性变换层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, teacher_attention=None, mask=None):  # 前向传播，计算注意力蒸馏
        batch_size = query.shape[0]  # 获取批次大小

        # 将查询、键、值进行线性变换并分配到多个头
        Q = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分，点积除以头的维度的平方根进行缩放
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 如果有mask，屏蔽掉无关位置
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 使用softmax计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 如果提供了教师模型的注意力权重，进行蒸馏
        if teacher_attention is not None:
            distillation_loss = F.mse_loss(attention_weights, teacher_attention)  # 计算蒸馏损失
            self.distillation_loss = distillation_loss  # 记录蒸馏损失
        else:
            distillation_loss = None

        # 计算加权后的值
        attention_output = torch.matmul(attention_weights, V)

        # 将所有头的输出拼接起来
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # 最后通过线性变换层得到输出
        output = self.out_linear(attention_output)
        
        return output, distillation_loss  # 返回最终的注意力输出和蒸馏损失（如果有）
