import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer  # 替换 evaluate 库
from tqdm import tqdm  # 添加进度条库

# 导入自定义的注意力模块
from multihead import MultiHeadAttention
from adaptive import AdaptiveAttention
from local import LocalAttention
from adversarial import AdversarialAttention
from attentiondistillation import AttentionDistillation
from attentionpooling import AttentionPooling
from axial import AxialAttention  # AxialAttention 修正
from bert_bidirectional import BertAttention
from convolution import ConvolutionalAttention
from co import CoAttention
from cross import CrossAttention
from frequency import FrequencyDomainAttention
from gated import GatedAttention
from globaltrans import GlobalAttention
from graph import GraphAttention
from hard import HardAttention
from hierarchical import HierarchicalAttention
from hybrid import HybridAttention
from memoryaugument import MemoryAugmentedAttention
from self import SelfAttention
from soft import SoftAttention
from sparse import SparseAttention
from XL_repeatparagraph import TransformerXLAttention

# 加载数据集
os.environ["HF_HUB_URL"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.abspath("./hf_cache")

# 直接加载整个数据集
dataset = load_dataset("wmt14", "de-en", cache_dir="./hf_cache", download_mode="force_redownload")

# 保留整个训练集和验证集
train_dataset = dataset["train"]  
val_dataset = dataset["validation"]

# 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir="./hf_cache")

# 数据加载器
def create_dataloader(dataset, batch_size=64):
    def collate_fn(batch):
        input_ids = [tokenizer.encode(entry['translation']['de'], truncation=True, padding='max_length', max_length=128) for entry in batch]
        target_ids = [tokenizer.encode(entry['translation']['en'], truncation=True, padding='max_length', max_length=128) for entry in batch]
        return torch.tensor(input_ids), torch.tensor(target_ids)
    
    return DataLoader(dataset['train'], batch_size=batch_size, collate_fn=collate_fn), \
           DataLoader(dataset['validation'], batch_size=batch_size, collate_fn=collate_fn)

# 定义模型参数
embed_size = 512
num_heads = 8
batch_size = 64

# 创建嵌入层
vocab_size = tokenizer.vocab_size
embedding = nn.Embedding(vocab_size, embed_size).to("cuda" if torch.cuda.is_available() else "cpu")

# 实例化所有模型
models = {
    "MultiHeadAttention": MultiHeadAttention(embed_size, num_heads),
    "AdaptiveAttention": AdaptiveAttention(embed_size, num_heads),
    "LocalAttention": LocalAttention(embed_size, num_heads, window_size=5),
    "AdversarialAttention": AdversarialAttention(embed_size, num_heads),
    "AttentionDistillation": AttentionDistillation(embed_size, num_heads),
    "AttentionPooling": AttentionPooling(embed_size, num_heads),
    "AxialAttention": AxialAttention(embed_size, num_heads, axis=0),  # 这里修正
    "BertBidirectional": BertAttention(embed_size, num_heads),
    "ConvolutionalAttention": ConvolutionalAttention(embed_size, num_heads, kernel_size=3),
    "CoAttention": CoAttention(embed_size, num_heads),
    "CrossAttention": CrossAttention(embed_size, num_heads),
    "FrequencyDomainAttention": FrequencyDomainAttention(embed_size, num_heads),
    "GatedAttention": GatedAttention(embed_size, num_heads),
    "GlobalAttention": GlobalAttention(embed_size, num_heads),
    "GraphAttention": GraphAttention(embed_size, num_heads),
    "HardAttention": HardAttention(embed_size, num_heads),
    "HierarchicalAttention": HierarchicalAttention(embed_size, num_heads),
    "HybridAttention": HybridAttention(embed_size, num_heads),
    "MemoryAugmentedAttention": MemoryAugmentedAttention(embed_size, num_heads, memory_size=128),
    "SelfAttention": SelfAttention(embed_size, num_heads),
    "SoftAttention": SoftAttention(embed_size, num_heads),
    "SparseAttention": SparseAttention(embed_size, num_heads, sparsity=0.1),
    "TransformerXLAttention": TransformerXLAttention(embed_size, num_heads, segment_len=128)
}

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for model in models.values():
    model.to(device)

# BLEU 计算的平滑函数
smoothing = SmoothingFunction()

# 评估函数
def evaluate_model(model, model_key, val_loader):
    model.eval()
    total_loss = 0
    all_predictions, all_references = [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with torch.no_grad():
        for input_ids, target_ids in tqdm(val_loader, desc=f"Evaluating {model_key}"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            embedded_input, target_embeddings = embedding(input_ids), embedding(target_ids)
            
            if model_key in ["AxialAttention", "ConvolutionalAttention", "SelfAttention"]:
                output = model(embedded_input)
            elif model_key == "CoAttention":
                output = model(embedded_input, embedded_input, embedded_input, embedded_input, embedded_input, embedded_input)
            else:
                output = model(embedded_input, embedded_input, embedded_input)
            
            if isinstance(output, tuple):
                output, extra_loss = output
            
            loss = F.mse_loss(output, target_embeddings)
            total_loss += loss.item()
            
            preds_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            refs_text = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
            
            all_predictions.extend(preds_text)
            all_references.extend(refs_text)
    
    bleu_score = sum(sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing.method1)
                     for pred, ref in zip(all_predictions, all_references)) / len(all_predictions)
    
    rouge_scores = [scorer.score(pred, ref) for pred, ref in zip(all_predictions, all_references)]
    avg_rouge = {key: sum(score[key].fmeasure for score in rouge_scores) / len(rouge_scores) for key in ['rouge1', 'rouge2', 'rougeL']}
    
    avg_loss = total_loss / len(val_loader)
    print(f'{model_key} - Validation Loss: {avg_loss:.4f}, BLEU: {bleu_score:.4f}, ROUGE: {avg_rouge}')

# 创建数据加载器
train_loader, val_loader = create_dataloader(dataset, batch_size=batch_size)

# 训练和评估
num_epochs = 1
for epoch in range(num_epochs):
    for name, model in tqdm(models.items(), desc="Evaluating Models"):
        evaluate_model(model, name, val_loader)
