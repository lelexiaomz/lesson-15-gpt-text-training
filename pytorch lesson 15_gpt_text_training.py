# ============================================================
# PyTorch Lesson 15: GPT Text Training
# PyTorch 第十五课：GPT 文本训练
# Author: lelexiaomz
#
# 目的：用一份小型固定文本演示 GPT 风格 Transformer 的数据准备、训练、评估和下一个 Token 预测。
# Purpose: Demonstrate data preparation, training, evaluation, and next-token prediction with a GPT-style Transformer on a small, fixed text corpus.
#
# 当前分词器使用单词级 Token，并将标点符号作为独立 Token。词表由整份固定文本构建；因此，预测输入中的 Token 必须都已出现在训练文本中。
# The tokenizer uses word-level tokens and treats punctuation as separate tokens. The vocabulary is built from the fixed corpus, so every prediction-input token must occur in the training text.
#
# 程序主要结构 / Main components:
# ├── 文本读取、分词和词表构建 / Text loading, tokenization, and vocabulary construction
# ├── 训练、验证和测试数据集准备 / Training, validation, and test dataset preparation
# ├── SelfAttention、MLPBlock、TransformerBlock 和 Transformer
# ├── GPTInput 和 SimpleGPT
# ├── train_one_epoch()、validate()、train_model() 和 test_model()
# ├── predict_next_token()
# └── plot_loss_curves()
# ============================================================

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt



# 利用正则表达式进行简单的英文分词。
# Use a regular expression for simple tokenization.
import re 
# 用于构造与当前 Python 文件位于同一目录下的训练文本路径。
# Used to build the path to the training text beside this Python file.
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. 文本数据处理 / Text processing
# ============================================================

def load_training_text(file_path):
    
    # 读取 UTF-8 编码的训练文本，并返回完整文本内容。
    # Read the UTF-8 training text and return its full contents.
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def tokenize_text(text):

    # 将文本转为小写，再把连续的单词字符和每个非空白标点符号分别提取为 Token。
    # Convert the text to lowercase, then extract runs of word characters and individual non-whitespace punctuation marks as tokens.
    # 因此，"Tom" 和 "tom" 会映射为同一个 Token；空白字符不会成为 Token。
    # Consequently, "Tom" and "tom" map to the same token; whitespace is not tokenized.

    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    
    return tokens



def build_vocabulary(tokens):
    # 根据 Token 序列建立双向映射：Token → ID 和 ID → Token。
    # Build two mappings from the tokens: token-to-ID and ID-to-token.

    # set(tokens) 去重；sorted(...) 按字典序排序，使同一词表每次获得相同的 ID 分配。
    # set(tokens) removes duplicates; sorted(...) assigns IDs in lexical order for a reproducible vocabulary.
    vocab = sorted(set(tokens))

    # token_to_id 将 Token 映射为从 0 开始的整数 ID。
    # token_to_id maps each token to an integer ID starting at 0.
    token_to_id = {token: idx for idx, token in enumerate(vocab)}

    # id_to_token 将整数 ID 映射回 Token。
    # id_to_token maps integer IDs back to tokens.
    id_to_token = {idx: token for idx, token in enumerate(vocab)}
    
    return token_to_id, id_to_token



def encode_tokens(tokens, token_to_id):
    # 按原顺序将 Token 转换为 ID，并保留重复项。
    # Convert tokens to IDs in order, preserving repeated tokens.
    # 若 Token 不在词表中，字典查找会引发 KeyError；本函数不会自动忽略未知 Token。
    # A token absent from the vocabulary raises KeyError; this function does not silently skip unknown tokens.
    token_ids = [token_to_id[token] for token in tokens]

    return token_ids    



def decode_tokens(token_ids, id_to_token):
    # 按原顺序将 Token ID 转回 Token；无效 ID 会被忽略。
    # Convert token IDs back to tokens in order; IDs not present in the mapping are skipped.

    tokens = [id_to_token[token_id] for token_id in token_ids if token_id in id_to_token]

    return tokens



def create_training_sequences(token_ids, seq_len):
    # 使用滑动窗口构造下一个 Token 预测样本；输入与目标长度均为 seq_len，目标相对输入右移一位。
    # Use a sliding window to create next-token prediction examples. Inputs and targets both have length seq_len, with targets shifted one position to the right.
    # 例如，token_ids=[0, 1, 2, 3, 4, 5] 且 seq_len=4 时，产生两个样本：
    # For token_ids=[0, 1, 2, 3, 4, 5] and seq_len=4, this creates two examples:
    # 输入 / Inputs:  [[0, 1, 2, 3], [1, 2, 3, 4]]
    # 目标 / Targets: [[1, 2, 3, 4], [2, 3, 4, 5]]
    
    input_sequences = []
    target_sequences = []

    for i in range(len(token_ids) - seq_len):

        # 取出从位置 i 开始、长度为 seq_len 的输入窗口。
        # Take an input window of length seq_len starting at position i.
        input_seq = token_ids[i:i + seq_len]

        # 取出右移一位的目标窗口，用于逐位置预测下一个 Token。
        # Take the target window shifted one position to the right for next-token prediction at each position.
        target_seq = token_ids[i + 1: i+seq_len + 1]

        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    
    return input_sequences, target_sequences



def prepare_dataset(text_path, seq_len, train_ratio=0.8, val_ratio=0.1):
    # 读取文本、建立词表、按顺序划分数据，再分别构造训练样本。
    # Load the text, build its vocabulary, split the token stream in order, and create examples for each split.

    # 1. 读取文本并进行分词 / Load and tokenize the text

    training_text = load_training_text(text_path)
    tokens = tokenize_text(training_text)

    if not tokens:
        raise ValueError('\nTraining text is empty!')

    print(f'\n--------- Training text ---------')
    print(f'File path of training text: {text_path}')
    print(f'\nTotal number of tokens: {len(tokens)}')
    print(f'\nFirst 30 tokens: {tokens[:30]}')


    # 2. 建立词表并将文本编码为 Token ID / Build the vocabulary and encode the text

    # 本示例使用整份固定文本建立词表，包括验证集和测试集中的 Token。
    # This example builds the vocabulary from the full fixed corpus, including tokens in the validation and test splits.
    token_to_id, id_to_token = build_vocabulary(tokens)

    token_ids = encode_tokens(tokens, token_to_id)

    # 检查编码后再解码是否还原原始 Token 序列。
    # Check that encoding and then decoding reproduces the original token sequence.
    recovered_tokens = decode_tokens(token_ids, id_to_token)

    print(f'\n--------- Vocabulary ---------')
    print(f'Vocabulary size:     {len(token_to_id)}')
    print(f'Encode/decode match: {tokens == recovered_tokens}')


    # 3. 按顺序划分 Token 序列 / Split the token stream in order

    if seq_len <= 0:
        raise ValueError('\nseq_len must be greater than 0!')

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError('\nTrain_ratio and val_ratio must be positive,' \
                          'and their sum must be less than 1!')

    total_tokens = len(token_ids)

    train_end = int(total_tokens * train_ratio)
    val_end = train_end + int(total_tokens * val_ratio)

    train_token_ids = token_ids[:train_end]
    val_token_ids = token_ids[train_end:val_end]
    test_token_ids = token_ids[val_end:]

    # 每个数据划分至少需要 seq_len + 1 个 Token，才能生成一个输入/目标窗口对。
    # Each split needs at least seq_len + 1 tokens to produce one input/target window pair.
    for name, split_ids in [
        ('Training', train_token_ids),
        ('Validation', val_token_ids),
        ('Test', test_token_ids)
    ]:
        if len(split_ids) <= seq_len:
            raise ValueError(
                f'{name} split has only {len(split_ids)} tokens.'
                f'At least {seq_len + 1} tokens are required.'
            )

    print(f'\n--------- Dataset split ----------')
    print(f'Training tokens:   {len(train_token_ids)}')
    print(f'Validation tokens: {len(val_token_ids)}')
    print(f'Test tokens:       {len(test_token_ids)}')


    # 4. 分别构造下一个 Token 预测样本 / Create next-token prediction examples for each split

    # 先划分原始 Token 序列，再各自生成窗口，避免重叠窗口跨越数据集边界。
    # Split the original token stream before creating windows so overlapping examples do not cross split boundaries.

    train_inputs, train_targets = create_training_sequences(
        train_token_ids,seq_len
    )

    val_inputs, val_targets = create_training_sequences(
        val_token_ids,seq_len
    )

    test_inputs, test_targets = create_training_sequences(
        test_token_ids,seq_len
    )


    # 5. 转换为张量并创建 Dataset / Convert to tensors and create datasets

    # 每个张量的形状为 [样本数, seq_len]，数据类型为 torch.long，以便作为 nn.Embedding 的索引。
    # Each tensor has shape [number_of_samples, seq_len] and dtype torch.long for use as nn.Embedding indices.
    train_input_tensor = torch.tensor(train_inputs, dtype=torch.long)
    train_target_tensor = torch.tensor(train_targets,dtype=torch.long)

    val_input_tensor = torch.tensor(val_inputs,dtype=torch.long)
    val_target_tensor = torch.tensor(val_targets,dtype=torch.long)

    test_input_tensor = torch.tensor(test_inputs,dtype=torch.long)
    test_target_tensor = torch.tensor(test_targets,dtype=torch.long)

    # TensorDataset 按第 0 维索引配对输入和目标；dataset[i] 返回第 i 个输入与对应目标。
    # TensorDataset pairs inputs and targets along dimension 0; dataset[i] returns the i-th input and its matching target.
    # 两个张量的第 0 维长度必须一致。之后 DataLoader 会按 batch_size 组合多个样本。
    # Both tensors must have the same size along dimension 0. A DataLoader can then collate samples into batches.
    train_dataset = TensorDataset(
        train_input_tensor,
        train_target_tensor
    )

    val_dataset = TensorDataset(
        val_input_tensor,
        val_target_tensor
    )

    test_dataset = TensorDataset(
        test_input_tensor,
        test_target_tensor
    )

    print(f'\n--------- Prepared Datasets --------')
    print(f'Sequence length:       {seq_len}')
    print(f'Training samples:      {len(train_dataset)}')
    print(f'Validation samples:    {len(val_dataset)}')
    print(f'Test samples:          {len(test_dataset)}')

    print(f'Training input shape:  {train_input_tensor.shape}')
    print(f'Training target shape: {train_target_tensor.shape}')
    print(f'First training input IDs:  {train_inputs[0]}')
    print(f'First training target IDs: {train_targets[0]}')

    return train_dataset, val_dataset, test_dataset, token_to_id, id_to_token


# 为训练集、验证集和测试集创建 DataLoader。
# Create DataLoaders for the training, validation, and test datasets.
def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
# 2. 多头自注意力 / Multi-head self-attention
# ============================================================

class SelfAttention(nn.Module):

    def __init__(self, d_model, num_heads=1):

        """初始化因果多头自注意力层。
        Initialize a causal multi-head self-attention layer.

        Args:
            d_model: 每个 Token 表示的特征维度。
                     Feature size of each token representation.
            num_heads: 注意力头数，必须整除 d_model。
                       Number of attention heads; must divide d_model evenly.
        """

        super().__init__()

        # 注意力头数必须为正数。
        # The number of attention heads must be positive.
        if num_heads <= 0:
            raise ValueError('num_heads must be greater than 0 !')

        # d_model 必须能被注意力头数整除，才能将特征维度均分给各个头。
        # d_model must be divisible by num_heads so the features can be split evenly across heads.
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads !')

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads


        # Q、K、V 线性层均将最后一维从 d_model 映射到 d_model。
        # The Q, K, and V linear layers each map the last dimension from d_model to d_model.
        # 随后将 d_model 拆分为 num_heads 个 head_dim 维的注意力头。
        # The d_model features are then split into num_heads attention heads of head_dim features each.

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # W_O 将拼接后的多头输出投影回 d_model 维。
        # W_O projects the concatenated multi-head output back to d_model dimensions.
        self.W_O = nn.Linear(d_model, d_model)


    def forward(self, x):

        """对输入序列执行因果多头自注意力。
        Apply causal multi-head self-attention to an input sequence.

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]。
               Input tensor with shape [batch_size, seq_len, d_model].

        Returns:
            形状为 [batch_size, seq_len, d_model] 的注意力输出。
            Attention output with shape [batch_size, seq_len, d_model].
        """

        batch_size, seq_len, d_model = x.shape

        # 1. 从输入生成 Q、K、V，并将特征拆分到多个注意力头。
        # 1. Project the input to Q, K, and V, then split features across attention heads.
        # 拆分并转置后，每个张量的形状为 [batch_size, num_heads, seq_len, head_dim]。
        # After splitting and transposing, each tensor has shape [batch_size, num_heads, seq_len, head_dim].
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        #print(f'Q shape: {Q.shape}')

        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        #print(f'K shape: {K.shape}')

        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        #print(f'V shape: {V.shape}')

        # print(f'----------- self attention -----------')
        # print(f'\nQ shape: {Q.shape}')
        # print(f'Q.size(-2): {Q.size(-2)}')
        # for i in range (Q.size(-2)):
        #      print(f'Q[{i}]: {Q[0,i]}')


        # print(f'\nK shape: {K.shape}')
        # print(f'K.size(-2): {K.size(-2)}')
        # for i in range (K.size(-2)):
        #     print(f'K[{i}]: {K[0,i]}')

        # print(f'\nV shape: {V.shape}')



        # 2. 对每个注意力头计算 QK^T，得到形状 [batch_size, num_heads, seq_len, seq_len] 的分数。
        # 2. Compute QK^T independently for each head, producing scores of shape [batch_size, num_heads, seq_len, seq_len].
        # transpose(-2, -1) 交换 K 的最后两个维度，使矩阵乘法沿 head_dim 维进行。
        # transpose(-2, -1) swaps K's last two dimensions so matrix multiplication contracts over head_dim.
        scores = Q @ K.transpose(-2,-1)


        # print(f'\n--------- Scores Shape: {scores.shape} ---------')
        # for i in range(scores.size(-2)):
        #     print(f'scores[{i}]: {scores[0,i]}')


        # 3. 按 head_dim 的平方根缩放点积分数，避免分数随特征维度增大而过大。
        # 3. Scale dot-product scores by sqrt(head_dim) to prevent their magnitude from growing with the feature dimension.
        # 每个头的 Q/K 向量维度是 head_dim，而不是完整的 d_model。
        # Each head's Q/K vector has head_dim features, not the full d_model.
        scores = scores / math.sqrt(self.head_dim)

        # print(f'\n--------- Scores Shape after sqrt: {scores.shape} ---------')
        # print(f'scores sqrt: {math.sqrt(self.head_dim)}')
        # for i in range(scores.size(-2)):
        #     print(f'after sqrt scores[{i}]: {scores[0,i]}')


        # 4. 创建因果掩码，只允许每个位置关注自身及其之前的位置。
        # 4. Create a causal mask so each position can attend only to itself and earlier positions.
        # 掩码形状为 [1, 1, seq_len, seq_len]，会广播到 batch 和注意力头维度。
        # The mask has shape [1, 1, seq_len, seq_len] and broadcasts across batches and attention heads.
        seq_len = x.size(1)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        ).view(1, 1, seq_len, seq_len)


        # print(f'seq_len:{seq_len}')
        # print(f'mask shape:{mask.shape}')
        # print(f'mask:{mask}')

        # 将被屏蔽的未来位置设为负无穷；Softmax 后这些位置的权重为 0。
        # Set masked future positions to negative infinity; their Softmax weights become zero.
        # print(f'scores:{scores}')

        scores = scores.masked_fill(mask == 0, float('-inf'))

        # print(f'scores after mask:{scores}')



        # 5. 沿最后一维（key 位置）执行 Softmax，得到每个 query 对可见 key 的注意力权重。
        # 5. Apply Softmax over the last dimension (key positions) to obtain attention weights for each query.
        # 每个 query、每个注意力头的权重和为 1；因果掩码位置的权重为 0。
        # Weights sum to 1 for each query and head; causally masked positions have weight 0.
        attention = torch.softmax(scores, dim=-1)

        # print(f'\n--------- Attention: {attention.shape} ---------')
        # print(f'attention: {attention}')


        # 6. 每个注意力头根据权重对其 V 向量进行加权求和。
        # 6. Each attention head computes a weighted sum of its V vectors.
        # attention 和 V 的形状分别为 [batch, heads, seq_len, seq_len] 和 [batch, heads, seq_len, head_dim]。
        # attention and V have shapes [batch, heads, seq_len, seq_len] and [batch, heads, seq_len, head_dim].
        # context 的形状为 [batch, heads, seq_len, head_dim]；每个位置汇总其可见上下文的信息。
        # context has shape [batch, heads, seq_len, head_dim]; each position aggregates information from its visible context.
        context = attention @ V


        # 7. 拼接多个注意力头，恢复为 [batch_size, seq_len, d_model]。
        # 7. Concatenate the attention heads to restore shape [batch_size, seq_len, d_model].
        context = context.transpose(1,2).contiguous().view(
             batch_size, seq_len, self.d_model
         )

        # print(f'\n--------- Context: {context.shape} ---------')
        # for i in range(context.size(-2)):
        #     print(f'context[{i}]: {context[0,i]}')

        # 8. 通过输出投影层混合各头的信息；张量形状保持不变。
        # 8. Apply the output projection to mix information from the heads; the tensor shape remains unchanged.
        output = self.W_O(context)

        # print(f'\n--------- Output: {output.shape} ---------')
        # for i in range(output.size(-2)):
        #     print(f'output[{i}]: {output[0,i]}')

        return output


# ============================================================
# 3. 前馈网络 / Feed-forward network
# ============================================================

class MLPBlock(nn.Module):

    def __init__(self, d_model, d_ff):

        """初始化逐位置前馈网络。
        Initialize the position-wise feed-forward network.

        Args:
            d_model: Transformer 的输入和输出特征维度。
                     Transformer input and output feature size.
            d_ff: 前馈网络的隐藏层维度。
                  Hidden size of the feed-forward network.
        """

        super().__init__()

        # 第一层将特征维度从 d_model 扩展到 d_ff。
        # The first layer expands the feature dimension from d_model to d_ff.
        self.linear1 = nn.Linear(d_model,d_ff)

        # ReLU 非线性激活 / ReLU nonlinearity
        self.relu = nn.ReLU()

        # 第二层将维度投影回 d_model，使输出可与残差输入相加。
        # The second layer projects back to d_model so the output can be added to the residual input.
        self.linear2 = nn.Linear(d_ff, d_model)


    def forward(self, x):

        # d_model → d_ff / Expand the feature dimension
        x = self.linear1(x)
        # 应用非线性激活 / Apply the nonlinearity
        x = self.relu(x)
        # d_ff → d_model / Project back to the model dimension
        x = self.linear2(x)

        return x


# ============================================================
# 4. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, d_model, d_ff, num_heads=1):

        """初始化一个 Transformer block。
        Initialize one Transformer block.

        d_model: Token 表示的特征维度。 / Feature size of each token representation.
        d_ff: 前馈网络隐藏层维度。 / Hidden size of the feed-forward network.
        num_heads: 自注意力头数。 / Number of self-attention heads.
        """

        super().__init__()

        # 将模型维度和头数传给多头自注意力模块。
        # Pass the model dimension and head count to the multi-head self-attention module.
        self.attention = SelfAttention(d_model, num_heads)

        # ----------------------------------------------------
        # Attention 残差相加后的 LayerNorm，沿最后的 d_model 维归一化。
        # LayerNorm after the attention residual addition; normalization is over the final d_model dimension.
        self.norm1 = nn.LayerNorm(d_model)


        # 创建逐位置前馈子模块 / Create the position-wise feed-forward submodule
        self.mlp = MLPBlock(d_model, d_ff)

        # 前馈残差相加后的 LayerNorm / LayerNorm after the feed-forward residual addition
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x):

        # 第一子层：多头自注意力、残差连接和 LayerNorm。
        # First sublayer: multi-head self-attention, residual connection, and LayerNorm.
        # 注意力输出与 x 的形状均为 [batch, seq_len, d_model]，因此可以逐元素相加。
        # The attention output and x both have shape [batch, seq_len, d_model], so they can be added element-wise.

        # print(f'\n--------- Transformer Block ---------')
        # print(f'x shape before processing:{x.shape}')
        # for i in range(x.size(-2)):
        #     print(f'x[{i}]:{x[0,i]}')


        x = self.norm1(x + self.attention(x))


        # print(f'\n--------- Transformer Block ---------')
        # print(f'x shape after norm1:{x.shape}')
        # for i in range(x.size(-2)):
        #     print(f'x[{i}]:{x[0,i]}')
        

        # 第二子层：逐位置前馈网络、残差连接和 LayerNorm。
        # Second sublayer: position-wise feed-forward network, residual connection, and LayerNorm.
        x = self.norm2(x + self.mlp(x))

        # 输出形状与输入相同：[batch, seq_len, d_model]。
        # The output shape matches the input: [batch, seq_len, d_model].
        return x



# ============================================================
# 5. 多层 Transformer / Transformer stack
# ============================================================

class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, num_layers, num_heads=1):

        super().__init__()

        # ModuleList 将多个子模块注册到模型中，以便 PyTorch 跟踪其参数。
        # ModuleList registers the submodules so PyTorch can track their parameters.
        # 列表推导式创建 num_layers 个彼此独立的 TransformerBlock；下划线表示循环变量不使用。
        # The list comprehension creates num_layers independent TransformerBlocks; the underscore marks an unused loop variable.

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model,d_ff,num_heads)
            for _ in range(num_layers)
        ])


    def forward(self, x):

        # 依次通过每个 Transformer block / Apply each Transformer block in sequence
        for block in self.blocks:
            x = block(x)

        return x



# ============================================================
# 6. GPT 输入嵌入 / GPT input embeddings
# Token ID → Token Embedding + Position Embedding → Transformer
# Token ID → Token 嵌入 + 位置嵌入 → Transformer
# ============================================================

class GPTInput(nn.Module):

    def __init__(self, vocab_size, seq_len, d_model):

        super().__init__()

        # Token Embedding 将每个词表 ID 映射为 d_model 维可训练向量。
        # Token Embedding maps each vocabulary ID to a trainable d_model-dimensional vector.
        # 参数矩阵形状为 [vocab_size, d_model]。
        # The embedding parameter matrix has shape [vocab_size, d_model].

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )


        # Position Embedding 为每个序列位置学习一个 d_model 维向量。
        # Position Embedding learns a d_model-dimensional vector for each sequence position.
        # 参数矩阵形状为 [seq_len, d_model]，因此输入序列长度不能超过 seq_len。
        # The parameter matrix has shape [seq_len, d_model], so inputs cannot exceed seq_len positions.

        self.position_embeddding = nn.Embedding(
            seq_len,
            d_model
        )


    def forward(self,token_ids):

        # token_ids 形状为 [batch_size, actual_seq_len]。
        # token_ids has shape [batch_size, actual_seq_len].

        batch_size, seq_len = token_ids.shape

        # Token Embedding 将 ID 张量 [batch_size, actual_seq_len] 查表为向量张量 [batch_size, actual_seq_len, d_model]。
        # Token Embedding looks up IDs from [batch_size, actual_seq_len] and returns vectors of shape [batch_size, actual_seq_len, d_model].

        token_embedding = self.token_embedding(token_ids)

        #print('\n---------- GPTInput ----------')
        #print(f'token_ids:      {token_ids}')
        #print(f'token_embedding shape:{token_embedding.shape}')
        #print(f'token_embedding:{token_embedding}')


        # 为当前输入生成位置 ID 0 到 actual_seq_len-1，并放到与 token_ids 相同的设备上。
        # Create position IDs from 0 to actual_seq_len-1 on the same device as token_ids.

        positions = torch.arange(
            seq_len,
            device=token_ids.device
        )

        #print(f'positions:{positions}')

        # 将位置 ID 查表为位置向量，形状为 [actual_seq_len, d_model]。
        # Look up position IDs to obtain position vectors with shape [actual_seq_len, d_model].

        position_embeddding = self.position_embeddding(positions)

        #print(f'position_embeddding shape:{position_embeddding.shape}')
        #print(f'\nposition_embedding:{position_embeddding}')


        # 将 Token Embedding 与 Position Embedding 相加，使表示同时包含 Token 内容和位置信息。
        # Add token and position embeddings so each representation contains token content and positional information.
        # Position Embedding 的形状 [actual_seq_len, d_model] 会沿 batch 维广播。
        # The position embedding of shape [actual_seq_len, d_model] is broadcast across the batch dimension.

        x = token_embedding + position_embeddding

        #print(f'token_embedding and position_embeddding shape:{x.shape}\n')
        #print(f'\ntoken_embedding + position_embeddding:{x}')

        return x
    


# ============================================================
# 7. 完整 GPT 模型 / Complete GPT model
# ============================================================

class SimpleGPT(nn.Module):

    def __init__(self,
                vocab_size,
                seq_len,
                d_model,
                d_ff,
                num_heads,
                num_layers):

        super().__init__()

        # 输入嵌入：Token ID → Token Embedding + Position Embedding。
        # Input embeddings: token IDs → token embeddings + position embeddings.
        self.input_embedding = GPTInput(
            vocab_size,
            seq_len,
            d_model
        )

        # Transformer 主体，由多个 Transformer block 组成。
        # Transformer backbone composed of multiple Transformer blocks.
        self.transformer = Transformer(
            d_model,
            d_ff,
            num_layers,
            num_heads
        )

        # Language-model head：将每个位置的 d_model 维表示映射为词表中每个 Token 的分数。
        # Language-model head: map each d_model-dimensional position representation to scores over the vocabulary.
        self.lm_head = nn.Linear(d_model, vocab_size)

        


    def forward(self, token_ids):

        # Token ID → Token Embedding + Position Embedding。
        # Token IDs → token embeddings + position embeddings.
        x = self.input_embedding(token_ids)

        # 通过多层 Transformer block / Pass through the Transformer stack
        x = self.transformer(x)

        # [batch_size, seq_len, d_model] → [batch_size, seq_len, vocab_size]
        logits = self.lm_head(x)

        return logits


# 创建 GPT 模型、损失函数和优化器。
# Create the GPT model, loss function, and optimizer.
def create_model_components(
        vocab_size,
        seq_len,
        d_model,
        d_ff,
        num_heads,
        num_layers,
        learning_rate
):

    model = SimpleGPT(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer
    


# ============================================================
# 8. 训练一个 Epoch / Train for one epoch
#
# 执行一个完整训练轮次：遍历训练批次、计算损失、反向传播并更新参数，最后返回按 Token 数加权的平均损失。
# Run one complete training epoch: iterate over batches, compute loss, backpropagate, update parameters, and return the token-weighted mean loss.
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, vocab_size):

    # 启用训练模式 / Enable training mode
    model.train()

    total_loss = 0.0
    total_tokens = 0

    # 遍历训练集的所有批次 / Iterate over all training batches
    for batch_inputs, batch_targets in train_loader:

        # 1. 清除上一步累积的梯度 / Clear gradients accumulated from the previous step
        optimizer.zero_grad()

        # 2. 前向传播，得到每个位置对词表 Token 的预测分数。
        # 2. Run the forward pass to obtain vocabulary scores for each position.
        logits = model(batch_inputs)    

        # 3. 将 batch 和序列维展平后计算交叉熵损失。
        # 3. Flatten the batch and sequence dimensions, then compute cross-entropy loss.
        loss = criterion(
            logits.reshape(-1, vocab_size),
            batch_targets.reshape(-1)
        )

        # 4. 反向传播，计算梯度 / Backpropagate to compute gradients
        loss.backward()

        # 5. 根据梯度更新模型参数 / Update model parameters using the gradients
        optimizer.step()

        # 6. 按 Token 数累计损失，以便正确计算各批次合并后的平均值。
        # 6. Accumulate loss weighted by the number of tokens to compute the correct overall mean.
        num_tokens = batch_targets.numel()  

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    # 计算整个 Epoch 按 Token 数加权的平均损失。
    # Compute the token-weighted mean loss for the full epoch.
    avg_loss = total_loss / total_tokens

    return avg_loss


# ============================================================
# 9. 验证模型 / Validate the model
# ============================================================
def validate(model, val_loader, criterion, vocab_size):

    # 切换到评估模式 / Switch to evaluation mode
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    # 验证阶段不需要计算梯度 / Gradients are not needed during validation
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:

            # 1. 前向传播，得到预测分数 / Run the forward pass to obtain prediction scores
            logits = model(batch_inputs)

            # 2. 计算交叉熵损失 / Compute cross-entropy loss
            loss = criterion(
                logits.reshape(-1, vocab_size),
                batch_targets.reshape(-1)
            )

            # 3. 按 Token 数累计损失 / Accumulate loss weighted by token count
            num_tokens = batch_targets.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # 计算整个验证集按 Token 数加权的平均损失。
    # Compute the token-weighted mean loss over the full validation set.
    avg_loss = total_loss / total_tokens

    return avg_loss
        


# ============================================================
# 10. 完整训练流程 / Full training procedure
#
# 负责循环训练轮次、记录训练与验证损失，并返回损失历史供绘图使用。
# Run multiple epochs, record training and validation losses, and return their histories for plotting.
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, vocab_size, num_epochs, patience, model_save_path=None):

    # 保存每轮的训练损失和验证损失 / Store per-epoch training and validation losses
    train_losses = []
    val_losses = []

    # Early Stopping 相关变量 / Variables for early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0

    # 循环执行指定数量的 Epoch / Run the requested number of epochs
    for epoch in range(num_epochs):
    
        # 1. 训练一个 Epoch / Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, vocab_size)

        # 2. 在验证集上评估模型 / Evaluate on the validation set
        val_loss = validate(model, val_loader, criterion, vocab_size)

        # 3. 保存本轮损失 / Store this epoch's losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 4. 打印本轮结果 / Print this epoch's results
        print(f'\nEpoch:      [{epoch+1}/{num_epochs}]', 
              f'Train Loss: {train_loss:.4f},', 
              f'Val Loss:   {val_loss:.4f}')

        # 5. 早停策略 / Early stopping strategy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # 保存最佳模型 / Save the best model
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    return train_losses, val_losses


# ============================================================
# 11. 测试模型 / Test the model
# ============================================================

def test_model(model, test_loader, criterion, vocab_size):

    # 切换到评估模式 / Switch to evaluation mode
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    # 测试阶段不需要计算梯度 / Gradients are not needed during testing
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:

            # 1. 前向传播，得到预测分数 / Run the forward pass to obtain prediction scores
            logits = model(batch_inputs)

            # 2. 计算交叉熵损失 / Compute cross-entropy loss
            loss = criterion(
                logits.reshape(-1, vocab_size),
                batch_targets.reshape(-1)
            )

            # 3. 按 Token 数累计损失 / Accumulate loss weighted by token count
            num_tokens = batch_targets.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # 计算整个测试集按 Token 数加权的平均损失。
    # Compute the token-weighted mean loss over the full test set.
    avg_loss = total_loss / total_tokens

    print(f'\nTest Loss: {avg_loss:.4f}')

    return avg_loss   



# ============================================================
# 12. 预测下一个 Token / Predict the next token
# ============================================================

def predict_next_token(model, input_text, token_to_id, id_to_token, seq_len):

    # 1. 将输入文本按训练时相同的规则切分成 Token。
    # 1. Tokenize the input text using the same rules as during training.
    tokens = tokenize_text(input_text)

    # 输入为空时无法生成预测 / An empty input cannot be used for prediction.
    if not tokens:
        raise ValueError("Input text must contain at least one token!")

    # 2. 使用训练词表将 Token 转换为 ID；词表外 Token 会触发 KeyError。
    # 2. Convert tokens to IDs using the training vocabulary; out-of-vocabulary tokens raise KeyError.
    input_ids = encode_tokens(tokens, token_to_id)

    # 3. 如果输入过长，仅保留最后 seq_len 个 Token，以适配模型的位置嵌入长度。
    # 3. If the input is too long, keep only the last seq_len tokens to fit the model's positional embedding length.
    input_ids = input_ids[-seq_len:]

    # 4. 将 ID 列表转为 LongTensor，并增加 batch 维度，形状为 [1, actual_seq_len]。
    # 4. Convert the IDs to a LongTensor and add a batch dimension, yielding shape [1, actual_seq_len].
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # 5. 切换到评估模式并前向计算 / Switch to evaluation mode and run the forward pass
    model.eval()  

    with torch.no_grad():
        logits = model(input_tensor)

    # 6. 取序列最后一个位置对整个词表的预测分数。
    # 6. Select the vocabulary scores at the final sequence position.
    # logits[:, -1, :] 选择所有 batch、最后一个序列位置和所有词表类别。
    # logits[:, -1, :] selects all batches, the last sequence position, and all vocabulary classes.
    last_token_logits = logits[:, -1, :]

    # 7. 沿词表维度应用 Softmax，将分数转换为概率。
    # 7. Apply Softmax over the vocabulary dimension to convert scores into probabilities.
    probabilities = torch.softmax(last_token_logits, dim=-1)

    print(f'\nProbabilities shape: {probabilities.shape}')
    #print(f'Probabilities: {probabilities}')

    # 8. 选择概率最高的 Token ID。
    # 8. Select the token ID with the highest probability.
    # torch.argmax 返回最大值所在的类别索引；.item() 将单元素张量转换为 Python 整数。
    # torch.argmax returns the index of the maximum value; .item() converts the single-value tensor to a Python integer.
    predicted_token_id = torch.argmax(probabilities, dim=-1).item()

    print(f'Predicted Token ID: {predicted_token_id}')
    print(f'Predicted Token: {id_to_token.get(predicted_token_id, "<UNK>")}')

    # 9. 将预测 ID 解码为 Token / Decode the predicted ID back to a token
    predicted_token = decode_tokens([predicted_token_id], id_to_token)[0]

    return predicted_token
    

# ============================================================
# 13. 绘制训练和验证损失曲线 / Plot training and validation loss curves
# ============================================================

def plot_loss_curves(train_losses, val_losses):

    # 1. 创建 Epoch 序号 / Create epoch indices
    epochs = range(1, len(train_losses) + 1)

    # 2. 绘制训练损失 / Plot training loss
    plt.plot(epochs, train_losses, label='Training Loss')

    # 3. 绘制验证损失 / Plot validation loss
    plt.plot(epochs, val_losses, label='Validation Loss')

    # 4. 设置图表标题和坐标轴标签 / Set the plot title and axis labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 5. 显示图例和网格 / Show the legend and grid
    plt.legend()
    plt.grid()

    # 6. 显示图表 / Display the plot
    plt.show()




# ============================================================
# 主程序 / Main entry point
# ============================================================

def main():

    # 固定随机种子，使模型初始化等随机操作可复现。
    # Set the random seed for reproducible model initialization and other random operations.
    torch.manual_seed(42)

    # 每个训练样本的序列长度 / Sequence length for each training example
    seq_len = 4
        
    batch_size = 8


    # 1. 准备训练、验证和测试数据集 / Prepare training, validation, and test datasets

    text_path = (Path(__file__).resolve().parent / 'lesson_15_training_text.txt')


    train_dataset, val_dataset, test_dataset, token_to_id, id_to_token = prepare_dataset(
        text_path=text_path,
        seq_len=seq_len,
        train_ratio=0.8,
        val_ratio=0.1
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size
    )

    # 检查第一个训练批次的形状和样本 / Inspect the first training batch's shapes and examples
    for batch_inputs, batch_targets in train_loader:

        print(f'\n--------- Batch data check: ---------')
        print(f'Batch input shape:     {batch_inputs.shape}')
        print(f'Batch target shape:    {batch_targets.shape}')
        print(f'First input sequence:  {batch_inputs[0]}')
        print(f'First target sequence: {batch_targets[0]}')

        # 只检查第一个批次 / Inspect only the first batch
        break  


    # 2. 设置 Transformer 模型参数 / Configure Transformer model parameters

    # 词表大小，即可预测的 Token 类别数 / Vocabulary size, i.e. the number of token classes
    vocab_size = len(token_to_id)

    # 每个 Token 表示的特征维度 / Feature dimension of each token representation
    d_model = 32

    # 前馈网络的隐藏层维度 / Hidden dimension of the feed-forward network
    d_ff = 64

    # Transformer block 的层数 / Number of Transformer blocks
    num_layers = 2

    # 每个 Transformer block 使用的注意力头数；d_model 必须能被 num_heads 整除。
    # Number of attention heads per Transformer block; d_model must be divisible by num_heads.
    num_heads = 4

    # 优化器学习率 / Optimizer learning rate
    learning_rate = 0.001

    # 训练轮数 / Number of training epochs
    num_epochs = 100

    # 早停策略：如果验证损失在连续 patience 个 Epoch 内没有改善，则提前停止训练。
    # Early stopping strategy: if validation loss does not improve for 'patience' consecutive epochs
    patience = 10  

    # 创建模型、损失函数和优化器 / Create the model, loss function, and optimizer
    
    model, criterion, optimizer = create_model_components(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        learning_rate=learning_rate
    )

    # 如需查看模型结构，可取消下一行注释。
    # Uncomment the next line to inspect the model architecture.
    #print(model)


    # 3. 训练模型 / Train the model
    # train_losses 和 val_losses 分别记录每轮的训练损失和验证损失。
    # train_losses and val_losses store the training and validation loss for each epoch.
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        vocab_size=vocab_size,
        num_epochs=num_epochs,
        patience=patience
    )


    # 4. 使用测试集评估模型 / Evaluate the model on the test set
    test_loss = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,    
        vocab_size=vocab_size
    )


    # 5. 绘制训练和验证损失曲线 / Plot training and validation loss curves
    plot_loss_curves(train_losses=train_losses, val_losses=val_losses)


    # 6. 使用训练好的模型预测下一个 Token / Predict the next token with the trained model
    input_text = 'Mia finds a paper'

    predicted_token = predict_next_token(
        model=model,
        input_text=input_text,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        seq_len=seq_len)

    print(f'\nInput text: "{input_text}"')
    print(f'Predicted next token: "{predicted_token}"')






# ============================================================
# Python 程序入口 / Python entry point
# ============================================================
#
# 只有直接运行此文件时才调用 main()；作为模块导入时不会自动运行。
# Call main() only when this file is executed directly; importing it as a module does not run main().
# ============================================================

if __name__ == '__main__':
    main()


