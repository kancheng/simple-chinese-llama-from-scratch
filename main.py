# -*- coding: utf-8 -*-
"""

LLM 学习-从 0 构建一个自己的LLM .ipynb 
LLM from scratch — build your own LLM (notebook)
"""

import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import urllib.request

# 创建一个字典用于存储 config
# Create a dict for storing config
MASTER_CONFIG = {
    # 参数放这里
    # Put parameters here
}

# 读数据（指定 UTF-8 編碼，避免 Windows 預設 cp950 解碼錯誤）
# Read data (use UTF-8 to avoid Windows cp950 decode error)
lines = open("xiyouji.txt", 'r', encoding='utf-8').read()

# 创建简易版词表（字符级）
# Build a simple character-level vocabulary
vocab = sorted(list(set(lines)))

# 查看词表前 n 个字符
# Show the first n characters of the vocabulary
head_num=50
print('词表前{}个:'.format(head_num), vocab[:head_num])

print('词表大小:', len(vocab))

# 将词表编码成为数字，普通的整数
# Encode vocabulary to integer IDs
itos = {i: ch for i, ch in enumerate(vocab)}

# 双向映射
# Bidirectional mapping
stoi = {ch: i for i, ch in enumerate(vocab)}

# 编码器
# Encoder
def encode(s):
    return [stoi[ch] for ch in s]

# 解码器
# Decoder
def decode(l):
    return ''.join([itos[i] for i in l])

# 来试一下这个高端的编解码器
# Try this encoder/decoder
decode(encode("悟空"))
encode("悟空")

# 对全文进行编码，并映射成为 tensor
# Encode full text and map to tensor
dataset = torch.tensor(encode(lines), dtype=torch.int16)

# 看一下形状，实际上就是多少个字符，一共 65 万个字符
# Check shape — total character count (~650k)
print(dataset.shape)
print(dataset)

# 构建 batch
# Build batches
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    # 切分训练集，验证集，测试集，比例为，训练 80%，验证 10%，测试 10%
    # Split into train/val/test: 80% / 10% / 10%
    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    # 将全部的训练数据作为 batch，验证集，测试集也换个变量存储（单纯为了方便看）
    # Use train/val/test as batch_data by split
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # 这里需要学习 torch.randint，生成大小为 batch_size，内部数值为随机整数的tensor。生成随机数数值域为[0, 训练集字符数量-滑动窗口大小-1] 之间的整数
    # torch.randint: random start indices in [0, len - context_window - 1]
    # 详情可以参考官方文档，或者这个博客：https://blog.csdn.net/qq_41813454/article/details/136326473
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    # print('ix输出:')


    # 这里需要学习 torch.stack，执行操作类似于 Python 的 zip 关键字，只不过操作对象是 tensor 张量，指定任意维度的张量进行组合
    # torch.stack: combine tensors along a new dimension (like zip for tensors)
    # 详情参考官方文档，或者这个博客：https://blog.csdn.net/dongjinkun/article/details/132590205

    # 这里 x 作为特征，y 作为预测值，因为文本生成任务是根据前 n 个字符，去推理后面的 1 个字符，因此 y 的构造会使窗口在保持原大小的基础上向后移一位
    # x = features, y = targets (next token); y is x shifted by one position
    # 通过滑动窗口，对 batch_data 中的训练数据，进行随机取样，相当于随机选择训练数据。
    # 在原 65 万多个字符中，随机选取一个字符作为开始，并以这个开始点，向后选取滑动窗口个数的字符，作为训练数据，向后移一位就是其目标值。  因此ix的构造不能超出 index。
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    # 返回特征值，目标值
    # Return features and targets
    return x, y

# 根据上面构造的get_batchs()函数，更新参数字典。
# Update config from get_batches() usage
MASTER_CONFIG.update({
    'batch_size': 8,          # 不解释
    'context_window': 16,      # 滑动窗口采样，设置采样大小
    'vocab_size':4325         # 咱们的西游记数据集，一共包含 4325 个不重复的汉字，标点符号
})

# 获取训练数据
# Get training batch
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# 因为是随机生成的采样，我们可以看一下数据，其中每个采样数据，来自于原文随机的起始点，每个元组为一个（x,y），可以观察每个x和y的首位去直观感受一下滑动窗口执行的操作
# Inspect random (x,y) samples to see sliding-window behavior
decoded_samples = [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

print(decoded_samples)

# 构造一个评估函数
# Evaluation function
@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
    # 评估结果存储变量
    # Container for evaluation results
    out = {}

    # 将模型置为评估模式
    # Set model to eval mode
    model.eval()

    # 分别会在训练集和验证集里通过get_batchs()函数取评估数据
    # Evaluate on train and val via get_batches
    for split in ["train", "val"]:

        losses = []

        # 评估 10 个batch
        # Evaluate over 10 batches
        for _ in range(10):
            # 拿到特征值（输入数据），以及目标值（输出数据）
            # Get input and target batches
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])

            # 把拿到的数据丢进模型，得到loss值
            # Forward pass to get loss
            _, loss = model(xb, yb)

            # 更新 loss 存储
            # Append loss
            losses.append(loss.item())

        # 这里就是大家经常在控制台看到的 "train_loss"  "valid_loss"由来
        # This gives the "train_loss" / "valid_loss" you see in logs
        out[split] = np.mean(losses)

    # 评估完了，别忘了把模型再置回训练状态，下一个epoch还要继续训练呢
    # Switch model back to train mode for next epoch
    model.train()

    return out

# 在进行分析 LlaMa 架构分析之前，我们从最简单的文本生成模型开始创建，然后在最简单的文本生成模型的基础上，把 LlaMa 的 RSM，Rope 等一点点添加进去。为此我们先：
# Before LlaMa, we start with a simple text model, then add RMSNorm, RoPE, etc.
# 创建一个有毛病的模型架构
# Create a deliberately flawed model
# 分析一下这个架构（其实也没什么分析的）
# (No deep analysis needed)
class StupidModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config

        # embedding 层，输入：词表大小，输出：维度大小
        # Embedding: vocab_size -> d_model
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # 创建线性层用于捕捉特征关系
        # Linear layers to capture features
        # 下面突击检查：这玩意是不是隐藏层！线性层堆叠越多是不是越好！堆叠越多是不是更计算开销越大！
        # LlaMa 使用的激活函数是 SwiGLU，目前在这个斯丢匹德模型架构里面先用Relu
        # LlaMa uses SwiGLU; here we use ReLU for this simple model
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # 这个命令可以背一下，或者复制粘贴到自己的学习笔记。 因为这行命令会直接帮你查看模型的参数量。
        # Print total parameter count (handy for 7B/20B/etc.)
        # 否则要么自己手算，要么就是听别人讲某某模型 7B  20B  108B   有了这个命令，你就能直接查看你创建的模型参数量多少
        print("模型参数量：", sum([m.numel() for m in self.parameters()]))

# 为我们创建的小模型添加前向传播
# Add forward pass to our small model
class SimpleBrokenModel(nn.Module):
    # init里的跟上面一样，没变化
    # __init__ same as above
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )



      # 添加前向传播函数
      # Forward pass
    def forward(self, idx, targets=None):
        # 实例化embedding层，输入映射为id的数据，输出嵌入后的数据
        # Embedding: token ids -> embeddings
        x = self.embedding(idx)

        # 线性层承接embedding层输出的数据
        # Linear layers on embedding output
        a = self.linear(x)

        # 对线性层输出的数据在最后一个维度，做softmax，得到概率分布
        # Softmax on last dim for probability distribution
        logits = F.softmax(a, dim=-1)

        # 如果有目标值（也就是我们前面的 y），则计算通过交叉熵损失计算loss结果。给输出的概率矩阵变个形状，再给目标值变个形状。  统一一下输入输出，然后计算loss。其中最后一维代表着一条数据。
        # If targets given, compute cross-entropy loss (reshape logits and targets first)
        # 此处需要了解tensor.view()函数，带上几何空间想象力去想一下矩阵的形状。
        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        # 如果没有目标值，则只返回概率分布的结果
        # If no targets, return logits only
        else:
            return logits

        # 查看参数量
        print("模型参数量：", sum([m.numel() for m in self.parameters()]))

# 这里我们设置这个模型为128维的embedding
# Set embedding dimension to 128
MASTER_CONFIG.update({
    'd_model': 128,
})

# 实例化模型，传参
# Instantiate model with config
model = SimpleBrokenModel(MASTER_CONFIG)

# 再看看参数量
# Check parameter count again
print("咱们的模型这么多参数量:", sum([m.numel() for m in model.parameters()]))
# 于是乎，我们创建了一个1128307个参数的模型，上面参数想怎么改，自己改！电脑不会爆炸！
# ~1.1M params; tweak config as you like.

# 获取训练的特征数据与目标数据
# Get training features and targets
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# 扔进模型获取概率分布矩阵与loss
# Forward pass to get logits and loss
logits, loss = model(xs, ys)
loss

# 更新参数，训练伦次，batch_size，log 日志打印步长
# Update config: epochs, log interval, batch size
MASTER_CONFIG.update({
    'epochs': 1000,
    'log_interval': 10,      # 每10个batch打印一次log
    'batch_size': 32,
})
# print log every 10 batches

# 实例化模型
# Instantiate model
model = SimpleBrokenModel(MASTER_CONFIG)

# 创建一个 Adam 优化器，基础知识，
# Create Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),      # 优化器执行优化全部的模型参数
)
# Optimizer updates all model parameters

# 构建训练函数
# Training loop
def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    # loss存储
    # Store losses
    losses = []

    # 训练时间记录开始时间
    # Start timer
    start_time = time.time()

    # 循环训练指定 epoch 的轮数
    # Loop over epochs
    for epoch in range(config['epochs']):
        # 优化器要初始化啊，否则每次训练都是基于上一次训练结果进行优化，效果甚微
        # Zero gradients (otherwise accumulation from previous step)
        optimizer.zero_grad()

        # 获取训练数据
        # Get batch
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])

        # 前向传播计算概率矩阵与 loss
        # Forward pass
        logits, loss = model(xs, targets=ys)

        # 反向传播更新权重参数，更新学习率优化器
        # Backward and optimizer step
        loss.backward()
        optimizer.step()

        # 如果提供学习率调度器，那么学习率会通过调度器进行修改，比如学习率周期性变化，或者梯度减小，增加，具体策略需要综合考虑进行设置，详情自行查询，关键字：lr_scheduler
        # Step scheduler if provided (e.g. cosine annealing)
        if scheduler:
            scheduler.step()

        # 打印 log
        # Log periodically
        if epoch % config['log_interval'] == 0:
            # 训练时间
            # Elapsed time
            batch_time = time.time() - start_time

            # 执行评估函数，在训练集和验证集上计算loss
            # Evaluate on train and val
            x = evaluate_loss(model)

            # Store the validation loss
            losses += [x]

            # 打印进度日志
            # Print progress
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")

            # 重置开始时间，用于计算下一轮的训练时间
            # Reset timer for next interval
            start_time = time.time()

            # 打印下一轮的学习率，如果使用了 lr_scheduler
            # Print LR if using scheduler
            if scheduler:
                print("lr: ", scheduler.get_lr())

    # 上面所有 epoch 训练结束，打印最终的结果
    # After all epochs, print final val loss
    print("Validation loss: ", losses[-1]['val'])

    # 返还每一步 loss 值的列表，因为我们要画图，返还的是 loss 迭代的图像
    # Return loss curve for plotting
    return pd.DataFrame(losses).plot()

# 启动训练
# Start training
train(model, optimizer)

"""上面那个训练框架存在一些问题。  回到前向传播的代码，也就是forward()中。 我们使用了 logits = F.softmax(a, dim=-1)   对线性层输出的结果做了一次概率分布的计算。  而loss的计算选择了交叉熵损失， 目标值的词表映射结果是整数，而模型输出的logits是概率矩阵。  为了使loss计算更精确，我们需要将softmax去除。  以保证交叉熵损失的计算效果更好。"""

# 拿掉 softmax，logits 改为获取最后一个线性层输出的结果，不进行softmax计算概率分布。
# Remove softmax; use raw linear output for logits (better with cross-entropy)
# 因此将这个架构取名为：不那么蠢的模型架构
# Hence the name: "not so stupid" model
class SimpleNotStupidModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )
      print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)

        # 看这里，线性层直接输出结果，不转换为概率矩阵，只修改这里，其余不动。
        # Linear output as logits, no softmax
        logits = self.linear(x)
        # print(logits.shape)

        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits
        print("Model parameters:", sum([m.numel() for m in self.parameters()]))

# 再来一次实例化各种功能，再启动一次训练
# Instantiate and train again
model = SimpleNotStupidModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)

# loss 开窍了，下降了很多
# Loss drops a lot now

# 推理函数（输出结果就别纠结其效果了，权重都没保存，就是根据模型初始化生成的随机数组成的矩阵做的推理）
# Generation function (output is random without saved weights)
def generate(model, config=MASTER_CONFIG, max_new_tokens=20):
    # 生成随机数，作为输入数据,5行一列，代表输入5个字符。 这个地方可以自行替换其他随机数测试。
    # Start with 5x1 zeros as initial context; replace with other tokens to test
    idx = torch.zeros(5, 1).long()
    print(idx[:, -config['context_window']:])
    for _ in range(max_new_tokens):
        # 因为推理的时候，依赖后面的 n 个 token，所以滑动窗口要从后往前选择输入数据的倒数几个 token，这个是超过字符数量会对输入进行截断，只选取最后几个 token：idx[:, -config['context_window']:]
        # Use last context_window tokens as input (causal sliding window)
        logits = model(idx[:, -config['context_window']:])
        # print(logits.size())
        # 得到模型输出的结果，进行解码，这里logits[:, -1, :]挺抽象的，实际上第一维度是输入的字符数，第二维度是时间步，第三维度是词表
        # logits[:, -1, :] = last time step, shape (batch, vocab)
        # 即，对每一步的解码结果，取最后一个时间步的数据，作为输出的数据。解码的过程是第一次解码，输入5个token，第二次解码依赖的是原来5个token的最后4个，加上上一步解码生成的一个，也是5个token，如此循环。
        last_time_step_logits = logits[:, -1, :]
        # print('last_time_step_logits')
        # print(last_time_step_logits.shape)
        # 计算概率分布
        # Softmax for next-token distribution
        p = F.softmax(last_time_step_logits, dim=-1)
        # print('p_shape')
        # print(p.shape)
        # 根据概率分布计算下一个token，这里使用 torch.multinomial做的是随机采样
        # Sample next token with torch.multinomial
        idx_next = torch.multinomial(p, num_samples=1)
        # print('idx_next_shape')
        # print(idx_next.shape)
        # 将新的idx通过张量拼接写入到解码序列中
        # Append new token to sequence
        idx = torch.cat([idx, idx_next], dim=-1)
    # 使用之前定义的解码函数，将 ID 转换为汉字，我们得到的 5 行 21 列的数据，来源于每一个输入字符作为开始位置，生成 20 个字符。 因为5个输入都是 0，在词表中编号为 0 的数据是'\n'。
    # Decode IDs to text; 5 rows x (1+20) cols
    print(idx.shape)
    return [decode(x) for x in idx.tolist()]

generate(model)

"""# 将LlaMa优化部分加入到上面的notStupidModel

需要做的有三部分:

  1.RMS_Norm

  2.ROPE

  3.SwiGLU

## RMSNorm
### RMSNorm快速了解

norm，做标准化，训练过程中的张量标准化操作，通过计算均值和方差，将样本进行归一化。
在大学课程《概率与统计》我们学过，样本的均值代表样本的特征，而方差代表离散程度。

因此，通过计算，让数据变为均值为0，方差为1的数据。 这样可以使数据服从标准的正态分布。

记得大学时候，老师讲这一段的时候，着重强调：“高斯分布，正态分布”，也可以叫自然分布，自然界的很多统计情况，几乎都满足高斯分布。 两边向中心靠拢，超过中心的，随着逐渐增大，会越来越少，没超过中心的，距离中心越远，数量也越来越少。而分布的众数永远都是在中间。 数学之美。

使用均值和方差计算数据的标准差，这样既保留了数据的异常值，同时维持数据的异常结构，这样可以稳定梯度，让梯度变化更稳定，减少梯度消失或者爆炸的问题，因为维持了异常结构，也能减少过拟合问题，增强泛化能力。

RMSNorm出来之前，广泛使用的 batch_normlize，针对批次数据做标准化。标准化的数值是一个batch作为一个样本总体，计算其均值与方差。

而后，又出现了 layer_norm，其是针对每个token的特征向量做归一化处理（不知道特征向量，请看本人之前的rope文章。应该可以理解token和特征向量的关系。）依旧需要计算均值和方差。


RMSNorm 和 layer_norm 的主要区别在于RMSNorm不需要同时计算均值和方差两个统计量，而只需要计算均方根这一个统计量。在模型表现效果几乎与layer_norm持平的前提下，节省7%-64%的计算量。

猜想：  既然都平方根了，突然想起程序员之神，约翰卡马克的快速平方根倒数算法了。让所有人直呼waht the f**k的神来一笔。  当然，也有可能这么经典的数值计算方法已经被集成进了numpy。

RMS基本介绍差不多了，下面开始实现RMSNorm模块
"""

class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        # torch中register_parameter()功能为：向我们建立的网络module添加parameter
        # register_parameter adds a trainable parameter to the module
        # 因此，我们需要对 pytorch 官方封装好的 RMSNorm 功能模块添加一个可以训练参数的层，命名为 scale，并初始化为形状为layer_shape，所有值为1的张量矩阵。
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        # 计算Frobenius范数（球某个矩阵中所有元素的平方和再开方得到，该范数用来衡量矩阵的大小，详情请百度）, RMS = 1/sqrt(N) * Frobenius
        # RMS = 1/sqrt(N) * Frobenius norm
        # 具体来说，torch.linalg.norm(x, dim=(1, 2))计算了x在第1和第2维度上的范数。然后，将结果乘以x[0].numel() ** -.5。x[0].numel()表示x第一个元素（即x的第一行）的元素个数，** -.5表示求平方根的倒数。
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        # print(ff_rms.shape)
        # 将ff_rms算子应用于输入的张量x，依据公式，做除法，因为输入向量x是三维的，因此需要对ff_rms进行升两维，也变成三维的张量。这样可以进行元素之间的计算。
        # Normalize x by RMS (unsqueeze for broadcasting)
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        # print(raw.shape)
        # 返回 scale 缩放后归一化的张量
        # Return scale * normalized tensor
        # print(self.scale[:x.shape[1], :].unsqueeze(0) * raw)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw

class SimpleNotStupidModel_RMS(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      # 在这里，我们添加 RMS 层
      # Add RMSNorm layer
      self.rms = RMSNorm((config['context_window'], config['d_model']))
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )
      print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        # 在这里，添加实例化后的RMS层，承接 Embedding 层输出的张量
        # Apply RMSNorm after embedding
        x = self.rms(x)

        logits = self.linear(x)
        # print(logits.shape)

        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits
        print("Model parameters:", sum([m.numel() for m in self.parameters()]))

# 好啦，这样我们对原来的 NotStupidModel 添加了 RMSNorm，现在执行一下看看
# NotStupidModel + RMSNorm; run it
model = SimpleNotStupidModel_RMS(MASTER_CONFIG)

xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)

optimizer = torch.optim.Adam(model.parameters())

train(model, optimizer)

# 在同样的训练超参数设置上，加入了RMSNorm的训练速度明显加快。
# With RMSNorm, training is noticeably faster.

"""## 增加RoPE

具体原理请参考俺的上一篇文章。
"""

def get_rotary_matrix(context_window, embedding_dim):
    # 初始化一个0 填充，形状为（context_window, embedding_dim, embedding_dim）的张量矩阵，其中context_window为token数量，后面两个embedding_dim组成正方形矩阵，与后面的attention计算对齐格式
    # Zero-initialized (context_window, d, d) rotation matrices for RoPE
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)

    # 遍历每一个位置的token
    # For each position
    for position in range(context_window):
        # 还记得我的上一篇文章中说的，对于特征，两两组合吗，因此需要循环的次数为embedding_dim除以2
        # Pair dimensions; loop embedding_dim/2 times
        for i in range(embedding_dim // 2):
            # 设置θ值，采样频率，或者说旋转频率，旋转角都可以，除以embedding_dim防止梯度问题。
            # theta = base^(-2*i/d) for RoPE
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            # 根据欧拉公式，计算旋转的角度，分别有sin 和cos，将计算拉到复数空间，并将旋转角度应用在上面的0填充的矩阵
            # Fill 2x2 rotation block with cos/sin
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
            # 得到的结果是旋转位置编码矩阵，到这里还没覆盖到attention
    return R

"""## 接下来创建注意力机制"""

# 此为单头注意力机制
# Single-head attention with RoPE
class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 计算Q权重矩阵
        # Q, K, V projection matrices
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 计算K权重矩阵
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 计算V权重矩阵
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 获得旋转位置编码矩阵，接下来会覆盖Q和K权重矩阵
        # Precompute RoPE matrix for Q and K
        self.R = get_rotary_matrix(config['context_window'], config['d_model'])


    # 这里将上一个代码块中实现的创建旋转位置编码的功能函数原封不动的拿过来
    # (Local copy of get_rotary_matrix)
    def get_rotary_matrix(context_window, embedding_dim):
        # 初始化一个0填充，形状为（context_window, embedding_dim, embedding_dim）的张量矩阵，其中context_window为token数量，后面两个embedding_dim组成正方形矩阵，与后面的attention计算对齐格式
        # (context_window, d, d) rotation matrices for RoPE
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)

        # 遍历每一个位置的token
        # For each position
        for position in range(context_window):
            # 还记得我的上一篇文章中说的，对于特征，两两组合吗，因此需要循环的次数为embedding_dim除以2
            # Pair dimensions; loop d/2 times
            for i in range(embedding_dim // 2):
                # 设置 θ 值，采样频率，或者说旋转频率，旋转角都可以，除以embedding_dim防止梯度问题。
                # theta = base^(-2*i/d)
                theta = 10000. ** (-2. * (i - 1) / embedding_dim)
                # 根据欧拉公式，计算旋转的角度，分别有 sin 和 cos，将计算拉到复数空间，并将旋转角度应用在上面的0填充的矩阵
                # Fill 2x2 rotation block with cos/sin
                m_theta = position * theta
                R[position, 2 * i, 2 * i] = np.cos(m_theta)
                R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
                R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
                R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
                # 得到的结果是旋转位置编码矩阵，到这里还没覆盖到 attention
                # Result is RoPE matrix (applied to Q/K later)
        return R

    def forward(self, x, return_attn_weights=False):
        # 前向传播时，输入矩阵的形状为(batch, sequence length, dimension)
        # Input shape: (batch, seq_len, dim)

        b, m, d = x.shape  # batch size, sequence length, dimension

        # 线性变换 Q,K,V
        # Linear projections for Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 将旋转位置编码应用于 Q 和 K ，其中 torch.bmm 为矩阵做外积，transpose 是转置，对 Q 矩阵转置，并与旋转位置编码做外积，再转置回原状， Q 便应用了旋转位置编码。
        # Apply RoPE to Q and K via batch matmul with R[:m]
        # 考虑到输入文本的长度，因此对位置编码矩阵在第一维度做截断，因为长了也没用，与文本长度一样。
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        # 同理对K也应用旋转位置编码进行覆盖
        # Same for K
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)

        # 对注意力机制点积进行等比例缩放，防止 attention 张量过长引发梯度爆炸，对应
        # Scaled dot-product attention (causal)
        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
        )
        # 如果return_attn_weights参数置为 1，则需要对 attention 进行掩码，因为在学习的时候，希望模型能依据前 n 个 token 去预测 token，而不是开卷考试。
        # If returning weights, use causal mask
        if return_attn_weights:
            # 创建注意力掩码矩阵，其中 torch.tril 函数为：对于矩阵，取左下三角，剩下的都置 0
            # Lower triangular mask (causal)
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
            # 计算注意力机制的权重矩阵，并对最后一维度做归一化，（突击检查）为什么是最后一维！因为最后一维度是每个 token 的特征向量！
            # Attention weights: QK^T/sqrt(d), then softmax on last dim (per-token logits)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations

# 单头注意力机制实现完毕，下面实现多头注意力机制
# Multi-head attention (multiple RoPE heads + linear out)
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 一个注意力机制头对象构建完毕了，多头的，首先多次创建这个对象。生成多个注意力机制头，塞到一个列表里。
        # n_heads attention heads
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        # 在模型结构上，创建一个线性层（隐藏层），用于线型输出注意力机制头输出的张量矩阵，寻找多头之间的特征，但是更主要的是，x经过多头计算后形状改变了，创建线性层，让张量矩阵变回原来输入的形状。
        # Project concatenated heads back to d_model
        # 同时为了防止过拟合，使用随机神经元失活，比率 0.1
        # 线性层输入形状：注意力机制的头数，乘以矩阵的维度，关联到俺的上一篇文章，就是 key 矩阵，在多头之间共享权重，减少计算的思维。 输出为：模型的 embedding 维度数
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 输入矩阵形状x： (batch, sequence length, dimension)
        # Input (batch, seq_len, dim)

        # 每一个注意力机制头，都传入 X 进行计算。（这个地方开启并行执行会不会快一些，但是不知道 pytorch 是不是自动调用并行）
        # Each head processes x; concat results
        heads = [h(x) for h in self.heads]
        # 输入张量x经过多个头计算 attention（同时，attention是已经覆盖了RoPE的），重新拼接成新的矩阵，重新放入变量x。到这里你应该觉得：那矩阵形状不就变了吗
        x = torch.cat(heads, dim=-1)

        # 这不，线性层的作用来了
        # Project back to d_model and dropout
        x = self.linear(x)

        # 随机失活一下，防止过拟合
        # Dropout to reduce overfitting
        x = self.dropout(x)
        return x

# Llama 32 个注意力机制头，我们来8个吧
# LlaMa uses 32 heads; we use 8

MASTER_CONFIG.update({
    'n_heads': 8,
})

# 我们已经创建完了所需要的算子，  现在积木已创建完毕，将这些积木组合起来！！！！
# All building blocks ready; assemble them into RopeModel
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding 层
        # Embedding layer
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # RMSNorm 层
        # RMSNorm layer
        self.rms = RMSNorm((config['context_window'], config['d_model']))

        # 旋转位置编码器+注意力机制
        # RoPE + multi-head attention
        self.rope_attention = RoPEMaskedMultiheadAttention(config)

        # 线性层+激活函数变为非线性输出！
        # Linear + ReLU (nonlinear)
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
        )

        # 最终的输出，因为需要解码，因为输出的维度与词表大小统一！！！
        # Final linear: output dim = vocab_size for decoding
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        print("model params:", sum([m.numel() for m in self.parameters()]))
    # 前向传播
    # Forward pass
    def forward(self, idx, targets=None):
        # embedding，不解释
        # Embedding
        x = self.embedding(idx)
        # 归一化数值，不解释
        # RMSNorm
        x = self.rms(x)
        # 相加，解释一下，因为 attention 是要覆盖到原矩阵的，想象两个形状一样的矩阵为两张纸，左手一张纸，右手一张纸，双手合十，啪！覆盖。 使用加算，就是将两个矩阵中的元素按位置相加！直接覆盖值！
        # Residual: x + attention(x)
        x = x + self.rope_attention(x)
        # 再归一化！
        # RMSNorm again
        x = self.rms(x)
        # 因为直接计算归一化的数值可能出现梯度问题，因此把归一化的值作为修正系数，再覆盖！
        # Residual: x + linear(x)
        x = x + self.linear(x)
        # 到这里，才是最终输出vocab数量的神经元输出！！！！！！
        # Final logits (vocab size)
        logits = self.last_linear(x)

        # 训练阶段有目标值
        # Training: compute loss if targets given
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        # 验证或者推理阶段，目标值y没有！只有结果，没有loss！
        # Inference: return logits only
        else:
            return logits

# 再跑一下！
# Run training again
model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)
# loss下降了百分之0.1！
# Loss drops by ~0.1%

"""## SwiGLU实现

将swish和glu结合起来。这两个激活函数单独拿出来都很强，结合起来。

这玩意挺玄学，说它不好吧，但是这玩意确实比relu这败家子保留更多语义特征参数，不至于某个权重突然到了小于0的区间，然后糊里糊涂的消失。说它好吧，它的计算量确实挺大。   

swish用了sigmoid，GLU用了门控结构（门控结构思想，可以学习一下RNN,GRU,LSTM什么的）
"""

class SwiGLU(nn.Module):

    def __init__(self, size):
        super().__init__()
        # 定义一个门控的线性层，输入输出都是门控结构的尺寸
        # Gate linear layer (in/out size)
        self.linear_gate = nn.Linear(size, size)
        # 门控结构主干线性层
        # Main linear layer
        self.linear = nn.Linear(size, size)
        # 初始化一个随机数作为 beta 系数
        self.beta = torch.randn(1, requires_grad=True)

        # nn.Parameter 用于指定某一层参数为可学习的，即本来不能通过训练更改参数，现在变成了可以经过训练来更新的参数。
        # Learnable beta (nn.Parameter)
        self.beta = nn.Parameter(torch.ones(1))
        # 将随机数 beta 指定为一个名为 beta 的神经网络层
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        # Swish 门控但愿的计算：（从括号里开始）对于原始输入的数据张量，经过线性变换乘以 beta 系数，再经过 sigmoid 变换为 0-1 之间的值，再乘以原数据经过门控线性变换。总的来说，线型输出经过非线性变换，再应用到线性变换的结果，元素按位置相乘，修正原本数据张量，就是这个门控结构做的事情。
        # SwiGLU: gate = linear_gate(x) * sigmoid(beta * linear_gate(x)); out = gate * linear(x)
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        # 将门控结构输出的值再按位乘以线型输出的原数据张量
        # 为啥这么做，我不知道，但是论文复现的代码就是这样滴，有兴趣可以研究一下，我没研究过。
        # Element-wise multiply gate with linear(x) (paper/implementation convention)
        out = swish_gate * self.linear(x)
        return out

# 再将 swiglu 添加进上面的模型
# Add SwiGLU into RopeModel
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMaskedMultiheadAttention(config)
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            # 在这里，增加了 SwiGLU 层
            # Add SwiGLU layer
            SwiGLU(config['d_model']),
        )
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])
        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        x = self.rms(x)
        x = x + self.rope_attention(x)
        x = self.rms(x)
        x = x + self.linear(x)
        logits = self.last_linear(x)

        if targets is not None:
            # Calculate cross-entropy loss if targets are provided
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

# 一二三四！再来一次！
# Train again (one more time)
model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)

# OK！ 现在我们更新一下，隐藏层维度堆叠多少层，我们先来 4 层尝尝咸淡！！！！
# Set number of layers to 4
MASTER_CONFIG.update({
    'n_layers': 4,
})

# 现在我们拥有了所有的算子，RMS，ROPE,SWIGLU，我们搭建我们的 LlaMa！ 首先实现 LlaMa 的功能块，然后堆叠。
# Assemble LlaMa from RMSNorm, RoPE, SwiGLU; implement LlamaBlock then stack.
# 功能没什么好讲的，如果仔细看到了这里，下面的每一行代码都难不住你。
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):

        x = self.rms(x)
        x = x + self.attention(x)

        x = self.rms(x)
        x = x + self.feedforward(x)
        return x

# 看一下我们的超参数字典
# Inspect config
MASTER_CONFIG

# 用 config 字典，创建 llama 的功能块
# Create one LlamaBlock with config
block = LlamaBlock(MASTER_CONFIG)

# 生成一条随机数据，丢到这个 llama 功能块里，看一下是不是有bug
# Random input to test block
random_input = torch.randn(MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model'])

# 执行以下看看输出
# Run block and check output shape
output = block(random_input)
output.shape

# 现在，我们组装LlaMa
# Assemble full Llama model
from collections import OrderedDict
class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Embedding 不解释
        # Embedding
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        # 根据传入的堆叠层数，创建 Llama 功能块，注意 OrderedDict 为一种特殊类型的字典数据，保留字典写入的顺序，先插入的数据在前，后插入的数据在后。
        # Stack n_layers LlamaBlocks (OrderedDict keeps order)
        # 这里，我们将llama的功能块堆叠4层
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        # FFN层，包含：线性层、激活函数非线性变换、再用线性层输出最终解码数值。
        # FFN: linear -> SwiGLU -> linear to vocab_size
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # 看看咱们的大模型多少参数！
        # Print total parameter count
        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        # embedding 嵌入
        # Embed
        x = self.embeddings(idx)
        # Llama 模型计算
        # Stacked Llama blocks
        x = self.llama_blocks(x)
        # FFN 计算，得到 logits
        # FFN -> logits
        logits = self.ffn(x)

        # 推理阶段没有目标值，只输出结果
        # Inference: no targets, return logits only
        if targets is None:
            return logits
        # 训练阶段，有目标值，需要输出结果，以及 loss，用于反向传播更新权重！
        # Training: return logits and loss
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

# 开始训练咱们的 Llama
# Train our Llama
llama = Llama(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = llama(xs, ys)
optimizer = torch.optim.Adam(llama.parameters())
train(llama, optimizer)

# 练它一万轮！有条件的开启，本厮实在是不愿意等了。
# Optional: train 10k epochs (commented out by default)
MASTER_CONFIG.update({
    'epochs': 10000,
})

#train(llama, optimizer, scheduler=None, config=MASTER_CONFIG)

# 再看一下推理效果（实际上也没什么效果-。-）
# Try generation (quality may be poor)
# 别忘了 generate 里面的输入数据是咱们弄的 5 个 0，如果替换为 encode 之后的数也是可以的！组成列表，转换 tensor ，这个应该没问题的吧~
generated_text = generate(llama, MASTER_CONFIG, 500)[0]
print(generated_text)

# 下面是测试集跑一下
# Run on test set
# 获取测试集的特征值和目标值
# Get test batch
xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# 丢进 Llama 获取 loss
# Forward pass on test batch for loss
logits, loss = llama(xs, ys)

print(loss)
# 4.7326！
# (example test loss)

# 还有优化的点哦，别忘了 optimizer ！以及学习率调度器！
# Further tuning: optimizer and LR scheduler
# 调整参数再来一次！
# Adjust config and train again

MASTER_CONFIG.update({
    "epochs": 1000
})

# 学习率优化器选择余弦退火
# Use cosine annealing LR scheduler
llama_with_cosine = Llama(MASTER_CONFIG)

llama_optimizer = torch.optim.Adam(
    llama.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)
# 余弦退火学习率优化器，让学习率逐渐减小，在结束时达到最低值。
# Cosine annealing: LR decays to eta_min over 300 steps
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)

# 跑一下！
# Run training with scheduler
train(llama_with_cosine, llama_optimizer, scheduler=scheduler)

# 保存你的模型！
# Save model

torch.save(llama, 'llama_model.pth')


torch.save(llama.state_dict(), 'llama_model_params.pth')

"""## 当然了，也可以换一种方式保存，并加载推理"""
# Alternatively save/load in HF-style and run inference

# 確保保存目錄存在
# Ensure save directory exists
os.makedirs("./hf_model_save", exist_ok=True)

# 保存模型权重
# Save model weights
model_save_path = "./hf_model_save/pytorch_model.bin"
torch.save(llama_with_cosine.state_dict(), model_save_path)

# 生成一个 config 文件
# Save config as JSON
import json

config_save_path = "./hf_model_save/config.json"
with open(config_save_path, 'w') as f:
    json.dump(MASTER_CONFIG, f)

# 保存 optimizer 和学习率调度器的状态，方便继续微调
# Save optimizer and scheduler state for resuming training
optimizer_save_path = "./hf_model_save/optimizer.pt"
torch.save(llama_optimizer.state_dict(), optimizer_save_path)

scheduler_save_path = "./hf_model_save/scheduler.pt"
torch.save(scheduler.state_dict(), scheduler_save_path)

# 接下来是加载模型
# Load model
llama_with_cosine = Llama(MASTER_CONFIG)

# 加载模型权重
# Load weights from disk
model_save_path = "./hf_model_save/pytorch_model.bin"
llama_with_cosine.load_state_dict(torch.load(model_save_path))

# 设置为评估模式
# Set to eval mode
llama_with_cosine.eval()

# 加载优化器和学习率调度器
# Load optimizer and scheduler state (optional, for resuming)
llama_optimizer.load_state_dict(torch.load(optimizer_save_path))
scheduler.load_state_dict(torch.load(scheduler_save_path))

# 进行推理
# Run generation
output = generate(llama_with_cosine, MASTER_CONFIG)
print(output)

"""## 接下来可以整点花活儿，比如：部署一个异步的远程服务"""
# Optional: deploy an async remote API (e.g. FastAPI)

# !pip install fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F

# 初始化 FastAPI
# Initialize FastAPI app
app = FastAPI()

# 模型加载
# Load model for API
model_path = "./hf_model_save/pytorch_model.bin"
model = Llama(MASTER_CONFIG)
model.load_state_dict(torch.load(model_path))
model.eval()

class InputData(BaseModel):
    idx: list

@app.post("/generate/")
async def generate(model, config=MASTER_CONFIG, max_new_tokens=20):
    # 生成随机数，作为输入数据, 5 行一列，代表输入 5 个字符。 这个地方可以自行替换其他随机数测试。
    # Start with 5x1 zeros; replace with custom input if needed
    idx = torch.zeros(5, 1).long()
    print(idx[:, -config['context_window']:])
    for _ in range(max_new_tokens):
        # 因为推理的时候，依赖后面的 n 个 token，所以滑动窗口要从后往前选择输入数据的倒数几个 token，这个是超过字符数量会对输入进行截断，只选取最后几个token：idx[:, -config['context_window']:]
        # Use last context_window tokens (causal)
        logits = model(idx[:, -config['context_window']:])
        # print(logits.size())
        # 得到模型输出的结果，进行解码，这里 logits[:, -1, :]挺抽象的，实际上第一维度是输入的字符数，第二维度是时间步，第三维度是词表
        # Last time step logits (batch, vocab)
        # 即，对每一步的解码结果，取最后一个时间步的数据，作为输出的数据。解码的过程是第一次解码，输入 5 个 token，第二次解码依赖的是原来 5 个 token 的最后 4 个，加上上一步解码生成的一个，也是 5 个 token，如此循环。
        last_time_step_logits = logits[:, -1, :]
        # print('last_time_step_logits')
        # print(last_time_step_logits.shape)
        # 计算概率分布
        # Softmax for next-token distribution
        p = F.softmax(last_time_step_logits, dim=-1)
        # print('p_shape')
        # print(p.shape)
        # 根据概率分布计算下一个 token，这里使用 torch.multinomial 做的是随机采样
        # Sample next token
        idx_next = torch.multinomial(p, num_samples=1)
        # print('idx_next_shape')
        # print(idx_next.shape)
        # 将新的 idx 通过张量拼接写入到解码序列中
        # Append to sequence
        idx = torch.cat([idx, idx_next], dim=-1)
    # 使用之前定义的解码函数，将ID转换为汉字，我们得到的 5 行 21 列的数据，来源于每一个输入字符作为开始位置，生成 20 个字符。 因为 5 个输入都是 0 ，在词表中编号为 0 的数据是'\n'。
    # Decode IDs to text and return
    print(idx.shape)
    return [decode(x) for x in idx.tolist()]

# 在 Colab 里启动还是挺麻烦的。  建议把所有代码整理一下，在服务器，或者个人电脑里运行
# Running in Colab is cumbersome; better to run on a server or local machine
import nest_asyncio
import uvicorn

nest_asyncio.apply()

# 启动 FastAPI 应用
# Run FastAPI app
uvicorn.run(app, host="0.0.0.0", port=8000)

# 服务部署成功后，可以发送请求测试效果
# After server is up, test with a request
import requests

input_data = {"idx": [[0]]}  # 根据需求提供输入数据
# Provide input as needed
response = requests.post("http://localhost:8000/generate/", json=input_data)
print(response.json())