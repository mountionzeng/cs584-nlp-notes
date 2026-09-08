# 🧠 CS 584 NLP — Module 7: 深度学习架构
## 职业导向笔记 v1.0

> **编译自**：Module 7 原始笔记（CNN / RNN / LSTM / Attention / Transformer / BERT / GPT）
> **目标岗位**：AI 算法工程师 × AI Native 产品经理（双视角）
> **地位**：⭐⭐⭐ 整个课程最核心的模块，现代 AI 产品的技术底座
> **最后更新**：2026-04-15

---

## 🗺️ 知识演进地图

```
局部特征提取          序列建模              全局建模
    CNN          →   RNN/LSTM       →    Transformer
（n-gram模式）    （记忆 + 时序）         （并行 + 注意力）
                                              ↓
                                    预训练大模型时代
                                   BERT（理解）GPT（生成）
```

> **这条线就是近 10 年 NLP 的技术演进主线。** 面试被问"NLP 发展史"，画这张图就够了。

---

## 📌 Chapter 1：CNN in NLP —— 局部模式提取器

### 核心思想

> **文本也有局部模式**，就像图像有边缘和纹理，文本有 n-gram 和短语。

```
图像中的局部模式：猫耳朵、眼睛
文本中的局部模式："not good"、"very happy"、"strongly recommend"
         ↓
同一套卷积思路，适用于两种媒体
```

### 卷积 + 池化 = 特征提取流水线

```
输入句子 → 词嵌入矩阵
    ↓
卷积核（窗口大小 k）滑动扫描
    ↓ 每个位置产生一个特征值
特征图（Feature Map）= [0.98, 0.95, 1.11, 0.90]
    ↓
Max Pooling → 取最大值 1.11
    ↓
该卷积核检测的模式：存在 ✅，最强位置在 "natural language"
```

### TextCNN 架构（文本分类标配）

```
输入 → 词嵌入 → 多尺度卷积（bigram/trigram/4-gram）→ Max Pooling → 拼接 → 全连接 → Softmax
```

**数学公式**：

$$c_i = f\left(\sum_{j=1}^{k}\sum_{d=1}^{D} w_{j,d} \cdot x_{i+j-1,d} + b\right)$$

#### 公式逐层拆解

**一句话理解**：$c_i$ = 卷积核"盖"在第 $i$ 个词开始的窗口上，两个矩阵**逐位相乘再加起来**，过一次激活函数，得到一个分数。分数越大，这个位置越像卷积核要找的模式。

**符号速查**：

| 符号 | 含义 | 举例 |
|------|------|------|
| $c_i$ | 第 $i$ 个位置的输出分数 | 一个数字 |
| $k$ | 窗口看几个词 | k=2 看 bigram，k=3 看 trigram |
| $D$ | 词向量维度 | 300 |
| $w_{j,d}$ | 卷积核在（第 j 行, 第 d 列）的权重 | 可训练参数 |
| $x_{i+j-1,d}$ | 输入矩阵中（第 i+j-1 个词, 第 d 维）的值 | 来自词嵌入 |
| $b$ | 偏置 | 一个数 |
| $f(\cdot)$ | 激活函数 | 通常是 ReLU |

**两层求和的直觉**：$\sum_j$ 遍历窗口里的每个词（行），$\sum_d$ 遍历每个词向量的每个维度（列）—— 合起来就是**卷积核（$k \times D$）和输入窗口（$k \times D$）做点积**。

---

**走一遍完整计算**（k=2，D=3，看第 i=1 个位置）：

```
① 输入矩阵（整句话 5×3）：     ② 卷积核 w（2×3，固定不变）：
  I:    [ 0.1,  0.3, -0.2]       [0.5, -0.3,  0.8]
  love: [ 0.8, -0.1,  0.7]       [0.2,  0.6, -0.4]
  NLP:  [ ...              ]
  so:   [ ...              ]
  much: [ ...              ]

③ 取 i=1 的窗口（前 2 个词）→ 和 w 逐位相乘求和：
   0.1×0.5 + 0.3×(-0.3) + (-0.2)×0.8
 + 0.8×0.2 + (-0.1)×0.6 + 0.7×(-0.4)
 = -0.38

④ 加偏置 b=0.1  →  -0.38 + 0.1 = -0.28
⑤ 过 ReLU      →  max(0, -0.28) = 0  ← c₁
```

---

**卷积核继续往下滑**，每个位置都做同样的计算：

```
i=1: ["I",    "love"] → c₁
i=2: ["love", "NLP"]  → c₂
i=3: ["NLP",  "so"]   → c₃
i=4: ["so",   "much"] → c₄

→ Feature Map = [c₁, c₂, c₃, c₄]
         ↓ Max Pooling（只留最大值）
     1 个分数 = "这个模式在整句话中的最强响应"
```

---

**用 Python 跑一遍公式**（把上面的手算过程写成代码，可以复制到 Jupyter 验证）：

```python
import numpy as np

# ① 输入矩阵：句子 "I love NLP so much"，每个词 3 维
x = np.array([
    [ 0.1,  0.3, -0.2],   # I
    [ 0.8, -0.1,  0.7],   # love
    [ 0.5,  0.4,  0.1],   # NLP
    [-0.3,  0.2,  0.6],   # so
    [ 0.4, -0.5,  0.3],   # much
])

# ② 卷积核：k=2（看 bigram），D=3
w = np.array([
    [0.5, -0.3,  0.8],
    [0.2,  0.6, -0.4],
])
b = 0.1
k, D = w.shape
N = x.shape[0]

def relu(z):
    return np.maximum(0, z)

# ③ 滑动窗口，计算 Feature Map
feature_map = []
for i in range(N - k + 1):          # i = 0, 1, 2, 3（对应 c₁~c₄）
    window = x[i : i + k]            # 取出 k×D 的窗口
    c_i = relu(np.sum(w * window) + b)   # 公式核心：逐位相乘 → 求和 → +b → ReLU
    feature_map.append(c_i)

print("Feature Map:", feature_map)
# 输出: [0.0, 1.11, 0.4, 0.0]（数值依赖输入，这里示意）

# ④ Max Pooling
pooled = max(feature_map)
print("Max Pooled:", pooled)         # → 该卷积核在整句话中的最强响应
```

> **对应关系**：`np.sum(w * window)` 就是公式里的双重求和 $\sum_j \sum_d w_{j,d} \cdot x_{i+j-1,d}$；`relu(... + b)` 就是外面那层 $f(\cdot + b)$。**一行代码 = 一个数学公式**。

在实际工程中会用 `torch.nn.Conv2d` 替代手写循环（见本章后面的 TextCNN 完整代码），原理完全一样，只是 PyTorch 帮你并行化 + 自动求导。

#### Max Pooling 的三大作用

背景：卷积核扫描完整个句子后，得到一串分数，比如 `Feature Map = [0.98, 0.95, 1.11, 0.90]`。Max Pooling 就是**只留下最大值 1.11**，其他全扔掉。这么做有三个好处：

**1️⃣ 降维 —— 把"一串数字"压缩成"一个数字"**

```
原始 Feature Map：[0.98, 0.95, 1.11, 0.90]   ← 4 个数
         ↓ Max Pooling
           1.11                              ← 1 个数
```

> **为什么要压缩？** 后面要接全连接层做分类，如果每个卷积核都留 4 个数、有 100 个卷积核，就是 400 个特征；压缩后只有 100 个，**参数少 4 倍，训练更快、不容易过拟合**。

---

**2️⃣ 位置不变性 —— 不管关键词在句子哪里，都能识别**

假设卷积核检测的是 **"not good"** 这个负面情感模式：

```
句子 A："The movie was not good, really boring."
           ↑ not good 在前面
  Feature Map = [1.11, 0.2, 0.1, 0.3]  → Max = 1.11 ✅

句子 B："I thought it was fine, but actually not good."
                                          ↑ not good 在后面
  Feature Map = [0.2, 0.1, 0.3, 1.11]  → Max = 1.11 ✅
```

> **关键洞察**：两个句子位置不同，但 Max Pooling 的输出**一模一样**。模型只关心"有没有这个模式"，不关心"在哪里出现"—— 这对情感分析、垃圾邮件检测这类任务特别重要（关键词在句子任何位置都该被捕捉）。

---

**3️⃣ 特征选择 —— 只留最强信号，过滤噪音**

```
Feature Map：[0.98, 0.95, 1.11, 0.90]
            弱     弱    强⭐    弱
             ↓
       只保留 1.11（最确信的那个）
```

> **类比**：你问 4 个朋友"这部电影好看吗？"，打分分别是 0.98、0.95、1.11、0.90。Max Pooling 的逻辑是：**"我只听最强烈的那个意见（1.11），其他都是噪音"**。这样模型能聚焦在"卷积核到底有没有捕捉到目标模式"，而不被周围的弱响应干扰。

```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[2,3,4], num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        conv_results = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled = torch.max_pool2d(conv_out, (conv_out.size(2), 1))
            conv_results.append(pooled.squeeze())
        x = torch.cat(conv_results, dim=1)
        x = self.dropout(x)
        return self.fc(x)
```

### CNN vs RNN 选型

**一句话区分**：CNN 像**一群人分头干活**（并行），RNN 像**一个人排队做事**（串行）。

| 对比维度 | CNN | RNN/LSTM |
|---|---|---|
| **能不能同时处理多个词** | ✅ 能 | ❌ 不能 |
| **处理速度** | ✅ 快（GPU 友好） | ❌ 慢 |
| **能记住多远的信息** | ❌ 只能看卷积核窗口内（k=3 就只看 3 个词） | ✅ 理论上可以看到句子开头 |
| **擅长什么任务** | 整句话给 1 个判断（分类） | 每个词给 1 个标签 / 逐词生成 |

---

#### 🔍 每一行到底在说什么？

**1️⃣ 并行计算：为什么 CNN 可以同时算、RNN 必须一个一个来？**

```
CNN：所有卷积窗口彼此独立，可以"同时"计算
  句子："I love NLP so much"
  窗口 1: [I, love]     ┐
  窗口 2: [love, NLP]   ├─ 这 4 个可以同时丢给 GPU 并行算 ✅
  窗口 3: [NLP, so]     │
  窗口 4: [so, much]    ┘

RNN：h_t = f(h_{t-1}, x_t) —— 当前状态依赖上一步
  必须先算 h₁ → 才能算 h₂ → 才能算 h₃ ...
  像串珠子，每颗珠子都要等前一颗穿好 ❌
```

> **这就是为什么 Transformer 干掉了 RNN**：GPU 有几千个核心，能并行的模型可以用满算力，不能并行的只能闲着一个核心转。

---

**2️⃣ 长距离依赖：CNN 的"视野盲区"**

假设卷积核窗口 k=3，看下面这句话：

```
"The movie I watched last weekend with my friends was NOT good"
  ↑                                                    ↑
  主语                                              关键情感词
```

CNN 只能看连续 3 个词，**"movie" 和 "NOT good" 隔太远，同一个卷积核永远看不到它们同时出现**，所以捕捉不到"电影不好"这个完整语义。

RNN 理论上可以：信息从左到右一路传下去，理论上 "movie" 的信息能一直保留到读完 "NOT good"。（**实际上** RNN 也容易遗忘，所以才需要 LSTM 的记忆门来救场。）

---

**3️⃣ 适用任务：为什么一个擅长分类、一个擅长序列标注？**

| 任务类型             | 适合谁                   | 原因                                             |
| ---------------- | --------------------- | ---------------------------------------------- |
| **情感分析 / 垃圾邮件**  | CNN                   | 只要判断整句话是正面还是负面，**位置不重要**，Max Pooling 刚好把位置信息压掉 |
| **NER / POS 标注** | RNN                   | 要给**每个词**打标签（"苹果"是水果还是公司？），需要逐词的上下文理解          |
| **机器翻译 / 文本生成**  | RNN（及后来的 Transformer） | 要按顺序一个词一个词生成，天然是序列任务                           |

> **记忆口诀**：**"整句一个答案 → CNN；每词一个答案 → RNN"**

---

### 🎯 产品经理要掌握到：

✅ **CNN 适合文本分类任务**：情感分析、垃圾邮件检测、内容审核 —— NLP 产品最常见落地场景

✅ **CNN 的局限**：捕捉不了长距离依赖 —— "The man who I met **yesterday** was **not** nice"，CNN 可能漏掉

> **产品面试常问**：
> - "如果要做实时评论审核，你会选什么方案？"（→ 先用 CNN/TextCNN，速度快；复杂内容再用 BERT）

---

### 📐 算法工程师要掌握到：

✅ 能手写 TextCNN（见上方代码）

✅ 理解**参数共享**（Parameter Sharing）：同一卷积核扫描所有位置，大幅减少参数量

> **算法面试常问**：
> - "TextCNN 里为什么要用多个不同大小的卷积核？"（→ 同时捕捉 bigram/trigram/4-gram 模式）

---

## 📌 Chapter 2：RNN/LSTM —— 序列记忆机器

### RNN 核心公式

$$h_t = g(U \cdot h_{t-1} + W \cdot x_t)$$
$$y_t = \text{softmax}(V \cdot h_t)$$

> **用人话说**：当前理解 = 神经网络加工（上一步记忆 + 当前词）

### RNN 的两大死穴

| 问题 | 原因 | 影响 |
|---|---|---|
| **梯度消失** | 链式求导，早期词的梯度越传越小 | 长句子记不住开头 |
| **串行计算** | $h_t$ 依赖 $h_{t-1}$，无法并行 | 训练速度慢 |

### LSTM：给 RNN 装上"选择性记忆"

| 门 | 公式 | 功能 |
|---|---|---|
| **遗忘门** | $f_t = \sigma(U_f h_{t-1} + W_f x_t)$ | 决定忘掉什么 |
| **输入门** | $i_t = \sigma(U_i h_{t-1} + W_i x_t)$ | 决定记住什么 |
| **输出门** | $o_t = \sigma(U_o h_{t-1} + W_o x_t)$ | 决定输出什么 |

**Cell State 更新**：
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
$$h_t = o_t \odot \tanh(c_t)$$

**Cell State（细胞状态）= 长期记忆的高速公路**，梯度可以直接流过，解决梯度消失。

### RNN 四大任务架构

| 架构 | 输入→输出 | 典型应用 |
|---|---|---|
| **序列标注** | $x_i \to y_i$（每词一个标签） | POS 标注、NER |
| **序列分类** | $x_{1:n} \to y$（整句一个标签） | 情感分析、垃圾邮件 |
| **语言建模** | $x_t \to x_{t+1}$（预测下一词） | 文本生成 |
| **编码器-解码器** | $x_{1:n} \to y_{1:m}$ | 机器翻译、摘要 |

### Teacher Forcing

> 训练时用真实目标词作为解码器输入，而非模型自己的预测词。
> **好处**：训练稳定；**坏处**：Exposure Bias（训练推理不一致）

### Bidirectional RNN

$$h_t = [h_t^f ; h_t^b] = h_t^f \oplus h_t^b$$

同时跑前向+后向，拼接隐藏状态，用于需要完整上下文的任务（NER、分类）。

---

### 🎯 产品经理要掌握到：

✅ **NER（命名实体识别）是最直接的 B 端落地场景**：从合同/简历/邮件中提取关键信息

✅ **RNN 的串行性是产品的延迟瓶颈**：LSTM 推理速度比 Transformer 慢，影响实时产品设计

---

### 📐 算法工程师要掌握到：

✅ 能手推 LSTM 前向传播（三门公式 + cell state 更新）

✅ 理解 **BPTT（时间反向传播）**：将 RNN 展开为前馈图

✅ 理解 **Weight Tying**：输入嵌入矩阵 E 和输出层矩阵 $E^T$ 共享参数

> **算法面试必考**：
> - "LSTM 如何解决梯度消失？"（→ Cell State 提供梯度高速公路）
> - "Bidirectional LSTM 什么时候用双向？"

---

## 📌 Chapter 3：Attention 机制 —— 信息动态聚焦

### 解决的核心问题

```
编码器-解码器的瓶颈：整个句子 → 固定向量 → 信息丢失
Attention 的解法：解码器生成每个词时，动态"回头看"编码器所有隐藏状态
```

### Attention 计算四步

| 步骤 | 操作 | 目的 |
|---|---|---|
| **Step 1** | $\text{score}(Q, K_i) = Q \cdot K_i^T$ | 计算相似度 |
| **Step 2** | $\text{scaled} = \text{score} / \sqrt{d_k}$ | 缩放，防止 softmax 饱和 |
| **Step 3** | $\alpha_i = \text{softmax}(\text{scaled})$ | 归一化成概率权重 |
| **Step 4** | $\text{output} = \sum \alpha_i V_i$ | 加权求和 |

**完整公式**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Q/K/V 直觉

| 角色 | 含义 | 图书馆类比 |
|---|---|---|
| **Query（Q）** | 我想要什么 | 查询关键词 |
| **Key（K）** | 我有什么信息的标签 | 书名/索引 |
| **Value（V）** | 实际信息内容 | 书的内容 |

### Self-Attention vs Cross-Attention

| | Self-Attention | Cross-Attention |
|---|---|---|
| **Q 来源** | 输入序列本身 | 解码器当前状态 |
| **K/V 来源** | 输入序列本身 | 编码器所有隐藏状态 |
| **用途** | 理解句子内部关系 | 解码时参考输入 |

---

### 🎯 产品经理要掌握到：

✅ **Attention = 可解释性的接口**：可视化模型在预测时"看了"哪些词，是向用户解释决策的工具

✅ **为什么要除以 $\sqrt{d_k}$？**：维度越高点积越大，softmax 变 one-hot，梯度消失

---

## 📌 Chapter 4：Transformer —— 现代 NLP 的基础设施

### 核心创新：抛弃 RNN，只用 Attention

| RNN 的问题 | Transformer 的解法 |
|---|---|
| 串行处理，无法并行 | **完全并行**，所有词同时处理 |
| 长距离依赖困难 | **直接建模**任意两词的关系 |
| 梯度消失 | 残差连接保证梯度流动 |

### Transformer 架构

```
输入 → 词嵌入 + 位置编码
         ↓
编码器（×6层）：
  └─ Multi-Head Self-Attention
  └─ Add & LayerNorm
  └─ Feed-Forward Network
  └─ Add & LayerNorm
         ↓ 上下文表示
解码器（×6层）：
  └─ Masked Multi-Head Self-Attention
  └─ Cross-Attention ← 从编码器拿信息
  └─ Feed-Forward Network
         ↓
输出层 → Softmax → 预测
```

### 四大关键组件

#### 1. Multi-Head Attention（多头注意力）

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$

**为什么多头？** → 不同头从不同角度理解句子：语法/语义/位置关系

#### 2. Positional Encoding（位置编码）

$$PE(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

**最终输入 = 词嵌入 + 位置编码**（词说"我是什么"，位置说"我在哪里"）

#### 3. FFN（前馈网络）

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

**类比**：开完会（自注意力，词间交流）→ 每个人独立写总结（FFN，对每词单独加工）

#### 4. 残差连接 + 层归一化

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**作用**：缓解梯度消失，允许堆叠很深（6层/12层/96层）

---

### 🎯 产品经理要掌握到：

✅ **Transformer = 现代 AI 产品的基础设施**，GPT/BERT/Claude 全部基于此

✅ **并行训练 = 可以用更多 GPU = 更大模型 = 更强产品能力**（Transformer 颠覆 RNN 的商业逻辑）

✅ **编码器 vs 解码器 = 理解 vs 生成**，决定用哪种预训练模型：

```
理解任务（分类/NER/QA）→ 编码器模型（BERT）
生成任务（写作/对话/翻译）→ 解码器模型（GPT）
两者都有（摘要/翻译）→ 完整编解码器（T5/BART）
```

> **产品面试必考**：
> - "Transformer 和 LSTM 最大的区别是什么？"（→ 并行 + 全局注意力）
> - "为什么现在 AI 产品都用 Transformer？"

---

### 📐 算法工程师要掌握到：

✅ 理解 **Masked Self-Attention**（解码器防止看到未来词）

✅ 理解 Transformer 时间复杂度：$O(n^2 \cdot d)$（长文本是瓶颈）

✅ 知道 **KV Cache**（推理加速核心技巧）

> **算法面试必考**：
> - "LayerNorm 和 BatchNorm 的区别？为什么 Transformer 用 LayerNorm？"
> - "Transformer 的时间复杂度？长文本时的瓶颈是什么？"

---

## 📌 Chapter 5：BERT vs GPT —— 预训练时代的两条路线

### 预训练范式

```
传统方法：每个任务从头训练 → 需要大量标注数据 + 训练时间
                ↓
预训练+微调：
  Step 1 预训练：海量无标注文本 → 学通用语言知识
  Step 2 微调：  少量任务数据 → 适配具体任务
```

### BERT（双向编码器）

**架构**：只用 Transformer 编码器 | **方向**：双向

**预训练任务 1：MLM（掩码语言模型）**
```
原句："The cat sat on the mat"
掩码："The [MASK] sat on the mat"  → 预测：cat
策略：80% MASK，10% 随机词，10% 保持不变
```

**预训练任务 2：NSP（下一句预测）**
```
句A + 句B → IsNext / NotNext
```

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer("The cat sat on the mat", return_tensors='pt')
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]
```

### GPT（单向解码器）

**架构**：只用 Transformer 解码器 | **方向**：单向（从左到右）

**预训练任务：自回归语言建模**
$$P(x_1,...,x_T) = \prod_{t=1}^T P(x_t | x_1,...,x_{t-1})$$

**生成策略对比**：

| 策略 | 方法 | 适用场景 |
|---|---|---|
| 贪婪搜索 | 每步选最高概率 | 快，但质量一般 |
| Beam Search | 保留 k 个候选 | 翻译/摘要 |
| Top-k 采样 | 从前 k 个词随机采样 | 创意生成 |
| Top-p 采样 | 从累积概率 p 内采样 | 自然对话 |

### BERT vs GPT 终极对比

| 维度 | BERT | GPT |
|---|---|---|
| **架构** | Transformer 编码器 | Transformer 解码器 |
| **方向** | 双向（看全文） | 单向（只看历史） |
| **预训练任务** | MLM + NSP | 自回归语言模型 |
| **优势** | 深度理解上下文 | 自然流畅生成 |
| **代表应用** | 搜索、分类、NER、QA | ChatGPT、代码补全 |

---

### 🎯 产品经理要掌握到：

✅ **这是所有现代 AI 产品选型的核心框架**：

```
需要"理解"？（分类/抽取/排序）→ BERT 系列
需要"生成"？（写作/对话/翻译）→ GPT 系列
两者都要？（RAG/Agent）         → 两者结合
```

✅ **微调（Fine-tuning）是产品落地的关键一步**：预训练模型是通用的，微调让它专注于你的业务

> **产品面试必考**：
> - "BERT 和 GPT 的核心区别？你们用的是哪个？为什么？"
> - "什么时候用微调，什么时候用提示工程（Prompting）？"

---

### 📐 算法工程师要掌握到：

✅ 能用 HuggingFace 实现 BERT 微调（情感分类示例见原始笔记）

✅ 理解微调的不同方式：
- **特征提取**：冻结 BERT，只训练任务层（快）
- **全量微调**：更新所有参数（效果好）
- **LoRA/Adapter**：参数高效微调，工业界主流

---

## ⚡ 踩坑 & 经验

1. **Transformer ≠ LLM**：Transformer 是架构，BERT/GPT 是具体模型，面试不要混用
2. **BERT 不能直接生成文本**：它是编码器，没有生成能力
3. **"Attention is All You Need" 是最值得读的论文之一**
4. **多头注意力的头数是超参数**：头越多越丰富，但计算量更大
5. **位置编码不一定用 sin/cos**：现代模型（LLaMA）用 RoPE（旋转位置编码）

---

## 💼 双视角面试题库

| 面试题 | 算法回答要点 | 产品回答要点 |
|---|---|---|
| "Transformer 和 LSTM 的区别？" | 并行 vs 串行；全局 Attention vs 顺序记忆 | Transformer 训练更快可以更大，是现代 AI 产品的基础 |
| "BERT 和 GPT 怎么选？" | 理解任务用编码器，生成任务用解码器 | 理解业务需求：分类/抽取 vs 生成 |
| "多头注意力的作用？" | 不同头关注不同类型的关系 | 类比：多个专家同时分析同一段话，各有侧重 |
| "为什么预训练+微调效果好？" | 预训练学通用知识，微调快速适配 | 节省标注成本，提升效果，缩短项目周期 |
| "Self-Attention 的时间复杂度？" | $O(n^2 \cdot d)$，长文本是瓶颈 | 文档越长处理越慢越贵，影响产品的文档长度限制 |

---

## 🔖 核心公式速查卡

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$

$$\text{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$$

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

$$h_t^{\text{LSTM}} = o_t \odot \tanh(c_t)$$

---

**笔记版本**：v1.0 | **覆盖范围**：Module 7 (Deep Learning Architectures)
