# Module 6: 神经网络与序列标注

> **课程**: CS584 Natural Language Processing  
> **阅读**: SLP Chapter 7, 8  
> **知识体系**: [[Module_05_Vector_Semantics]] ← **本章** → [[Module_07_Deep_Learning]]  
> **核心问题**: 如何用循环神经网络处理变长序列？如何为序列中每个元素自动标注语法/语义标签？

---

## 推荐学习资源

| 类型 | 资源 | 链接 | 说明 |
|------|------|------|------|
| 教材 | SLP Chapter 7: Neural Networks (Jurafsky & Martin) | https://web.stanford.edu/~jurafsky/slp3/7.pdf | 本章主要参考，免费PDF |
| 教材 | SLP Chapter 8: Sequence Labeling (Jurafsky & Martin) | https://web.stanford.edu/~jurafsky/slp3/8.pdf | 序列标注专章 |
| 视频 | Stanford CS224N Lecture 5: RNN & LSTM | https://www.youtube.com/watch?v=0LixFSa7yts | 斯坦福NLP课程，含LSTM可视化 |
| 视频 | 3Blue1Brown: Neural Networks Series | https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi | 神经网络最佳入门动画 |
| 博客 | Understanding LSTM Networks (Colah) | https://colah.github.io/posts/2015-08-Understanding-LSTMs/ | LSTM 最经典的可视化讲解 |
| 论文 | BiLSTM-CRF for NER (Lample et al. 2016) | https://arxiv.org/abs/1603.01360 | NER 奠基论文 |
| 工具 | spaCy NER Documentation | https://spacy.io/usage/linguistic-features#named-entities | 工业级NER使用指南 |
| 工具 | HuggingFace NER Pipeline | https://huggingface.co/docs/transformers/task_summary#token-classification | BERT-based NER一行代码 |
| 数据集 | CoNLL-2003 NER Dataset | https://www.clips.uantwerpen.be/conll2003/ner/ | 标准NER benchmark |
| 代码 | PyTorch LSTM Tutorial | https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html | 官方序列标注教程 |

---

## 目录

1. [为什么需要神经网络](#1-为什么需要神经网络)
2. [神经元与前馈网络（FFNN）](#2-前馈神经网络)
3. [循环神经网络（RNN）](#3-循环神经网络)
4. [LSTM：长短期记忆](#4-lstm)
5. [GRU：轻量化替代](#5-gru)
6. [序列标注任务](#6-序列标注)
7. [双向 LSTM（BiLSTM）](#7-双向lstm)
8. [条件随机场（CRF）](#8-条件随机场)
9. [BiLSTM-CRF：工业标准架构](#9-bilstm-crf)
10. [评估指标](#10-评估指标)
11. [三视角职业分析](#11-职业分析)
12. [面试高频题](#12-面试题)

---

## 1. 为什么需要神经网络

### 1.1 传统机器学习的瓶颈

```
传统方法（以垃圾邮件过滤为例）：
  步骤 1: 人工设计特征 → "含'中奖'词", "含'免费'词", "大写字母比例" ...
  步骤 2: 用逻辑回归分类
  问题: 特征工程耗时、依赖专业知识、泛化能力有限

神经网络方法：
  步骤 1: 输入原始文本
  步骤 2: 自动学习有用特征（端到端）
  优势: 不需要人工设计特征！
```

**NLP 的额外挑战**：
- 输入是**变长**序列（句子长度不固定）
- 词语之间有**长距离依赖**（"The cat that sat... **was** hungry"）
- 语言有**歧义性**（"bank" = 银行 or 河岸）

### 1.2 本章技术演进路线

```
FFNN（固定输入）
    ↓ 无法处理变长序列
RNN（循环状态，可变长）
    ↓ 梯度消失，长距离依赖差
LSTM（门控机制，长记忆）
    ↓ 单向上下文不够
BiLSTM（双向上下文）
    ↓ 标签之间相互独立
BiLSTM + CRF（标签依赖，全局最优）
```

---

## 2. 前馈神经网络

### 2.1 单个神经元

$$y = f\!\left(\sum_{i=1}^n w_i x_i + b\right) = f(\mathbf{w}^\top \mathbf{x} + b)$$

```
    x₁ ──w₁──┐
    x₂ ──w₂──┤
    x₃ ──w₃──┼──→ Σ(加权求和) ──→ f(激活) ──→ y
     ⋮        │                      ↑
    xₙ ──wₙ──┘                      b（偏置）
```

**直观例子**：预测是否喜欢一部电影

```
输入特征：
  x₁ = 动作场面评分 = 8
  x₂ = 剧情评分     = 6
  x₃ = 演员评分     = 7

权重（学习得到）：
  w₁ = 0.5, w₂ = 0.3, w₃ = 0.2, b = -5

计算：
  z = 0.5×8 + 0.3×6 + 0.2×7 - 5 = 4 + 1.8 + 1.4 - 5 = 2.2
  y = sigmoid(2.2) = 0.90 → 90% 概率会喜欢
```

### 2.2 激活函数对比

| 激活函数 | 公式 | 输出范围 | 优点 | 缺点 | 典型用途 |
|----------|------|----------|------|------|----------|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | (0, 1) | 可解释为概率 | 梯度消失严重 | 二分类输出层 |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | (-1, 1) | 以零为中心 | 仍有梯度消失 | RNN/LSTM 门控 |
| **ReLU** | $\max(0, x)$ | [0, +∞) | 计算简单，缓解梯度消失 | 死亡ReLU | 隐藏层首选 |
| **Softmax** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | (0,1)，和为1 | 概率分布 | - | 多分类输出层 |

**⚠️ 关键理解**：若无激活函数（线性激活），无论叠多少层，整个网络等价于一个线性变换。激活函数引入**非线性**，是深度学习的核心。

### 2.3 前向传播

$$\mathbf{h}^{(1)} = f^{(1)}(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$$
$$\mathbf{h}^{(2)} = f^{(2)}(\mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$
$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{W}^{(3)}\mathbf{h}^{(2)} + \mathbf{b}^{(3)})$$

**维度追踪示例**（情感分类）：

```
输入:      [0.2, -0.5, 0.8, 0.1]            shape: (4,)
    ↓ W¹(5×4) + b¹(5,) → ReLU
隐藏层1:   [0.5, 0.3, 0.0, 0.7, 0.1]        shape: (5,)
    ↓ W²(3×5) + b²(3,) → ReLU
隐藏层2:   [0.8, 0.0, 0.4]                   shape: (3,)
    ↓ W³(2×3) + b³(2,) → Softmax
输出:      [0.85, 0.15]                       shape: (2,)
           正面  负面
```

### 2.4 反向传播（链式法则）

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**参数更新**：

$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

其中 $\eta$ 是学习率，$L$ 是损失函数（分类任务用交叉熵）。

---

## 3. 循环神经网络

### 3.1 名称解析："循环"的含义

> **RNN = Recurrent Neural Network**  
> recurrent = re-（再次）+ current（流动）= 循环、反复出现

```
前馈网络（Feedforward NN）：
信息只向前流动：输入 → 隐藏层 → 输出

循环网络（Recurrent NN）：
隐藏层输出会循环回来，参与下一步计算：
输入 → 隐藏层 ⟲ → 输出
       ↑______|

核心特点：同一组参数在每个时间步重复使用（参数共享）
```

### 3.2 RNN 的结构与展开

**折叠图**（抽象）：

```
         h
         ↑
x ──→ [RNN] ──→ y
         ↑
        (h 循环)
```

**展开图**（时间维度）：

```
时间步:   t=1          t=2          t=3
输入:     "I"          "love"       "NLP"
           ↓            ↓            ↓
h₀ ──→ [RNN] ──h₁──→ [RNN] ──h₂──→ [RNN] ──h₃
           ↓            ↓            ↓
输出:     y₁           y₂           y₃

关键：三个 [RNN] 是同一个模块，共享 Wₓ、W_h 权重
```

**数学公式**：

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
$$y_t = W_y h_t + b_y$$

**参数解读**：

| 参数 | 含义 | 特点 |
|------|------|------|
| $h_t$ | 当前时刻隐藏状态（"记忆"）| 传递给下一时刻 |
| $x_t$ | 当前时刻输入 | 每步不同 |
| $W_h$ | 隐藏→隐藏权重 | 所有时间步共享 |
| $W_x$ | 输入→隐藏权重 | 所有时间步共享 |

### 3.3 完整追踪示例：语言模型

**任务**：给定"I love"，预测下一个词

```
初始状态: h₀ = [0, 0, 0]（零向量）

t=1: 输入 "I" 的词向量 x₁
  h₁ = tanh(Wₕ·[0,0,0] + Wₓ·x₁ + b)
     = tanh(Wₓ·x₁ + b)
  h₁ 代表："知道第一个词是 I"

t=2: 输入 "love" 的词向量 x₂
  h₂ = tanh(Wₕ·h₁ + Wₓ·x₂ + b)
  h₂ 代表："知道见过 I，当前词是 love"
  预测下一词: softmax(Wᵧ·h₂) 
           → P("you") = 0.35, P("NLP") = 0.28, P("it") = 0.15 ...
```

### 3.4 RNN 的两大问题

#### 问题一：梯度消失（Vanishing Gradient）

**现象**：长序列中，早期信息逐渐"消失"。

```
句子: "The cat, which was sitting on the mat, was hungry."

处理最后的 "was hungry" 时，需要记住主语是 "cat"
但梯度在反向传播时逐渐衰减：

∂h_t/∂h₀ = ∏ᵢ (∂hᵢ/∂hᵢ₋₁) = W_h^t × (对角矩阵)

若 |W_h| < 1 → 连乘后指数级衰减 → "忘记"早期信息
```

#### 问题二：梯度爆炸（Exploding Gradient）

若 $|W_h| > 1$ → 连乘后指数级增大 → 参数更新不稳定

**解决方案**：梯度裁剪（Gradient Clipping）

```python
# 梯度范数超过阈值时，等比缩放
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

---

## 4. LSTM

### 4.1 核心思想：门控机制

LSTM（Long Short-Term Memory）引入**细胞状态**（Cell State）$C_t$ 作为长期记忆，通过三个"门"选择性地读写信息。

**类比**：细胞状态像一条传送带，门控决定信息能不能"上传送带"或"被取走"。

### 4.2 三个门的直觉理解

**情景**：LSTM 正在阅读 "The cat, which ate the fish, was hungry"

```
遗忘门（Forget Gate）：读到新主语时，"忘掉"之前的无关信息
                        读到 "was" 时，"cat 是主语"的记忆应当保留

输入门（Input Gate）：读到 "hungry" 时，将"饥饿状态"存入记忆
                       读到介词 "which" 时，不需要更新核心记忆

输出门（Output Gate）：预测下一个词时，输出当前最相关的记忆
                        如果在问"主语是什么"，输出 "cat"
```

### 4.3 完整数学公式

**遗忘门**（决定丢弃 $C_{t-1}$ 中哪些信息）：

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \in (0,1)^n$$

**输入门**（决定存储哪些新信息）：

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{（候选新记忆）}$$

**更新细胞状态**（遗忘旧 + 记住新）：

$$C_t = \underbrace{f_t \odot C_{t-1}}_{\text{遗忘部分旧记忆}} + \underbrace{i_t \odot \tilde{C}_t}_{\text{加入部分新信息}}$$

**输出门**（决定输出哪些信息）：

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

其中 $\odot$ 是逐元素乘法，$\sigma$ 是 sigmoid（输出 0~1 作为"门开度"）。

### 4.4 门控机制的数值追踪示例

**场景**：处理 "The cat sat on the mat"，追踪 LSTM 如何记住主语

```
假设简化版 LSTM（1维），初始 C₀ = h₀ = 0

t=1: 读入 "The"（非关键词）
  f₁ ≈ 0.9  → 保留 90% 旧记忆（旧记忆本来就是 0）
  i₁ ≈ 0.1  → 只写入 10% 新信息
  C₁ = 0.9×0 + 0.1×0.05 ≈ 0.005  （几乎没变化）

t=2: 读入 "cat"（名词，潜在主语！）
  f₂ ≈ 0.8  → 保留 80%
  i₂ ≈ 0.8  → 大量写入新信息
  C̃₂ ≈ 0.9  → 候选新记忆高（cat 是重要词）
  C₂ = 0.8×0.005 + 0.8×0.9 ≈ 0.72  （细胞状态显著增大！）

t=5: 读入 "mat"（新名词，但不是主语）
  f₅ ≈ 0.9  → 几乎不遗忘（cat 作为主语仍重要）
  C₅ ≈ 0.65  （主语信息保留）

→ 即使间隔了 "sat on the"，cat 的信息仍被保留
```

### 4.5 LSTM vs RNN vs GRU 对比

| 特性 | RNN | LSTM | GRU |
|------|-----|------|-----|
| **参数数量** | 最少 | 最多（4×RNN）| 中等（3×RNN）|
| **训练速度** | 最快 | 最慢 | 中等 |
| **长距离依赖** | ❌ 差 | ✅ 强 | ✅ 较强 |
| **梯度消失** | ❌ 严重 | ✅ 显著缓解 | ✅ 显著缓解 |
| **适用场景** | 短序列 | 长序列/复杂任务 | 长序列/资源有限 |
| **核心结构** | 1个状态 | 细胞状态+隐藏状态 | 1个状态 |

**选择建议**：
- 序列 < 20 词：RNN 够用
- 长序列、NER/MT：LSTM（准确率优先）
- 资源受限、移动端：GRU（速度/准确率平衡）

---

## 5. GRU

### 5.1 GRU 的简化设计

GRU（Gated Recurrent Unit）将 LSTM 的三个门简化为两个，去掉独立的细胞状态：

**重置门（Reset Gate）**：决定忽略多少过去信息

$$r_t = \sigma(W_r [h_{t-1}, x_t])$$

**更新门（Update Gate）**：决定保留多少旧状态 vs 写入多少新候选状态

$$z_t = \sigma(W_z [h_{t-1}, x_t])$$

**候选隐藏状态**：

$$\tilde{h}_t = \tanh(W[r_t \odot h_{t-1}, x_t])$$

**最终隐藏状态**（由更新门在旧状态和新候选之间插值）：

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**直觉**：当 $z_t \to 0$ 时，完全保留旧状态（类似 LSTM 遗忘门开=1）；当 $z_t \to 1$ 时，完全使用新候选状态。

---

## 6. 序列标注

### 6.1 任务定义

**序列标注**：为输入序列 $x_1, x_2, ..., x_n$ 中的每个元素分配一个标签 $y_1, y_2, ..., y_n$。

| 任务 | 输入示例 | 输出示例 |
|------|----------|----------|
| **词性标注 (POS)** | "I love NLP" | "PRP VBP NN" |
| **命名实体识别 (NER)** | "Apple CEO Tim Cook" | "B-ORG O B-PER I-PER" |
| **中文分词** | "我爱自然语言处理" | "S S B M M E B E" |
| **语义角色标注 (SRL)** | "Cats eat fish" | "A0 V A1" |

### 6.2 词性标注（POS Tagging）

**常见 Penn Treebank 词性标签**：

| 标签 | 词性 | 例子 |
|------|------|------|
| NN | 普通名词单数 | cat, NLP |
| NNS | 普通名词复数 | cats |
| VBD | 动词过去式 | sat, ran |
| VBP | 动词现在式非第三人称 | love, run |
| JJ | 形容词 | big, fast |
| RB | 副词 | quickly, very |
| DT | 限定词 | the, a |
| IN | 介词/从属连词 | on, in, that |
| PRP | 人称代词 | I, he, she |

**完整标注示例**：

```
句子: "The quick brown fox jumps over the lazy dog"
标注: "DT  JJ    JJ    NN  VBZ   IN   DT  JJ   NN"
```

**应用**：语法分析、信息抽取、机器翻译前处理

### 6.3 命名实体识别（NER）

**BIO 标注体系（最常用）**：

| 标签 | 含义 | 规则 |
|------|------|------|
| **B-TYPE** | 实体的**开始**（Begin） | 每个实体第一个 token |
| **I-TYPE** | 实体的**内部**（Inside） | 实体后续 tokens，必须跟在 B/I 后 |
| **O** | **非实体**（Outside） | 不属于任何实体 |

**完整标注示例**：

```
输入: "Apple CEO Tim Cook visited Microsoft in Seattle"

Apple      → B-ORG    (组织实体开始)
CEO        → O        (非实体)
Tim        → B-PER    (人名开始)
Cook       → I-PER    (人名内部，紧接 B-PER)
visited    → O
Microsoft  → B-ORG    (新的组织实体)
in         → O
Seattle    → B-LOC    (地点实体)
```

**常见实体类型（CoNLL-2003）**：

| 类型 | 全称 | 例子 |
|------|------|------|
| PER | Person（人名）| Tim Cook, 马云 |
| LOC | Location（地点）| Seattle, 北京 |
| ORG | Organization（组织）| Apple, UN |
| MISC | Miscellaneous（其他）| 节日名、语言名 |

### 6.4 BIO 标注的约束

```
合法转换：
O → O
O → B-X      （开始新实体）
B-X → I-X    （实体内部延续）
B-X → B-Y    （实体结束，新实体立刻开始）
I-X → I-X    （实体内部延续）
I-X → O      （实体结束）

非法转换（CRF 需要学习排除）：
O → I-X      ❌ I 不能紧跟 O
B-X → I-Y    ❌ B-PER 后不能跟 I-ORG
```

### 6.5 中文分词：BMES 标注

```
标注方案：
B = 词的开始 (Begin)
M = 词的中间 (Middle)
E = 词的结束 (End)
S = 单字词 (Single)

示例：
  我  爱  自  然  语  言  处  理
  S   S   B   M   M   E   B   E

分词结果: 我 / 爱 / 自然语言 / 处理
```

---

## 7. 双向LSTM

### 7.1 单向 LSTM 的局限

```
句子: "I went to the bank by the river"

单向 LSTM（从左到右）处理 "bank" 时：
已看到: "I went to the"
未看到: "by the river"
结果: 不知道 "bank" 是银行还是河岸 ❌

双向 LSTM 处理 "bank" 时：
前向: 看到 "I went to the"
后向: 看到 "by the river"
结果: 结合两个方向 → 是河岸（river bank）✅
```

### 7.2 BiLSTM 结构

```
前向 LSTM:   →h₁  →h₂  →h₃  →h₄
输入:         "I"  "love" "NLP"  "."
后向 LSTM:   ←h₁  ←h₂  ←h₃  ←h₄

每个位置的最终表示（拼接前后向）：
h₁ = [→h₁ ; ←h₁]   （包含 "I" 的两个方向上下文）
h₂ = [→h₂ ; ←h₂]   （包含 "love" 的两个方向上下文）
h₃ = [→h₃ ; ←h₃]   （包含 "NLP" 的两个方向上下文）
```

**数学表示**：

$$\overrightarrow{h}_t = \text{LSTM}_{\text{fwd}}(\overrightarrow{h}_{t-1}, x_t)$$
$$\overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(\overleftarrow{h}_{t+1}, x_t)$$
$$h_t = [\overrightarrow{h}_t \,;\, \overleftarrow{h}_t] \quad \in \mathbb{R}^{2d}$$

**效果提升**：BiLSTM 在 NER 上 F1 比单向 LSTM 通常高 1-3 个百分点。

---

## 8. 条件随机场

### 8.1 LSTM 单独做序列标注的问题

```
问题：LSTM 独立预测每个位置的标签，不考虑标签间的约束

示例：
输入: "Tim Cook"
LSTM 可能预测:
  Tim  → I-PER   ❌（第一个词不能是 I-，必须是 B-）
  Cook → B-PER

正确答案:
  Tim  → B-PER   ✓
  Cook → I-PER   ✓
```

**根本原因**：LSTM 的 softmax 在每个位置**独立**预测，不知道相邻位置的标签约束。

### 8.2 CRF 的核心思想

CRF（Conditional Random Field）学习**标签转移概率**，在**全局**范围内找最优标签序列：

$$P(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\left(\sum_{t=1}^T \left[\underbrace{s(y_t, x_t)}_{\text{发射分数(来自BiLSTM)}} + \underbrace{T(y_{t-1}, y_t)}_{\text{转移分数(CRF学习)}}\right]\right)$$

**转移矩阵（示例，5 个标签）**：

```
转移矩阵 T[y_prev][y_next]（值越大越合法）：

         O    B-PER  I-PER  B-ORG  I-ORG
O      [ 5.2,  3.1,  -∞,    2.8,  -∞   ]
B-PER  [ 2.1,  0.5,  4.8,   0.2,  -∞   ]
I-PER  [ 1.9,  0.3,  4.5,   0.1,  -∞   ]
B-ORG  [ 2.3,  0.4,  -∞,    0.6,  4.9  ]
I-ORG  [ 1.8,  0.2,  -∞,    0.4,  4.7  ]

解读：
- T[O][I-PER] = -∞  → O 后面不能跟 I-PER（学习到的约束）
- T[B-PER][I-PER] = 4.8  → B-PER 后面跟 I-PER 非常合理
```

**解码**：用 Viterbi 算法找全局最优路径（$O(n \cdot |\mathcal{Y}|^2)$）。

---

## 9. BiLSTM-CRF

### 9.1 完整架构

```
输入: ["Apple", "CEO", "Tim", "Cook"]
    ↓ 词嵌入（可加字符级嵌入）
词向量: [v₁, v₂, v₃, v₄]     shape: (4, embed_dim)
    ↓ BiLSTM
上下文表示: [h₁, h₂, h₃, h₄]  shape: (4, 2×hidden)
    ↓ 全连接层（发射分数矩阵）
发射分数: shape: (4, num_tags)   每个词对每个标签的"原始得分"
    ↓ CRF 层（加入转移分数，Viterbi 解码）
最优标签序列: [B-ORG, O, B-PER, I-PER]
```

### 9.2 完整 PyTorch 实现

```python
import torch
import torch.nn as nn

class BiLSTM_CRF(nn.Module):
    """
    BiLSTM-CRF 序列标注器（工业标准 NER 架构）
    参考: Lample et al. (2016) "Neural Architectures for NER"
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.num_tags = num_tags
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 双向 LSTM
        self.bilstm = nn.LSTM(
            embed_dim, 
            hidden_dim // 2,      # 双向各用 hidden/2，拼接后 = hidden
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=0.3
        )
        
        # 发射分数：BiLSTM 输出 → 标签得分
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF 转移矩阵（可学习参数）
        # transitions[i][j] = 从标签 j 转移到标签 i 的得分
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # 约束：START 标签后只能跟某些标签，END 前只能有某些标签
        # （可选，用于更严格的约束）
    
    def get_emissions(self, x):
        """获取 BiLSTM 的发射分数矩阵"""
        embeds = self.embedding(x)             # (batch, seq, embed)
        lstm_out, _ = self.bilstm(embeds)      # (batch, seq, hidden)
        emissions = self.hidden2tag(lstm_out)  # (batch, seq, num_tags)
        return emissions
    
    def viterbi_decode(self, emissions, mask=None):
        """
        Viterbi 算法：找全局最优标签序列
        emissions: (seq_len, num_tags)
        """
        seq_len, num_tags = emissions.shape
        
        # dp[t][j] = 到时间步 t，以标签 j 结尾的最大得分
        dp = torch.full((num_tags,), -1e9)
        dp = emissions[0]  # 初始化为 t=0 的发射分数
        
        backpointers = []
        
        for t in range(1, seq_len):
            # dp_expand: (num_tags, num_tags) — 上一步各标签的得分
            dp_expand = dp.unsqueeze(1).expand(num_tags, num_tags)
            
            # 转移分数 + 上一步得分
            scores = dp_expand + self.transitions  # (num_tags, num_tags)
            
            # 对每个当前标签，找最优前驱
            best_scores, best_tags = scores.max(dim=0)  # (num_tags,)
            backpointers.append(best_tags)
            
            # 加上当前步的发射分数
            dp = best_scores + emissions[t]
        
        # 回溯最优路径
        best_last_tag = dp.argmax().item()
        best_path = [best_last_tag]
        
        for bptrs in reversed(backpointers):
            best_last_tag = bptrs[best_last_tag].item()
            best_path.append(best_last_tag)
        
        best_path.reverse()
        return best_path
    
    def neg_log_likelihood(self, x, tags):
        """
        训练损失：-log P(gold_tags | x)
        使用前向算法计算配分函数 Z(x)
        """
        emissions = self.get_emissions(x)  # (batch, seq, num_tags)
        # ... CRF 前向算法（计算 log Z）
        # loss = log_Z - gold_score
        # （完整实现参见 pytorch-crf 库）
        pass
    
    def forward(self, x):
        """推理：返回最优标签序列"""
        emissions = self.get_emissions(x)  # (batch, seq, num_tags)
        batch_size = emissions.shape[0]
        
        results = []
        for i in range(batch_size):
            tags = self.viterbi_decode(emissions[i])
            results.append(tags)
        return results


# ============ 使用示例 ============

# 推荐使用 torchcrf 库（专业CRF实现）
# pip install pytorch-crf
from torchcrf import CRF

class NERModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=100, hidden_dim=256, num_tags=9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim//2, 
                               bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            # 训练模式：返回负对数似然
            return -self.crf(emissions, tags, mask=mask)
        else:
            # 推理模式：返回最优标签序列
            return self.crf.decode(emissions, mask=mask)

# 初始化和训练
model = NERModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环（简化版）
for batch in train_dataloader:
    words, tags, mask = batch
    loss = model(words, tags, mask)   # 返回负对数似然
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
    optimizer.step()
    optimizer.zero_grad()
```

### 9.3 维度全程追踪

```
输入词 ID:  [5, 234, 89, 12]          shape: (batch=1, seq=4)
    ↓ Embedding(10000, 100)
词向量:     shape: (1, 4, 100)
    ↓ BiLSTM(100→128, bidirectional)
BiLSTM输出: shape: (1, 4, 256)        (128×2，前后向拼接)
    ↓ Linear(256, 9)
发射分数:   shape: (1, 4, 9)          每个词对9个标签的得分
    ↓ CRF Viterbi
标签序列:   [0, 2, 3, 4]              → [B-ORG, O, B-PER, I-PER]
```

---

## 10. 评估指标

### 10.1 序列标注的评估单位

NER 评估以**实体**为单位（不是单个 token），完整的实体 span 都预测正确才算 TP：

```
金标准: [B-ORG(Apple), O(CEO), B-PER(Tim), I-PER(Cook)]
预测:   [B-ORG(Apple), O(CEO), B-PER(Tim), O(Cook)]

分析：
  Apple 实体 (span: [0,0]):  ✓ 预测正确 → TP
  Tim Cook 实体 (span: [2,3]): ✗ Cook 预测错了 → FN（漏检）
  
  如果额外预测了一个错误实体 → FP（虚检）
```

### 10.2 P / R / F1（实体级别）

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{（预测对的实体 / 预测的全部实体）}$$

$$\text{Recall} = \frac{TP}{TP + FN} \quad \text{（预测对的实体 / 真实的全部实体）}$$

$$F_1 = 2 \times \frac{P \times R}{P + R}$$

**完整计算示例**：

```
测试集包含：
  10 个 PER 实体 + 8 个 ORG 实体 + 5 个 LOC 实体 = 23 个真实实体

模型预测：
  预测了 20 个实体
  其中 17 个与金标准完全匹配

P = 17/20 = 0.85   (预测的 20 个中 17 个对)
R = 17/23 = 0.74   (真实的 23 个中找到了 17 个)
F1 = 2 × 0.85 × 0.74 / (0.85 + 0.74) = 0.79
```

### 10.3 当前 SOTA 水平

| 数据集 | 模型 | F1 |
|--------|------|-----|
| CoNLL-2003 (英语 NER) | BiLSTM-CRF | ~91% |
| CoNLL-2003 (英语 NER) | BERT-large-CRF | ~93% |
| CoNLL-2003 (英语 NER) | RoBERTa + CRF | ~94% |

**人类水平**：~97% F1（注意：标注者之间也存在分歧）

---

## 11. 职业分析

### 11.1 AI 算法工程师视角

**核心岗位技能**：

1. **NER 工程化**
   - 预训练 BERT 微调 = 最快路径（3-5天达到 ~93% F1）
   - 数据量<1000条：BERT微调优于从头训 BiLSTM-CRF
   - 数据量>10万条：可考虑纯 BiLSTM-CRF 节省推理成本

2. **序列标注面试必备**
   - 手写 LSTM forward pass（包括 $C_t$ 和 $h_t$）
   - 解释梯度消失的数学原因（$\prod W_h$ 连乘衰减）
   - LSTM vs Transformer 取舍（序列标注现在首选 BERT）

3. **常见工程 Trick**
   ```python
   # 技巧1: 字符级嵌入（处理 OOV 词）
   char_embed = CharCNN(word)
   final_embed = concat(word_embed, char_embed)
   
   # 技巧2: Dropout 防过拟合
   nn.Dropout(p=0.5)  # 放在 LSTM 输入和输出
   
   # 技巧3: 学习率调度
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
   ```

4. **技术选型（2026年）**
   - 新项目：BERT/RoBERTa + CRF（精度最高）
   - 遗留系统维护：BiLSTM-CRF（仍大量存在）
   - 边缘设备：BiGRU-CRF（更轻量）

### 11.2 AI 产品经理视角

**NLP 能力的产品转化**：

| 技术能力 | 产品形态 | 商业价值 |
|----------|----------|----------|
| NER (PER/ORG/LOC) | 合同关键要素自动抽取 | 法律科技：减少律师 80% 文档审阅时间 |
| 自定义 NER | 医疗病历结构化 | 药名、剂量、症状自动提取 |
| POS + 依存分析 | 简历信息结构化 | HR 科技：技能、经历自动分类 |
| 序列标注 + 规则 | 金融新闻事件抽取 | 量化基金：舆情信号生成 |

**产品决策框架**：
```
问题：是否需要自训练 NER 模型？

通用场景（人名/地名/组织）：
  → 直接用 spaCy 或 BERT-NER，不必自训练

垂直场景（药品名、法律术语、公司内部实体）：
  → 需要标注 1000-5000 条数据 + 微调
  → 估算成本：标注人工费 + 工程时间

实体类别>20 或频繁更新：
  → 考虑规则+模型混合，降低维护成本
```

### 11.3 创业者视角

**市场机会**：

1. **垂直领域 NER 服务（SaaS）**
   - 痛点：通用 NER 在医疗/法律/金融领域准确率只有 70-80%
   - 机会：垂直专业化，收取 API 调用费
   - 案例：Comprehend Medical（AWS 医疗 NLP）
   - 门槛：需要积累行业标注数据（护城河）

2. **低资源语言 NER**
   - 全球 7000+ 语言，只有约 50 个有商业级 NER 工具
   - XLM-RoBERTa 零样本迁移 + 少量标注 = 快速覆盖
   - 目标市场：东南亚（越南语、泰语）、非洲（斯瓦希里语）

3. **隐私保护 NER（PII 检测）**
   - GDPR/CCPA 合规需求驱动
   - 自动识别并脱敏个人信息（姓名、地址、身份证）
   - 市场：企业合规、云存储服务商

---

## 12. 面试题

### 基础概念

**Q1: 为什么 RNN 会出现梯度消失？LSTM 如何缓解？**

> **答**：梯度消失的数学原因：在 BPTT（时序反向传播）中，梯度需要通过每个时间步的 $W_h$ 传递。若 $|W_h| < 1$，则 $\prod_{i}^{t} W_h$ 会指数级衰减 → 远处时间步的梯度几乎为零 → 参数无法从远处信息中学习。
> 
> LSTM 的解决方案：细胞状态 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ 提供了一条"梯度高速公路"。遗忘门 $f_t$ 接近 1 时，$\partial C_t / \partial C_{t-1} = f_t \approx 1$，梯度可以几乎无衰减地通过。这是 LSTM 能处理长距离依赖的核心原因。

**Q2: 解释 BiLSTM-CRF 中 CRF 层的作用。如果去掉 CRF 直接用 Softmax，有什么问题？**

> **答**：去掉 CRF 后，每个位置的标签预测相互独立（局部最优）。BiLSTM+Softmax 可能预测出非法序列（如 O 后直接跟 I-PER，或 I-ORG 后跟 I-PER），因为它不知道标签转移的合法约束。
> 
> CRF 层通过学习转移矩阵 $T[y_{prev}][y_{cur}]$，用 Viterbi 算法在全局范围内找最优标签序列。训练目标是最大化金标准路径的得分相对于所有路径总得分（前向算法计算）的对数似然。
> 
> 实验结果：CoNLL-2003 上，BiLSTM 约 89-90% F1，BiLSTM-CRF 约 91% F1（+1-2%）。

**Q3: BIO 和 BIOES 标注体系各有什么优劣？**

> **答**：
> - **BIO**（Begin/Inside/Outside）：简单，常用。缺点：I 标签在单词实体和多词实体中行为相同，信息不完整。
> - **BIOES**（Begin/Inside/Outside/End/Single）：更细粒度，E 标记实体结尾，S 标记单词实体。
> 
> 实验表明 BIOES 在部分数据集上 F1 高约 0.5-1%，因为模型能明确知道实体边界。但标签集增大（从 2n+1 到 4n+1），训练数据需求也增加。

**Q4: 写出 LSTM 遗忘门和细胞状态更新的公式，并解释 $\odot$ 符号。**

> **答**：
> 
> 遗忘门：$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
> 
> 细胞状态更新：$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
> 
> $\odot$ 是**逐元素乘法**（Hadamard product），即向量中对应位置相乘。
> 
> 直觉：$f_t$ 是一个 0-1 之间的"门"向量，$f_t \odot C_{t-1}$ 表示以 $f_t$ 为比例保留旧记忆——$f_t$ 某维接近 0 则该维记忆被遗忘，接近 1 则完全保留。

### 代码实现

**Q5: 用 spaCy 做 NER，提取文本中的所有人名和组织名**

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Apple CEO Tim Cook announced that Microsoft will partner with Amazon in Seattle."

doc = nlp(text)

persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
locs = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]

print("人名:", persons)   # ['Tim Cook']
print("组织:", orgs)     # ['Apple', 'Microsoft', 'Amazon']
print("地点:", locs)     # ['Seattle']
```

**Q6: 解释 `nn.LSTM(embed_dim, hidden_dim//2, bidirectional=True)` 为什么是 `hidden_dim//2`**

> **答**：`nn.LSTM` 中 `hidden_dim//2` 是**单方向**的隐藏层维度。`bidirectional=True` 时，前向和后向各产生 `hidden_dim//2` 维的输出，拼接后得到 `hidden_dim` 维。如果写 `hidden_dim` 而不是 `hidden_dim//2`，拼接后输出会是 `2×hidden_dim`，后续全连接层维度对不上。

---

## 总结：知识图谱

```
Module 6: 神经网络与序列标注
│
├── 神经网络基础
│   ├── 神经元：加权求和 + 激活函数
│   ├── 激活函数：ReLU(隐藏层), Sigmoid(二分类), Softmax(多分类)
│   └── FFNN：前向传播 → 反向传播 → 参数更新
│
├── 序列模型
│   ├── RNN：$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$，参数共享
│   │   └── 问题：梯度消失（$W_h^t$ 连乘衰减）
│   ├── LSTM：3个门（遗忘/输入/输出）+ 细胞状态 $C_t$（长期记忆）
│   └── GRU：2个门（重置/更新），轻量化 LSTM
│
├── 序列标注
│   ├── 任务：POS 标注、NER、中文分词
│   ├── BIO 体系：B（开始）/ I（内部）/ O（非实体）
│   └── 约束：I 必须跟在 B 或同类 I 后面
│
└── 标准架构：BiLSTM-CRF
    ├── BiLSTM：双向上下文特征提取
    ├── CRF：标签转移约束 + Viterbi 全局最优解码
    └── 效果：CoNLL-2003 ~91% F1（BERT版本 ~94%）
```

**与其他模块的联系**：
- [[Module_05_Vector_Semantics]]：词嵌入（Word2Vec/GloVe）是 BiLSTM-CRF 的输入层，预训练词向量显著提升 NER 效果
- [[Module_07_Deep_Learning]]：Transformer/BERT 是序列标注的最新 SOTA 方法，替代了 BiLSTM 作为特征提取器
- [[Module_10_Dependency_Parsing]]：依存句法分析也用到序列标注技术（Arc-Standard 的动作预测）
- [[Module_11_QA_IR_RAG]]：NER 是 QA 系统的上游组件，用于识别答案中的实体类型
