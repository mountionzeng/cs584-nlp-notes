# Module 10: Dependency Parsing (依存句法分析)

> **课程**: CS584 Natural Language Processing  
> **知识体系**: [[Module_09_CFG_Parsing]] ← **本章** → [[Module_11_QA_IR_RAG]]  
> **核心问题**: 给定一个句子，如何自动识别词语之间的语法依存关系？

---

## 推荐学习资源

| 类型 | 资源 | 链接 | 说明 |
|------|------|------|------|
| 教材 | Speech and Language Processing Ch.14 (Jurafsky & Martin) | https://web.stanford.edu/~jurafsky/slp3/14.pdf | 标准NLP教材依存句法章节，免费PDF |
| 教材 | Neural Network Methods for NLP (Goldberg) | https://www.amazon.com/dp/1627052984 | 神经网络依存句法解析深度讲解 |
| 视频 | Stanford CS224N Lecture 5 — Dependency Parsing | https://www.youtube.com/watch?v=nC9_RfjYwqA | 斯坦福NLP课程，Christopher Manning主讲 |
| 视频 | CMU CS11-711 Dependency Parsing | https://www.youtube.com/watch?v=Qf-3LGaMiEk | CMU NLP课程，Graham Neubig主讲 |
| 工具 | Universal Dependencies Project | https://universaldependencies.org/ | 跨语言统一依存标注项目，含数据集 |
| 工具 | Stanford CoreNLP | https://stanfordnlp.github.io/CoreNLP/ | 工业级NLP工具包，含依存解析器 |
| 工具 | spaCy Dependency Parsing | https://spacy.io/usage/linguistic-features#dependency-parse | Python NLP库，含可视化工具 |
| 论文 | A Fast and Accurate Dependency Parser (Chen & Manning 2014) | https://aclanthology.org/D14-1082.pdf | 神经网络transition-based解析奠基论文 |
| 论文 | Deep Biaffine Attention (Dozat & Manning 2017) | https://arxiv.org/abs/1611.01734 | 当前最优graph-based依存解析方法 |
| 数据集 | Penn Treebank (LDC) | https://catalog.ldc.upenn.edu/LDC99T42 | 英语标注语料库（付费，机构可访问） |

---

## 目录

1. [基础概念：依存关系的本质](#1-基础概念)
2. [依存结构的形式化定义](#2-形式化定义)
3. [Transition-Based 解析（弧转移法）](#3-transition-based解析)
4. [Graph-Based 解析（图算法法）](#4-graph-based解析)
5. [神经网络依存解析](#5-神经网络解析)
6. [评估指标：UAS / LAS / PARSEVAL](#6-评估指标)
7. [三视角职业分析](#7-职业分析)
8. [面试高频题](#8-面试题)

---

## 1. 基础概念

### 1.1 依存关系是什么？

依存句法（Dependency Syntax）描述句子中词与词之间的**二元不对称关系**。

**核心直觉类比：**

> **家族树比喻**：想象一棵家谱树。Root（根词）是"族长"，每个词是一个"家庭成员"。每个成员（除族长外）有且只有一个"父亲"（Head），但可以有多个"儿子"（Dependents）。词语之间的关系不是"兄弟平等"，而是有明确的支配/从属方向。

> **公司组织架构比喻**：句子的主要动词（Root）是 CEO，主语是 VP，修饰语是 Manager... 每个员工（词）向且仅向一个上级（Head）汇报，体现的是**支配关系**（governance）。

**关键术语对照表：**

| 术语 | 中文 | 定义 | 例子（"The cat sat"）|
|------|------|------|---------------------|
| **Head** | 核心词 | 语法上支配另一个词的词 | `sat` 是 `cat` 的 Head |
| **Dependent** | 依存词 | 受另一个词支配的词 | `cat` 依存于 `sat` |
| **Root** | 根节点 | 整个句子的核心（无Head）| `sat` 是句子的 Root |
| **Arc / Edge** | 依存弧 | 从 Head 到 Dependent 的有向边 | `sat → cat`（nsubj 关系）|
| **Relation Label** | 关系标签 | 弧的语法功能标记 | nsubj, obj, det, amod... |

**⚠️ Root vs Head 的易混点：**
- **Root** 是全局概念：整棵依存树只有**一个** Root，它是整个句子的"最顶层"词（通常是主要动词），没有任何词支配它。
- **Head** 是局部概念：每条弧都有一个 Head（弧的起点）和一个 Dependent（弧的终点）。Root 本身也是某些弧的 Head（它支配它的直接子节点）。
- Root 是没有 Head 的那个词；所有其他词都有且仅有一个 Head。

### 1.2 依存句法 vs 成分句法

```
成分句法（Constituency）:            依存句法（Dependency）:
        S                                    sat (ROOT)
       / \                                  /        \
      NP   VP                            cat          mat
     / \  /  \                           |            |
   The cat sat  on...                   The      on
                                                  |
                                                 the mat
```

| 维度 | 成分句法（CFG/PCFG）| 依存句法 |
|------|-------------------|----------|
| 核心单元 | 短语（NP, VP...）| 词-词关系 |
| 树结构 | 二叉/n叉短语树 | 有向依存树 |
| 适合语言 | 词序固定语言（英语）| 自由词序语言（德语、俄语、中文）|
| 长距离依存 | 需要特殊规则处理 | 直接用弧表示 |
| 词汇化程度 | 低（节点是短语标记）| 高（节点就是词）|
| 工程用途 | 句法树银行、文法归纳 | 信息抽取、关系抽取 |

---

## 2. 形式化定义

### 2.1 依存图的数学定义

给定句子 $w_1, w_2, ..., w_n$，依存图 $G = (V, A)$ 定义为：
- **顶点集** $V = \{w_0, w_1, ..., w_n\}$，其中 $w_0$ 是虚拟根节点 ROOT
- **弧集** $A \subseteq V \times V \times L$，每条弧 $(w_i, w_j, l)$ 表示 $w_i$ 是 $w_j$ 的 Head，关系标签为 $l \in L$

**合法依存树的约束条件：**
1. **单头约束（Single Head）**: 每个词（除 ROOT 外）恰好有一个 Head
2. **无环约束（Acyclicity）**: 图中不存在环
3. **连通约束（Connectivity）**: 所有词都通过弧路径连接到 ROOT
4. **（可选）投影性约束（Projectivity）**: 弧不交叉（见下文）

### 2.2 投影性（Projectivity）：关键概念

**定义**：弧 $(i, j)$ 是投影性的，当且仅当所有位于 $i$ 和 $j$ 之间（位置上）的词 $k$（$i < k < j$ 或 $j < k < i$），都是 $i$ 的后代（descendant）。

**直觉**：投影性弧在线性词序图上**不会和其他弧交叉**。

**投影性例子分析** — 句子："That man there makes good food"

```
位置:  1     2    3      4     5     6
词:   That  man  there  makes good  food

依存弧（带标签）：
  makes → That   (det，1←4，跨越2,3)    ← 非投影！
  makes → man    (nsubj，2←4，跨越3)   ← 投影性？取决于 there 是否是 man 的后代
  man   → That   (det，1←2)             ← 投影
  man   → there  (advmod，3←2)          ← 投影（3在2右侧，是2的后代）
  makes → food   (obj，6←4，跨越5)      ← 取决于good
  food  → good   (amod，5←6)            ← 投影
```

**非投影依存的真实例子**（英语相对少见，斯拉夫语、德语常见）：
```
"Which car did you buy?"
 1      2   3   4   5

buy → car  弧：(5 → 2)，跨越了3,4但 did(3) 和 you(4) 不是 buy(5) 或 car(2) 的后代
→ 非投影弧！
```

```
投影性依存树（弧不交叉）：    非投影性依存树（弧交叉）：
    ___________                   _______ _______
   |     ___   |                 |      X      |
   |    |   |  |                 |     / \     |
w1  w2  w3  w4  w5              w1   w2  w3  w4  w5
```

**为什么投影性重要？**
- **Transition-based 解析**（Arc-Standard）只能处理投影性依存树
- **Graph-based 解析**可以处理非投影依存
- 英语依存树约 **~99%** 是投影性的；德语约 ~80%

### 2.3 通用依存关系标签（UD）

Universal Dependencies 项目定义了跨语言统一标签集：

| 标签 | 全称 | 例子 |
|------|------|------|
| nsubj | nominal subject | "**cats** eat fish" → cats←eat |
| obj | object | "eat **fish**" → fish←eat |
| det | determiner | "**the** cat" → the←cat |
| amod | adjectival modifier | "**big** cat" → big←cat |
| advmod | adverbial modifier | "run **fast**" → fast←run |
| nmod | nominal modifier | "book **of poems**" → of←book |
| case | case marking | "**of** poems" → of←poems |
| aux | auxiliary | "**will** run" → will←run |
| punct | punctuation | "Hello **.**" → .←Hello |
| root | root of sentence | artificial ROOT→main verb |

---

## 3. Transition-Based 解析

### 3.1 Arc-Standard 系统

Transition-based 解析将依存分析转化为**状态机**问题：从初始状态出发，通过一系列**动作（Transition）**到达终止状态，同时构建依存树。

**解析器状态** = 三元组 $(S, B, A)$：
- $S$：**栈（Stack）** — 存放正在处理的词
- $B$：**缓冲区（Buffer）** — 存放待处理词的队列
- $A$：**弧集（Arc Set）** — 已构建的依存弧集合

**Arc-Standard 三种动作：**

| 动作 | 触发条件 | 效果 | 
|------|----------|------|
| **SHIFT** | Buffer 非空 | 将 Buffer 头部词移入 Stack |
| **LEFT-ARC(l)** | Stack 有 ≥2 个元素，且栈顶下一个词不是 ROOT | 添加弧 `s1 →(l) s2`（s1是栈顶，s2是次栈顶）；从 Stack 弹出 s2 |
| **RIGHT-ARC(l)** | Stack 有 ≥2 个元素 | 添加弧 `s2 →(l) s1`（s2是栈顶，s1是次栈顶）；从 Stack 弹出 s1 |

**⚠️ 方向记忆**：
- LEFT-ARC：弧从**右指左**（栈顶 → 次栈顶），弹出**次栈顶**（左边的词）
- RIGHT-ARC：弧从**左指右**（次栈顶 → 栈顶），弹出**栈顶**（右边的词）

### 3.2 完整解析追踪

**句子**："I ate fish"（简化，无标签）

**初始状态**：Stack=[ROOT], Buffer=[I, ate, fish], A={}

```
步骤  动作          Stack           Buffer          新增弧
----  ---------    ---------------  --------------  ---------
0     初始状态      [ROOT]           [I, ate, fish]  {}
1     SHIFT         [ROOT, I]        [ate, fish]     -
2     SHIFT         [ROOT, I, ate]   [fish]          -
3     LEFT-ARC      [ROOT, ate]      [fish]          ate→I (nsubj)
      (nsubj)       (I被弹出)
4     SHIFT         [ROOT, ate, fish] []             -
5     RIGHT-ARC     [ROOT, ate]      []              ate→fish (obj)
      (obj)         (fish被弹出)
6     RIGHT-ARC     [ROOT]           []              ROOT→ate (root)
      (root)        (ate被弹出)
7     终止状态       [ROOT]           []              A={ate→I, ate→fish, ROOT→ate}
```

**构建的依存树：**
```
        ROOT
         |
        ate
       /   \
      I    fish
    (nsubj) (obj)
```

### 3.3 Arc-Standard 算法伪代码

```python
def arc_standard_parse(sentence):
    """
    Arc-Standard Transition-Based Dependency Parser
    
    需要一个 oracle/classifier 来决定每一步执行哪个动作
    """
    # 初始化
    stack = [ROOT_TOKEN]
    buffer = list(sentence)  # 从左到右的词序列
    arcs = set()
    
    while not (len(stack) == 1 and len(buffer) == 0):
        # 获取特征（用于分类器输入）
        features = extract_features(stack, buffer, arcs)
        
        # 分类器预测动作（训练时用 oracle，推断时用模型）
        action = classifier.predict(features)
        
        if action == SHIFT and len(buffer) > 0:
            stack.append(buffer.pop(0))
            
        elif action == LEFT_ARC and len(stack) >= 2:
            s1 = stack[-1]   # 栈顶
            s2 = stack[-2]   # 次栈顶
            label = action.label
            arcs.add((s1, label, s2))  # s1 是 s2 的 Head
            stack.pop(-2)              # 弹出 s2（次栈顶）
            
        elif action == RIGHT_ARC and len(stack) >= 2:
            s1 = stack[-1]   # 栈顶
            s2 = stack[-2]   # 次栈顶
            label = action.label
            arcs.add((s2, label, s1))  # s2 是 s1 的 Head
            stack.pop(-1)              # 弹出 s1（栈顶）
    
    return arcs

def extract_features(stack, buffer, arcs):
    """
    传统特征工程（Chen & Manning 2014 之前的方法）
    """
    features = []
    # 栈顶词及其POS标签
    if len(stack) >= 1:
        features.append(f"s0_word={stack[-1].word}")
        features.append(f"s0_pos={stack[-1].pos}")
    if len(stack) >= 2:
        features.append(f"s1_word={stack[-2].word}")
        features.append(f"s1_pos={stack[-2].pos}")
    # 缓冲区头部
    if len(buffer) >= 1:
        features.append(f"b0_word={buffer[0].word}")
        features.append(f"b0_pos={buffer[0].pos}")
    # ... 通常 ~40-70 个模板特征
    return features
```

### 3.4 Oracle 训练（有监督学习）

```python
def generate_oracle_transitions(gold_tree):
    """
    给定金标准依存树，生成 Arc-Standard 的正确动作序列（oracle）
    用于训练分类器
    """
    stack = [ROOT_TOKEN]
    buffer = list(gold_tree.tokens)
    transitions = []
    
    while not done(stack, buffer):
        s1 = stack[-1] if len(stack) >= 1 else None
        s2 = stack[-2] if len(stack) >= 2 else None
        b0 = buffer[0] if len(buffer) >= 0 else None
        
        # 优先检查能否执行 LEFT-ARC
        if s2 and gold_tree.has_arc(s1, s2):
            label = gold_tree.get_label(s1, s2)
            transitions.append(LEFT_ARC(label))
            stack.pop(-2)
            
        # 检查能否执行 RIGHT-ARC（需要s1的所有依存词都已处理）
        elif s1 and s2 and gold_tree.has_arc(s2, s1):
            if all_dependents_processed(s1, buffer, gold_tree):
                label = gold_tree.get_label(s2, s1)
                transitions.append(RIGHT_ARC(label))
                stack.pop(-1)
            else:
                transitions.append(SHIFT)
                stack.append(buffer.pop(0))
        else:
            transitions.append(SHIFT)
            stack.append(buffer.pop(0))
    
    return transitions
```

### 3.5 Arc-Eager 变体（简述）

Arc-Standard 的问题：RIGHT-ARC 必须等待 Dependent 的所有子树都构建完才能执行（因为弹出后就无法访问了）。

**Arc-Eager** 允许提前执行 RIGHT-ARC，引入第四个动作 **REDUCE**（弹出栈顶，不添加弧）。

| 动作 | Arc-Standard | Arc-Eager |
|------|-------------|-----------|
| SHIFT | ✓ | ✓ |
| LEFT-ARC | 弹出次栈顶 | 弹出次栈顶（立即） |
| RIGHT-ARC | 弹出栈顶 | **不弹出**（先保留） |
| REDUCE | ✗ | ✓（弹出已有 Head 的词）|

Arc-Eager 通常比 Arc-Standard 更快（平均少 ~n 步），但 oracle 设计更复杂。

---

## 4. Graph-Based 解析

### 4.1 核心思想

将依存解析建模为**有向图中寻找最大生成树（Maximum Spanning Tree, MST）**的问题。

**步骤：**
1. 对句子中**所有可能的弧** $(i, j, l)$ 计算得分 $\text{score}(i, j, l)$
2. 找到得分之和最大的合法依存树（满足单头、无环、连通约束）

**总得分**（假设弧之间条件独立）：
$$\text{score}(T) = \sum_{(i,j,l) \in T} \text{score}(i, j, l)$$

### 4.2 Chu-Liu/Edmonds 算法（有向 MST）

对于**有向图**，最大生成树用 **Chu-Liu/Edmonds 算法**（时间复杂度 $O(n^2)$）。

**算法步骤（伪代码）：**

```
function ChuLiuEdmonds(G, root):
    """
    G = (V, E) 有向带权图
    root = 指定根节点
    返回: 以 root 为根的最大有向生成树
    """
    
    Step 1: 对每个非根节点 v，选择得分最高的入弧
            best_in[v] = argmax_{u: (u,v)∈E} score(u, v)
    
    Step 2: 检查是否有环（cycle）
            If 无环:
                return best_in 构成的树（已经是 MST）
    
    Step 3: 如果有环 C = {v1, v2, ..., vk}:
        a) 将环 C 收缩为一个超级节点 C*
        b) 更新进入/离开 C 的弧的权重：
           进入 C 的弧 (u, vi) 的新权重 = score(u, vi) - score(best_in[vi]) + score(C)
        c) 递归调用 ChuLiuEdmonds(G', root)（G' 是收缩后的图）
        d) 展开超级节点，恢复环内的弧（除一条被替换的入弧外）
    
    return 最大有向生成树
```

**Chu-Liu/Edmonds 手动追踪（简单例子）：**

句子："dog bites man"（4个节点：ROOT, dog, bites, man）

```
可能的弧及得分（节选）：
ROOT→bites: 5,  ROOT→dog: 2,   ROOT→man: 1
bites→dog:  4,  bites→man: 3
dog→bites:  1,  man→bites: 2

Step 1: 每个词选最高入弧：
  dog:   best = bites→dog (4)
  bites: best = ROOT→bites (5)
  man:   best = bites→man (3)

Step 2: 检查环 → 无环（ROOT→bites→dog, ROOT→bites→man，树形结构）

结果树: ROOT→bites, bites→dog, bites→man
得分: 5+4+3 = 12  ✓
```

### 4.3 Eisner 算法（投影性有向 MST，$O(n^3)$）

对于只需投影性依存树的情况，可以用 **Eisner 算法**（动态规划，比 Chu-Liu/Edmonds 快）。

```
Eisner DP 状态定义：
  C[i][j][d] = 以 i 为根，覆盖区间 [i,j] 的最优完整跨度（complete span）
  I[i][j][d] = 以 i 为根，覆盖区间 [i,j] 的最优不完整跨度（incomplete span）
  d = 0 (左向) 或 1 (右向)

关键递推：
  I[i][j][右] = max_{i≤k<j} C[i][k][右] + C[j][k+1][左] + score(i→j)
  C[i][j][右] = max_{i≤k<j} C[i][k][右] + I[i][k+1][右]（???）

时间复杂度: O(n³)，适合投影性假设下的高效解析
```

### 4.4 Transition-Based vs Graph-Based 对比

| 维度 | Transition-Based | Graph-Based |
|------|-----------------|-------------|
| 时间复杂度 | $O(n)$ （线性）| $O(n^2)$ ~ $O(n^3)$ |
| 推断速度 | **快**（实时应用首选）| 较慢 |
| 非投影性支持 | Arc-Standard: ✗ | ✓ |
| 全局最优性 | ✗（贪心，易传播错误）| ✓（精确搜索）|
| 准确率（历史）| 略低 | 略高 |
| 神经网络结合 | Chen & Manning 2014 | Dozat & Manning 2017 |
| 适合场景 | 高吞吐量生产环境 | 高精度研究/比赛 |

---

## 5. 神经网络解析

### 5.1 Chen & Manning 2014：第一个神经网络依存解析器

**核心创新**：用神经网络替代传统特征工程，将 transition 决策建模为多分类问题。

```python
import torch
import torch.nn as nn

class NeuralTransitionParser(nn.Module):
    """
    Chen & Manning (2014) 架构简化实现
    """
    def __init__(self, vocab_size, pos_size, label_size, 
                 embed_dim=50, hidden_dim=200, num_transitions=None):
        super().__init__()
        
        # 词嵌入、POS嵌入、标签嵌入
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(pos_size, embed_dim)
        self.label_embed = nn.Embedding(label_size, embed_dim)
        
        # 特征：栈顶3词 + 缓冲区3词 + 6个依存词 = 18个特征位置
        # 每个位置3个特征（词、POS、标签） = 54个嵌入
        input_dim = 18 * 3 * embed_dim  # 2700
        
        # MLP: 输入→隐藏→输出（使用 cube 激活函数！）
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_transitions)
        
    def forward(self, word_ids, pos_ids, label_ids):
        # 获取嵌入
        w_emb = self.word_embed(word_ids)
        p_emb = self.pos_embed(pos_ids)
        l_emb = self.label_embed(label_ids)
        
        # 拼接特征
        x = torch.cat([w_emb, p_emb, l_emb], dim=-1).flatten(1)
        
        # Cube 激活（原论文特有，效果比 ReLU 好）
        h = torch.pow(self.hidden(x), 3)
        
        # 输出 transition 得分
        logits = self.output(h)
        return logits
```

### 5.2 Biaffine 注意力解析器（Dozat & Manning 2017）

目前 SOTA 的 graph-based 方法：

```python
class BiaffineParser(nn.Module):
    """
    Deep Biaffine Attention for Neural Dependency Parsing
    Dozat & Manning (2017) 核心架构
    """
    def __init__(self, hidden_dim=400, arc_dim=500, label_dim=100, num_labels=40):
        super().__init__()
        
        # BiLSTM 编码器（或可替换为 BERT/RoBERTa）
        self.bilstm = nn.LSTM(
            input_size=300,  # 词嵌入维度
            hidden_size=hidden_dim,
            num_layers=3,
            bidirectional=True,
            dropout=0.33
        )
        
        # Head/Dependent 投影（分别用于弧预测和标签预测）
        self.arc_head_mlp = nn.Linear(hidden_dim * 2, arc_dim)
        self.arc_dep_mlp = nn.Linear(hidden_dim * 2, arc_dim)
        self.label_head_mlp = nn.Linear(hidden_dim * 2, label_dim)
        self.label_dep_mlp = nn.Linear(hidden_dim * 2, label_dim)
        
        # Biaffine 层（核心！）
        # 弧预测：输出 n×n 得分矩阵
        self.arc_biaffine = Biaffine(arc_dim, arc_dim, 1)
        # 标签预测：输出 n×n×num_labels 得分张量
        self.label_biaffine = Biaffine(label_dim, label_dim, num_labels)
    
    def forward(self, words, chars=None):
        # 1. BiLSTM 编码
        lstm_out, _ = self.bilstm(word_embeddings(words))
        
        # 2. 计算弧得分（所有词对）
        arc_head = F.relu(self.arc_head_mlp(lstm_out))   # [n, arc_dim]
        arc_dep = F.relu(self.arc_dep_mlp(lstm_out))     # [n, arc_dim]
        arc_scores = self.arc_biaffine(arc_head, arc_dep) # [n, n]
        
        # 3. 用 Chu-Liu/Edmonds 找最大生成树
        predicted_heads = chu_liu_edmonds(arc_scores)
        
        # 4. 基于预测的 Head，计算标签得分
        label_head = F.relu(self.label_head_mlp(lstm_out))
        label_dep = F.relu(self.label_dep_mlp(lstm_out))
        label_scores = self.label_biaffine(label_head, label_dep)  # [n, n, L]
        
        return arc_scores, label_scores, predicted_heads

class Biaffine(nn.Module):
    """
    双仿射变换：score(i,j) = h_i^T W h_j + U h_i + V h_j + b
    """
    def __init__(self, in1_dim, in2_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in1_dim, output_dim, in2_dim))
        self.U = nn.Linear(in1_dim, output_dim, bias=False)
        self.V = nn.Linear(in2_dim, output_dim, bias=True)
    
    def forward(self, x1, x2):
        # x1: [batch, n, d1], x2: [batch, n, d2]
        # 双线性部分
        bilinear = torch.einsum('bnd,dod,bmd->bnom', x1, self.W, x2)
        # 线性部分
        linear = self.U(x1).unsqueeze(3) + self.V(x2).unsqueeze(2)
        return (bilinear + linear).squeeze(-1)
```

### 5.3 基于 BERT 的依存解析

```python
from transformers import BertModel

class BERTDependencyParser(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=40):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_dim = self.bert.config.hidden_size  # 768
        
        # 在 BERT 之上加 Biaffine 解析头
        self.arc_head = nn.Linear(hidden_dim, 500)
        self.arc_dep = nn.Linear(hidden_dim, 500)
        self.arc_biaffine = Biaffine(500, 500, 1)
        self.label_head = nn.Linear(hidden_dim, 100)
        self.label_dep = nn.Linear(hidden_dim, 100)
        self.label_biaffine = Biaffine(100, 100, num_labels)
    
    def forward(self, input_ids, attention_mask, subword_to_word_map):
        # BERT 编码（注意 subword 到 word 的映射！）
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq, 768]
        
        # 将 subword 表示聚合为 word 表示（取首个 subword）
        word_repr = gather_word_representations(hidden_states, subword_to_word_map)
        
        # Biaffine 解析（同上）
        arc_h = F.relu(self.arc_head(word_repr))
        arc_d = F.relu(self.arc_dep(word_repr))
        arc_scores = self.arc_biaffine(arc_h, arc_d)
        
        return arc_scores
```

---

## 6. 评估指标

### 6.1 UAS 和 LAS

**UAS（Unlabeled Attachment Score）= 弧方向正确率**：
$$\text{UAS} = \frac{\text{head 预测正确的词数}}{\text{词总数（不含 ROOT）}} \times 100\%$$

**LAS（Labeled Attachment Score）= 带标签弧正确率**：
$$\text{LAS} = \frac{\text{head AND 标签都预测正确的词数}}{\text{词总数（不含 ROOT）}} \times 100\%$$

**计算示例：**

```
金标准：                    预测结果：
ROOT → ate (root)          ROOT → ate (root)   ✓ UAS+LAS
ate → I (nsubj)            ate → I (nsubj)     ✓ UAS+LAS  
ate → fish (obj)           ate → fish (dobj)   ✓ UAS, ✗ LAS（标签错）
fish → the (det)           ate → the (?)       ✗ UAS, ✗ LAS（head错）

评估词：I, fish, the（3个，不含ROOT）
UAS = 2/3 = 66.7%
LAS = 1/3 = 33.3%
```

**当前 SOTA 水平**（英语 PTB 数据集）：
- UAS: ~96%+ （Biaffine + BERT）
- LAS: ~94%+ （Biaffine + BERT）

### 6.2 PARSEVAL 和 evalb（成分句法评估）

> **注意**：PARSEVAL/evalb 是**成分句法**（constituency parsing）的评估指标，不是依存句法的，但经常在对比讲解中一并介绍。

**PARSEVAL 三指标**：

$$\text{Precision} = \frac{|\text{预测短语} \cap \text{金标准短语}|}{|\text{预测短语}|}$$

$$\text{Recall} = \frac{|\text{预测短语} \cap \text{金标准短语}|}{|\text{金标准短语}|}$$

$$\text{F1} = \frac{2 \times P \times R}{P + R}$$

**"短语"的匹配条件**：标签 + 词跨度（span）都一致，才算匹配。

```
金标准树:                    预测树:
    S                            S
   / \                          / \
  NP   VP                      NP   VP
 /   /   \                    /   /   \
I  ate   NP                  I  ate  NP
          |                         |
         fish                     the fish
          
金标准短语集合：{(S,0-4), (NP,0-1), (VP,1-4), (NP,2-4)}
预测短语集合：  {(S,0-4), (NP,0-1), (VP,1-4), (NP,2-4)}

Precision = 4/4 = 100%
Recall = 4/4 = 100%
F1 = 100%
```

**evalb 工具**：PARSEVAL 的标准实现程序，比赛标准评测工具。

```bash
# evalb 使用方式
./evalb -p evalb.prm gold.mrg predicted.mrg

# 输出示例：
#  ============================================================
#   Number of sentence        =  2416
#   Number of Error sentence  =  0
#   Number of Skip  sentence  =  0
#   Number of Valid sentence  =  2416
#   Bracketing Recall         =  89.69 (22131/24673)
#   Bracketing Precision      =  90.03 (22131/24582)
#   Bracketing F-measure      =  89.86
#   Complete match            =  36.52
#   Average crossing          =  0.55
#   ...
#  ============================================================
```

**PARSEVAL 争议点**：
- **不区分功能词 vs 内容词**的对齐（但 LFB 变体会区分）
- 对**短句子**的小错误惩罚过重
- 不同语言标注规范不同导致**跨语言比较困难**（UD 项目部分解决了这个问题）
- **Complete match** (完全匹配率) 更严格，要求整棵树完全正确

### 6.3 评估指标对比表

| 指标 | 任务类型 | 粒度 | 主流使用 |
|------|----------|------|----------|
| UAS | 依存解析 | 词级 Head 正确率 | CoNLL 比赛 |
| LAS | 依存解析 | 词级 Head+标签正确率 | CoNLL 比赛（主要指标）|
| PARSEVAL F1 | 成分解析 | 短语 span 正确率 | PTB 评测 |
| evalb Complete Match | 成分解析 | 整句完全正确率 | 严格评测 |

---

## 7. 职业分析

### 7.1 AI 算法工程师视角

**核心工程挑战：**

1. **速度 vs 准确率权衡**
   - 生产环境要求 <10ms/句，Arc-Standard + 神经网络是首选
   - 离线分析可以用 Graph-based（Biaffine）追求高精度
   
2. **多语言支持**
   - Universal Dependencies 提供 100+ 语言统一标注
   - mBERT/XLM-RoBERTa + Biaffine 可以实现 zero-shot 跨语言迁移

3. **实际应用场景**
   - **信息抽取**：实体关系识别（谁 does 什么 to 谁）
   - **问答系统**：理解问题结构（nsubj + obj 对齐）
   - **机器翻译**：依存树辅助 reordering
   - **代码理解**：程序依存图 (PDG) 类似结构

4. **技术栈建议**
   - 解析器：spaCy (速度优先) / Stanza (精度优先) / Trankit
   - 训练框架：PyTorch + HuggingFace Transformers
   - 数据：Universal Dependencies Treebanks

### 7.2 AI 产品经理视角

**用户价值翻译：**

依存解析的"技术语言"→"产品价值"：

| 技术能力 | 产品应用 | 用户价值 |
|----------|----------|----------|
| nsubj/obj 提取 | 合同要素抽取、新闻摘要 | 自动提取"谁、做了什么、针对谁" |
| 修饰关系 (amod) | 商品评论分析（"好用的手机"）| 属性-情感对应分析 |
| 从句分析 | 法律文本理解 | 复杂条款的主从结构解析 |
| 跨语言依存 | 多语言客服机器人 | 统一的语义理解后端 |

**产品决策框架：**
- 何时选依存而非成分解析？→ 需要提取"语义关系"（做什么/对谁做）时
- 何时不需要依存解析？→ 纯分类任务（情感分析、主题分类），端到端神经网络往往更好
- 依存解析的 ROI 评估：标注成本高（专家标注），适合高价值垂直领域（法律、医疗、金融）

### 7.3 创业者视角

**市场机会：**

1. **垂直领域依存解析器**
   - 通用依存解析器在法律/医疗文本上准确率下降 10-20%
   - 垂直领域精标数据（1000-5000句）+ 微调 = 高价值差异化产品
   - 目标客户：法律科技公司（合同审查）、医疗AI公司（病历理解）

2. **低资源语言机会**
   - 东南亚、非洲语言的依存解析器几乎空白
   - Universal Dependencies 框架 + 众包标注 + 迁移学习可以快速覆盖

3. **代码依存分析**
   - 程序依存图（Program Dependence Graph）= 代码世界的依存句法
   - 应用：代码理解、自动重构、漏洞检测
   - GitHub Copilot 等产品验证了代码理解的市场需求

**竞争分析**：
- Stanford CoreNLP, spaCy：通用工具，开源，覆盖主流语言
- 创业机会在：特定语言、特定领域、实时性要求极高的场景

---

## 8. 面试题

### 基础概念

**Q1: 什么是投影性依存？为什么它很重要？**

> **答**：弧 (i,j) 是投影性的，当且仅当 i 和 j 之间（线性位置）的所有词都是 i 的后代。直观上，投影性弧在词序图上不交叉。
> 
> 重要性：Arc-Standard 等 transition-based 解析器只能正确处理投影性依存树（因为动作序列的线性性质决定了无法处理交叉弧）。英语中 ~99% 的依存弧是投影性的，但德语、荷兰语中非投影弧更常见，需要 Graph-based 方法或 Arc-Eager with swap 操作处理。

**Q2: UAS 和 LAS 的区别是什么？什么情况下 UAS 高但 LAS 低？**

> **答**：UAS 只要求预测的 Head 正确，LAS 要求 Head 和关系标签都正确。
> 
> UAS 高 LAS 低的情况：解析器能正确识别词语之间的支配方向，但标签预测错误。例如把 nsubj (主语) 标成 obj (宾语)，或把 amod 标成 advmod。这通常说明模型的**标签分类能力**比结构预测能力弱，可能需要更多标签标注数据或更好的标签分类器。

**Q3: 为什么 Arc-Standard 无法处理非投影依存？**

> **答**：Arc-Standard 的 Stack-Buffer 模型严格按照**线性词序**处理词语：词按从左到右的顺序进入 Stack，弧只能在栈顶两个元素之间建立。对于弧 (i, j)，如果 i 和 j 之间存在不属于 i 子树的词 k（即非投影词），则在处理 j 时 k 已经被弹出 Stack 且不属于 i 的子树，无法形成合法的非投影弧。
>
> 解决方案：Graph-based 方法（Chu-Liu/Edmonds）不受词序约束，可直接对所有词对打分后找全局最优树，自然支持非投影依存。

### 算法和实现

**Q4: 请手动追踪 Arc-Standard 解析 "She sees him"（标签：nsubj/obj）**

> **答**：
> ```
> Stack          Buffer           动作          新增弧
> [ROOT]         [She,sees,him]   SHIFT
> [ROOT,She]     [sees,him]       SHIFT
> [ROOT,She,sees][him]            LEFT-ARC      sees→She (nsubj)
> [ROOT,sees]    [him]            SHIFT
> [ROOT,sees,him][]               RIGHT-ARC     sees→him (obj)
> [ROOT,sees]    []               RIGHT-ARC     ROOT→sees (root)
> [ROOT]         []               终止
> ```

**Q5: Biaffine 解析器为什么比传统特征工程方法好？**

> **答**：
> 1. **表示能力**：词嵌入（尤其是 BERT contextual embeddings）能捕捉词的上下文语义，而传统特征是离散的、稀疏的。
> 2. **弧得分计算**：Biaffine 变换 $h_i^T W h_j$ 可以高效计算所有词对的弧得分，复杂度 $O(n^2)$ 但向量化实现极快。
> 3. **端到端训练**：特征提取和解析联合训练，避免传统流水线的错误传播。
> 4. **参数效率**：不需要手工设计 40-70 个特征模板，参数自动学习。

**Q6: 如何用 spaCy 进行依存解析并可视化？**

```python
import spacy
from spacy import displacy

# 加载模型
nlp = spacy.load("en_core_web_sm")

# 解析句子
doc = nlp("The quick brown fox jumps over the lazy dog")

# 遍历依存关系
for token in doc:
    print(f"{token.text:12} --{token.dep_:8}--> {token.head.text}")

# 可视化（在 Jupyter 中）
displacy.render(doc, style="dep", jupyter=True)

# 提取特定关系（如找到所有主语）
subjects = [tok for tok in doc if tok.dep_ == "nsubj"]
print("主语:", [s.text for s in subjects])

# 找到动词-宾语对
verb_obj_pairs = [(tok.head.text, tok.text) 
                  for tok in doc if tok.dep_ == "dobj"]
print("动词-宾语对:", verb_obj_pairs)
```

**Q7: 如何评估一个新的依存解析器？**

> **答**：
> 1. **选数据集**：英语用 Penn Treebank (WSJ) 的 CoNLL-2009 格式；多语言用 Universal Dependencies treebanks
> 2. **划分**：标准 train/dev/test 分割（通常 sec 2-21 train, sec 22 dev, sec 23 test for PTB）
> 3. **计算 UAS/LAS**：用 CoNLL 官方评测脚本（conll17_ud_eval.py 等）
> 4. **分析错误**：检查哪些句法关系类型（nsubj、obj、nmod...）错误率最高
> 5. **长距离依存**：单独统计依存弧长度 > 5 的 UAS/LAS，这是困难case
> 6. **速度测试**：句子/秒，通常在 CPU 和 GPU 上都测

---

## 总结：知识图谱

```
依存句法分析
├── 基础概念
│   ├── Head / Dependent / Root 三角关系
│   ├── 依存树 = 单头 + 无环 + 连通 + (投影性)
│   └── UD 关系标签体系
├── 解析方法
│   ├── Transition-Based
│   │   ├── Arc-Standard (O(n), 仅投影)
│   │   ├── Arc-Eager (O(n), 更快)
│   │   └── 神经网络: Chen & Manning 2014
│   └── Graph-Based
│       ├── Chu-Liu/Edmonds MST (O(n²), 非投影)
│       ├── Eisner DP (O(n³), 投影)
│       └── 神经网络: Dozat & Manning 2017 (Biaffine)
├── 神经网络增强
│   ├── 词嵌入 → BiLSTM → Biaffine
│   └── BERT contextual embeddings → Biaffine (SOTA)
└── 评估
    ├── UAS / LAS（依存解析）
    └── PARSEVAL / evalb F1（成分解析，对比参考）
```

**与其他模块的联系：**
- [[Module_09_CFG_Parsing]]：成分句法是另一类句法分析范式，与依存句法互补
- [[Module_08_Machine_Translation]]：POS 标注是依存解析的上游任务，POS 特征对传统解析器至关重要
- [[Module_11_QA_IR_RAG]]：依存结构是语义角色标注（SRL）、信息抽取的输入
- [[Module_07_Deep_Learning]]：BERT 等语言模型为依存解析提供上下文表示，大幅提升精度
