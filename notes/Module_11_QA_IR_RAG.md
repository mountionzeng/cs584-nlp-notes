# Module 11: Question Answering, Information Retrieval & RAG

> **课程**: CS584 Natural Language Processing  
> **日期**: March 31, 2026  
> **知识体系**: [[Module_10_Dependency_Parsing]] ← **本章** → [[Module_12_Dialogue_Systems]]  
> **核心问题**: 如何构建能自动回答自然语言问题的系统？如何从大量文档中精准检索信息？

---

## 推荐学习资源

| 类型 | 资源 | 链接 | 说明 |
|------|------|------|------|
| 教材 | SLP Chapter 23: QA and IR (Jurafsky & Martin) | https://web.stanford.edu/~jurafsky/slp3/23.pdf | 标准NLP教材，免费PDF，本章最直接参考 |
| 论文 | RAG (Lewis et al., 2020) | https://arxiv.org/abs/2005.11401 | 原始RAG论文，必读 |
| 论文 | BERT for QA (Devlin et al., 2019) | https://arxiv.org/abs/1810.04805 | BERT原文，SQuAD效果分析 |
| 论文 | ColBERT (Khattab & Zaharia, 2020) | https://arxiv.org/abs/2004.12832 | 高效密集检索，Late Interaction |
| 论文 | DPR (Karpukhin et al., 2020) | https://arxiv.org/abs/2004.04906 | Dense Passage Retrieval |
| 视频 | Stanford CS224N Lecture on QA | https://www.youtube.com/watch?v=NcqfHa0_YmU | 斯坦福NLP课程QA讲解 |
| 视频 | Pinecone RAG Tutorial | https://www.youtube.com/watch?v=T-D1OfcDW1M | RAG实战教程 |
| 工具 | Hugging Face QA Pipeline | https://huggingface.co/docs/transformers/task_summary#question-answering | 直接可用的QA API |
| 工具 | LangChain RAG | https://python.langchain.com/docs/use_cases/question_answering/ | RAG系统搭建框架 |
| 数据集 | SQuAD 2.0 | https://rajpurkar.github.io/SQuAD-explorer/ | 标准QA benchmark |
| 数据集 | Natural Questions | https://ai.google.com/research/NaturalQuestions | Google真实搜索问题 |
| 工具 | Elasticsearch BM25 | https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html | 工业级BM25实现 |

---

## 目录

1. [核心框架：QA、IR 与 RAG 的关系](#1-核心框架)
2. [QA 的本质：序列预测的特例](#2-qa的本质)
3. [RAG 的动机：为什么 LLM 需要检索](#3-rag动机)
4. [Information Retrieval 基础](#4-信息检索基础)
5. [TF-IDF：经典词项权重算法](#5-tf-idf)
6. [BM25：TF-IDF 的进化版](#6-bm25)
7. [Dense Vector Retrieval：语义检索](#7-密集向量检索)
8. [RAG 架构详解](#8-rag架构)
9. [Question Answering 系统](#9-问答系统)
10. [BERT for QA](#10-bert-for-qa)
11. [评估指标](#11-评估指标)
12. [QA 数据集](#12-数据集)
13. [三视角职业分析](#13-职业分析)
14. [面试高频题](#14-面试题)

---

## 1. 核心框架

### 1.1 三大概念的关系

```
学习路径（教授原话的教学顺序）：

信息检索 IR
  ├── TF-IDF, BM25（稀疏检索）
  └── Dense Retrieval（密集检索）
    ↓ （检索是工具）
    
检索增强生成 RAG
  ├── 检索阶段（用 IR）
  └── 生成阶段（用 LLM）
    ↓ （RAG 是组合）
    
问答系统 QA
  ├── 使用 IR 检索相关文档
  ├── 使用 RAG 生成答案
  └── 评估：EM, F1, MRR
```

**教授的"即插即用"哲学**：先学 IR 和 RAG 这两个"组件"，再把它们"插入"到 QA 系统中。就像先学切菜（IR）、调味（RAG），最后才学做一道完整的菜（QA）。

### 1.2 本模块涵盖的范式

```
Question Answering (QA)
    │
    ├─ IR-based QA（基于信息检索）
    │   ├─ TF-IDF
    │   ├─ BM25
    │   └─ Dense Vector Retrieval
    │
    ├─ Knowledge-Based QA（基于知识库）
    │   └─ Entity Recognition & Linking
    │
    ├─ RAG（检索增强生成）
    │   ├─ Retrieval Stage
    │   └─ Generation Stage
    │
    └─ Direct LLM（直接问 LLM）
        └─ Pre-trained Knowledge（有局限性）
```

---

## 2. QA的本质

### 2.1 QA = 序列预测的特例

**教授的核心洞察**：QA 不是全新任务，就是已学过的"序列预测"在特定场景下的应用。

| 任务 | 数学形式 | 示例 |
|------|----------|------|
| 通用语言模型 | $P(\text{next sequence} \mid \text{given sequence})$ | 给"The cat sat on the" → 预测"mat" |
| 机器翻译 | $P(\text{target} \mid \text{source})$ | 给中文 → 输出英文 |
| **QA** | $P(\text{answer} \mid \text{question})$ | 给问题 → 输出答案 |

**本质相同！** 底层都是条件概率建模，都可以用 Transformer 架构。唯一区别：输入是问题，输出是答案。

### 2.2 QA 的正式定义

> **QA**：自然语言处理中的一项任务，目标是自动回答以自然语言提出的问题。

**数学表达**：
$$\hat{A} = \arg\max_{A} P(A \mid Q)$$

其中 $Q$ = 问题，$A$ = 答案，$\hat{A}$ = 最可能的答案。

### 2.3 问答类型

#### Factoid Questions（事实性问题）

**教授的定义**："factoid = 简单事实 / loose fact"

```
问题："Where is the Louvre Museum located?"

多个"正确"答案（信息量递减）：
① "Rue de Rivoli, 75001 Paris, France"  ← 最有信息量
② "Paris, France"                        ← 中等
③ "Paris"                                ← 较少
④ "France"                               ← 很少
⑤ "Europe"                               ← 接近"空洞真理"
⑥ "On Earth"                             ← Vacuous truth（空洞真理）
```

**"Vacuous truth"（空洞真理）**：技术上正确，但信息量为零。

**历史背景**：
- **1961年**：Baseball QA System — 能回答棒球统计数字问题（结构化数据，factoid答案）
- **2011年**：IBM Watson 在 Jeopardy! 击败人类冠军 — 早期 RAG 的里程碑

#### Non-Factoid Questions（非事实性问题）

| 类型 | 例子 | 答案特点 |
|------|------|----------|
| 解释型 | "Why did WWI start?" | 段落，多个原因 |
| 步骤型 | "How to make a cake?" | 步骤列表 |
| 分析型 | "What are the pros and cons of RAG?" | 需要推理 |

### 2.4 答案类型

| 类型 | 格式 | 技术方案 |
|------|------|----------|
| **Span Extraction** | 从文档中提取片段 | BERT QA |
| **Multiple Choice** | 从选项中选择 | 分类模型 |
| **Free-Form Text** | 自由生成文本 | Seq2Seq / LLM |

---

## 3. RAG动机

### 3.1 纯 LLM 的三大问题

**教授原话**："Simple prompting obviously leads to hallucination."

#### 问题一：Hallucination（幻觉）

**根本原因**：LLM 是概率模型，不是知识数据库。它不"知道"事实，只知道"什么样的文本序列概率高"。

```
问："2024年诺贝尔物理学奖得主是谁？"
LLM："Dr. John Smith，因其在量子计算领域的研究..." ← 完全编造！

为什么编造？
- LLM 不会说"我不知道"
- 它会生成"听起来最合理"的答案
- 即使这个答案是假的
```

**类比**：闭卷考试 — 学生不会的题目会"合理地猜"，而不是空着。

#### 问题二：数据过时

```
GPT-4 训练截止：2023年4月
现在：2026年

问："2025年发生了什么大事？"
→ LLM 完全不知道

为什么不持续更新？
- 重训一次 GPT-3 级别模型：数百万美元
- 时间：数周到数月
- 数据收集+清洗：海量人工
→ 不可能每天/每周更新
```

#### 问题三：无法访问私有数据

```
问："我们公司的休假政策是什么？"
LLM：不知道，因为公司手册不在训练数据里
→ 不可能为每家公司专门训练一个模型
```

### 3.2 RAG 的解决方案

**教授原话**："RAG = output augmented by info retrieved"

**核心思想**：不更新 LLM 参数，而是检索外部文档，让 LLM 基于事实回答。

```
更新 LLM：
  修改 1750 亿参数 → 成本数百万美元，时间数月

更新 RAG 文档库：
  添加新文档（4000 词）→ 成本约等于零，时间即时

→ RAG 是更经济的选择！
```

**类比对比**：

| 比喻 | LLM | RAG |
|------|-----|-----|
| 考试 | 闭卷：凭记忆答题 | 开卷：可以查书 |
| 员工 | 靠脑子记所有规定 | 随时查公司手册 |
| 医生 | 凭经验诊断 | 查最新医学文献 |

### 3.3 RAG 的优势总结

| 问题 | 纯 LLM | RAG |
|------|--------|-----|
| 幻觉 | ⚠️ 可能编造 | ✅ 基于真实文档 |
| 数据过时 | ❌ 有截止日期 | ✅ 实时更新文档库 |
| 私有数据 | ❌ 没训练过 | ✅ 检索私有文档 |
| 可追溯性 | ❌ 无法引用来源 | ✅ 可引用源文档 |
| 更新成本 | 💰💰💰 重训 | 💰 更新文档库 |

---

## 4. 信息检索基础

### 4.1 核心术语

| 术语 | 定义 | 数学表示 | 例子 |
|------|------|----------|------|
| **Document** | 系统索引和检索的文本单元 | $d = \{t_1, t_2, ..., t_n\}$ | 网页、论文、段落 |
| **Collection** | 用于满足用户请求的文档集 | $C = \{d_1, d_2, ..., d_n\}$ | Wikipedia、公司文档库 |
| **Query** | 用户的信息需求，表示为词项 | $Q = \{t_{q1}, ..., t_{qm}\}$ | "hotel Mumbai nice" |
| **Token/Term** | 文档或查询中的单词 | $t_{ij}$ = 文档 $i$ 第 $j$ 个词 | "hotel", "Paris" |

**关键洞察**：Query 和 Document 在数学上是一样的！都是 token 序列。这意味着可以用同样的方法处理它们（计算词频、向量化、计算相似度）。

### 4.2 Ad Hoc Retrieval（临时检索）

**定义**：用户提出任意查询，系统实时返回按相关性排序的文档。

```
IR 的目标：
给定查询 Q，从集合 C 中找到最相关的文档：
IR(Q, C) = {d₁, d₂, ..., dₖ}（按相关性降序排列）
```

**核心问题**：如何计算相关性 relevance(d, Q)？这就是 TF-IDF、BM25、Dense Retrieval 要解决的问题。

### 4.3 倒排索引（Inverted Index）

高效检索的关键数据结构：

```
文档集合：
  Doc 1: "the cat sat on the mat"
  Doc 2: "the dog sat on the log"
  Doc 3: "cats and dogs"

倒排索引：
Term    │ Posting List（文档ID, 词频, 位置）
────────┼──────────────────────────────────
cat     │ (1, 1, [2])
cats    │ (3, 1, [1])
dog     │ (2, 1, [2])
sat     │ (1, 1, [3]), (2, 1, [3])
the     │ (1, 2, [1,5]), (2, 2, [1,5])

查询 "cat sat"：
  cat → Doc 1
  sat → Doc 1, Doc 2
  交集 → Doc 1 ✓
```

### 4.4 词汇不匹配问题（Vocabulary Mismatch）

**场景**：
- 查询："How long do cats live?"
- 文档A：包含 "cat"，但不回答问题
- 文档B：包含 "feline"，回答了问题，但不含 "cat"

**TF-IDF/BM25 的限制**：只匹配精确词项，无法理解语义相似性（"cat" ≈ "feline"）。

**解决方案**：Dense Retrieval（见第7节）。

---

## 5. TF-IDF

### 5.1 核心思想

> **TF-IDF 哲学**："重要的词 = 在文档中频繁出现（高TF）+ 在集合中稀有（高IDF）"

教授评价："sadly becoming a little outdated, but still useful" — 经典算法，理解它有助于理解更先进的方法。

### 5.2 Term Frequency (TF)

$$\text{TF}(t, d) = \begin{cases} 1 + \log_{10}(\text{count}(t, d)) & \text{if count}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

**为什么用 log？降低极端值影响。**

```
对比（词"hotel"）：

文档A出现10次，文档B出现100次：
不用log：B的权重是A的 10 倍
用log：
  TF_A = 1 + log(10) = 2.0
  TF_B = 1 + log(100) = 3.0
  B 只是 A 的 1.5 倍 ← 更合理！

直觉：词项相关性按"数量级"缩放，而不是线性缩放
```

### 5.3 Document Frequency & IDF

**DF(t)** = 包含词项 t 的文档数量

$$\text{IDF}(t) = \log_{10}\left(\frac{N}{\text{DF}(t)}\right)$$

| 词项 | DF（1000篇文档中）| IDF | 解释 |
|------|---------|-----|------|
| "the" | 950 | 0.02 | 几乎没有区分度 |
| "Paris" | 50 | 1.30 | 有一定区分度 |
| "Louvre" | 5 | 2.30 | 高度区分度 |

### 5.4 TF-IDF 组合

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) = \left(1 + \log_{10}(\text{count}(t, d))\right) \times \log_{10}\left(\frac{N}{\text{DF}(t)}\right)$$

**完整评分**（查询 $Q$，文档 $d$）：
$$\text{Score}(d, Q) = \sum_{t \in Q} \text{TF-IDF}(t, d)$$

### 5.5 余弦相似度

将查询和文档表示为向量，计算余弦相似度：

$$\text{cosine}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \times |\vec{d}|} \in [0, 1]$$

### 5.6 完整计算示例

**查询**："sweet love"，$N=100$ 个文档，"sweet" 在 10 个文档中，"love" 在 50 个文档中。

```
IDF("sweet") = log(100/10) = 1.0
IDF("love")  = log(100/50) = 0.30

文档1（"sweet" 出现5次，"love" 出现3次）：
TF("sweet") = 1 + log(5) = 1.70    →  TF-IDF = 1.70 × 1.0 = 1.70
TF("love")  = 1 + log(3) = 1.48    →  TF-IDF = 1.48 × 0.30 = 0.44
文档1总分 = 1.70 + 0.44 = 2.14 ✓
```

---

## 6. BM25

### 6.1 TF-IDF 的两大问题

#### 问题一：词频无限增长

```
TF-IDF 中：
词出现 10 次 → TF = 2.0
词出现 100 次 → TF = 3.0
词出现 1000 次 → TF = 4.0
→ 权重一直增长，容易被"关键词堆砌"作弊
```

#### 问题二：文档长度不公平

```
查询："hotel Mumbai"

文档A（短，50词）："hotel" 出现 2 次，密度 4%
文档B（长，500词）："hotel" 出现 5 次，密度 1%

TF-IDF：文档B得分更高 ← 不公平！
实际上文档A更聚焦
```

### 6.2 BM25 完整公式

$$\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \times \frac{\text{TF}(t, d) \times (k_1 + 1)}{\text{TF}(t, d) + k_1 \times \left(1 - b + b \times \frac{|d|}{\text{avgdl}}\right)}$$

其中：
- $k_1 \in [1.2, 2.0]$：词频饱和度控制参数
- $b = 0.75$：文档长度归一化参数
- $|d|$：文档长度，$\text{avgdl}$：平均文档长度

**BM25 的 IDF**（与 TF-IDF 略有不同）：

$$\text{IDF}(t) = \log\left(\frac{N - n(t) + 0.5}{n(t) + 0.5}\right)$$

### 6.3 核心改进一：词频饱和

词频饱和项：$\dfrac{\text{TF} \times (k_1 + 1)}{\text{TF} + k_1}$

| TF | 权重（$k_1=1.2$）| 增长率 |
|----|---------|--------|
| 1  | 1.00    | -      |
| 2  | 1.38    | +38%   |
| 5  | 1.77    | +28%   |
| 10 | 1.96    | +11%   |
| 50 | 2.15    | +2%    |
| 100| 2.17    | +1%    |

**直觉类比（教授式）**：
```
吃饭的满足感（饱和效应）：
- 第1碗：非常满足 ✅✅✅
- 第2碗：还不错 ✅✅
- 第3碗：有点饱了 ✅
- 第10碗：完全饱了（饱和）

→ 词频高过某个阈值后，继续增加也不会大幅提升相关性分数
```

### 6.4 核心改进二：文档长度归一化

长度归一化项：$k_1 \times \left(1 - b + b \times \frac{|d|}{\text{avgdl}}\right)$（假设 $k_1=1.2, b=0.75, \text{avgdl}=100$）

| 文档长度 | 归一化项 | 效果 |
|----------|---------|------|
| 50词（短）| 0.75 | 分母减小 → 分数提升 |
| 100词（平均）| 1.2 | 标准情况 |
| 200词（长）| 2.1 | 分母增大 → 分数降低 |

### 6.5 完整对比示例

**场景**：查询 "hotel Mumbai"，$k_1=1.2, b=0.75, \text{avgdl}=150$

| 方法 | 文档A（短50词，hotel×2, Mumbai×1）| 文档B（长300词，hotel×10, Mumbai×5）| 差距 |
|------|-----|-----|-----|
| **TF-IDF** | 3.94 | 6.41 | B 高 63% ⚠️ 不公平 |
| **BM25** | 5.29 | 5.84 | B 高 10% ✅ 公平 |

### 6.6 参数选择指南

| 参数 | 值 | 效果 | 适用场景 |
|------|-----|------|----------|
| $k_1 = 0$ | 完全忽略词频 | 所有词等权 | 极短文档 |
| $k_1 = 1.2$ | 适度饱和 | 通用 | 搜索引擎 |
| $k_1 = 2.0$ | 较慢饱和 | 词频更重要 | 学术论文 |
| $b = 0$ | 不归一化 | 长文档占优 | 长度一致的场景 |
| $b = 0.75$ | 适度归一化 | 通用 | 大多数场景 |
| $b = 1$ | 完全归一化 | 只看密度 | 长度差异极大 |

### 6.7 Python 实现

```python
import math

def bm25_score(tf, df, N, doc_len, avg_doc_len, k1=1.2, b=0.75):
    """
    计算 BM25 单个词项的得分
    
    参数：
    - tf: 词项在文档中的频率
    - df: 包含该词项的文档数
    - N: 集合中文档总数
    - doc_len: 当前文档长度
    - avg_doc_len: 平均文档长度
    """
    # IDF（BM25 版本）
    idf = math.log((N - df + 0.5) / (df + 0.5))
    
    # 词频饱和 + 长度归一化
    tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    
    return idf * tf_norm

def bm25_document_score(query_terms, document, collection_stats):
    """对整个查询计算文档得分"""
    total_score = 0
    for term in query_terms:
        if term in document:
            score = bm25_score(
                tf=document[term]['count'],
                df=collection_stats[term]['df'],
                N=collection_stats['N'],
                doc_len=document['length'],
                avg_doc_len=collection_stats['avg_doc_len']
            )
            total_score += score
    return total_score

# Elasticsearch 中使用 BM25（默认）
# 无需配置，match 查询默认使用 BM25
```

---

## 7. 密集向量检索

### 7.1 动机：解决词汇不匹配

**问题**：TF-IDF 和 BM25 只匹配精确词项。
- 查询："eye doctor" 无法匹配文档中的 "oculist"
- 查询："automobile" 无法匹配文档中的 "car"

**解决方案**：用向量语义表示词语，语义相似的词在向量空间中接近。

### 7.2 架构对比

#### Bi-Encoder（双编码器）

```
Query → [Encoder] → Query Vector (768-dim)
Document → [Encoder] → Doc Vector (768-dim)

Score = Dot Product(Query Vector, Doc Vector)
```

**优势**：文档可预先编码，查询时只需计算点积，**快速**  
**劣势**：查询-文档交互有限，准确性略低

#### Cross-Encoder（交叉编码器）

```
[CLS] Query [SEP] Document [SEP] → [BERT] → [CLS] token → Score
```

**优势**：Query-Document 联合建模，交互丰富，**准确**  
**劣势**：每对都要重新编码，**计算成本高**

#### ColBERT：两全其美

**核心创新**：Late Interaction（延迟交互）

```python
# Step 1: 分别编码（保留效率）
query_embs = BERT(query)   # [q_len, dim]
doc_embs = BERT(document)  # [d_len, dim]

# Step 2: 词项级别最大相似度（保留准确性）
score = sum(max(dot(q_tok, d_tok) for d_tok in doc_embs)
            for q_tok in query_embs)
```

**公式**：
$$\text{Score}(q, d) = \sum_{t_q \in q} \max_{t_d \in d} \text{emb}(t_q) \cdot \text{emb}(t_d)$$

**示例**：
```
Query: "eye doctor"
Doc:   "oculist examine eyes"

max_sim("eye")    = sim("eye", "eyes")    = 0.9
max_sim("doctor") = sim("doctor", "oculist") = 0.8

Score = 0.9 + 0.8 = 1.7 ✓
```

### 7.3 稀疏 vs 密集检索对比

| 维度 | TF-IDF / BM25 | Dense Retrieval |
|------|--------------|-----------------|
| 词汇匹配 | 精确匹配 | 语义匹配 |
| 同义词处理 | ❌ | ✅ |
| 计算速度 | 极快（倒排索引）| 较慢 |
| 训练需求 | 无需训练 | 需要大量标注对 |
| 可解释性 | ✅ 高 | ❌ 低 |
| 实践效果 | 强基线 | 通常更好 |
| 适合场景 | 关键词搜索 | 语义理解 |

---

## 8. RAG架构

### 8.1 完整 RAG 流程

```
用户查询: "Where is the Louvre Museum?"
    │
    ▼
┌─────────────────────────────────────────┐
│  RETRIEVAL STAGE（检索阶段）             │
│                                         │
│  1. Query Encoding：将问题向量化         │
│  2. Document Retrieval：搜索文档库       │
│     → BM25 / Dense Retrieval           │
│  3. Top-K Documents 选取               │
└─────────────────────────────────────────┘
    │
    ▼ Retrieved Docs:
    Doc 1: "The Louvre is in Paris, France..."
    Doc 2: "Located on Rue de Rivoli..."
    │
    ▼
┌─────────────────────────────────────────┐
│  GENERATION STAGE（生成阶段）            │
│                                         │
│  Prompt:                                │
│  "Based on these documents:             │
│   [Doc 1][Doc 2]                       │
│   Answer: Where is the Louvre?"        │
│                                         │
│  LLM → 生成答案                         │
└─────────────────────────────────────────┘
    │
    ▼
"The Louvre Museum is located in Paris, France, on Rue de Rivoli."
```

**教授的精炼定义**：

> **Generation** = 输出最可能的文本  
> **Augmented** = 该文本被检索信息增强  
> **Retrieval** = 用 IR 找到的相关信息  
> **RAG 核心** = "不创造答案，而是从真实文档中选取相关部分并填补空隙"

### 8.2 RAG 的数学形式

**从 LLM 到 RAG**：

$$P(\text{answer} \mid \text{question}) \rightarrow P(\text{answer} \mid \text{question}, \underbrace{D_1, D_2, ..., D_k}_{\text{检索文档}})$$

**生成过程**（逐词）：

$$P(A \mid Q, D) = \prod_{i=1}^{|A|} P(a_i \mid Q, D, a_{<i})$$

### 8.3 高级 RAG 技术

#### Multi-Hop Retrieval（多跳检索）

**场景**：回答复杂问题需要多个文档的信息。

```
问题："Who is the spouse of the director of Inception?"

Hop 1: 检索 → "Christopher Nolan directed Inception"
Hop 2: 检索 → "Christopher Nolan married Emma Thomas"

答案: "Emma Thomas"
```

#### Two-Stage Retrieval（两阶段检索）

```
Stage 1: Fast Retrieval（BM25）
  → 快速筛选 Top 1000 文档
Stage 2: Re-ranking（Dense Retrieval / Cross-Encoder）
  → 精确排序出 Top 10 文档
Stage 3: Generation
  → 基于 Top 10 生成答案

优势：
- Stage 1：高召回率，快速
- Stage 2：高精确率，准确
- 兼顾速度和质量
```

### 8.4 完整 RAG Python 实现

```python
from transformers import AutoTokenizer, AutoModel
from langchain import OpenAI, RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ============ Step 1: 构建向量数据库 ============
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "The Louvre Museum is located in Paris, France.",
    "The Eiffel Tower is also in Paris.",
    "Paris is the capital of France.",
]

vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ============ Step 2: 构建 RAG 链 ============
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 将所有检索文档 "stuff" 进 prompt
    retriever=retriever,
    return_source_documents=True  # 返回来源文档
)

# ============ Step 3: 问答 ============
result = qa_chain({"query": "Where is the Louvre?"})
print("Answer:", result['result'])
print("Sources:", [d.page_content for d in result['source_documents']])
```

---

## 9. 问答系统

### 9.1 QA 的历史沿革

```
1961年: Baseball QA System
  - 输入：棒球统计相关问题
  - 回答：具体数字
  - 特点：结构化数据，factoid答案

2011年: IBM Watson 赢得 Jeopardy!
  - 击败人类冠军
  - 需要理解自然语言问题
  - 从大量文档检索 → 生成答案
  - 实质：早期 RAG 系统！

2018年: BERT 在 SQuAD 超越人类
  - Fine-tuned BERT on SQuAD
  - F1 = 93.2 (人类 = 91.2)

2020年: GPT-3 少样本问答
  - Few-shot learning
  - 不需要 fine-tuning

2023-现在: LLM + RAG
  - ChatGPT + Bing Search
  - Claude + web retrieval
```

### 9.2 QA 意图类型

| 类型 | 定义 | 例子 |
|------|------|------|
| **Natural Intent** | 真正想获取信息 | "今天巴黎天气？"（真的想知道）|
| **Probing Intent** | 测试系统能力 | "你知道巴黎天气吗？"（测试系统）|

---

## 10. BERT for QA

### 10.1 SQuAD 任务定义

**任务**：给定 (问题, 段落) 对，找到段落中回答问题的**连续文本片段（span）**。

```
Context: "The Louvre Museum is located in Paris, France, on the Right Bank of the Seine."
Question: "Where is the Louvre Museum?"

答案 Span: "Paris, France"（位置: 35-48）
```

### 10.2 BERT QA 架构

```
输入: [CLS] Question [SEP] Context [SEP]
      ↓
  [BERT Encoder]
      ↓
Token Embeddings: [h_CLS, h_Where, h_is, ..., h_Paris, h_France, ...]
      ↓
两个线性分类器（共享 BERT）：
  Start Classifier → 每个位置的起始分数
  End Classifier → 每个位置的结束分数
      ↓
预测 Span: from argmax(Start_logits) to argmax(End_logits)
```

### 10.3 训练目标

$$\mathcal{L} = -\log P(\text{start} = s^*) - \log P(\text{end} = e^*)$$

其中 $s^*$ 和 $e^*$ 是金标准起止位置。

### 10.4 完整代码实现

```python
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# 加载预训练 BERT QA 模型
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def predict_answer(question: str, context: str) -> str:
    """
    给定问题和上下文，预测答案 span
    """
    # Step 1: Tokenize（注意：question 在 context 前面）
    inputs = tokenizer(
        question, 
        context, 
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    # Step 2: 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits  # [1, seq_len]
    end_logits = outputs.end_logits      # [1, seq_len]
    
    # Step 3: 找到最佳 span
    # 确保 start <= end，且不在 question 部分
    sep_idx = inputs['input_ids'][0].tolist().index(tokenizer.sep_token_id)
    
    # 只在 context 部分搜索（sep_idx 之后）
    start_logits[0, :sep_idx+1] = -float('inf')
    end_logits[0, :sep_idx+1] = -float('inf')
    
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)
    
    # 如果 end < start，尝试找最优组合
    if end_idx < start_idx:
        # 找得分最高的合法对
        scores = start_logits[0].unsqueeze(1) + end_logits[0].unsqueeze(0)
        scores = torch.triu(scores)  # 只保留 end >= start 的组合
        best = torch.argmax(scores)
        start_idx = best // len(start_logits[0])
        end_idx = best % len(end_logits[0])
    
    # Step 4: 解码答案
    answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
    answer = tokenizer.decode(answer_tokens)
    
    return answer

# 测试
question = "Where is the Louvre Museum?"
context = "The Louvre Museum is located in Paris, France, on the Right Bank of the Seine."
print(predict_answer(question, context))  # → "Paris, France"
```

### 10.5 处理无答案问题（SQuAD 2.0）

SQuAD 2.0 包含约 50% 无法从段落回答的问题，模型需要学会"说不知道"：

```python
def predict_with_unanswerable(question, context, threshold=0.0):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    
    # 计算有答案的得分
    start_score = torch.max(outputs.start_logits)
    end_score = torch.max(outputs.end_logits)
    answer_score = (start_score + end_score).item()
    
    # CLS token 的得分代表"无答案"
    no_answer_score = outputs.start_logits[0][0] + outputs.end_logits[0][0]
    no_answer_score = no_answer_score.item()
    
    # 比较：有答案 vs 无答案
    if answer_score > no_answer_score + threshold:
        # 提取答案
        start_idx = torch.argmax(outputs.start_logits[0][1:]) + 1
        end_idx = torch.argmax(outputs.end_logits[0][1:]) + 1
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        return tokenizer.decode(answer_tokens)
    else:
        return "No answer found in the provided context."
```

---

## 11. 评估指标

### 11.1 UAS/LAS（依存解析，参考 Module 10）

本模块的评估指标主要是 EM、F1、MRR、MAP。

### 11.2 Exact Match (EM)

$$\text{EM} = \frac{\text{Number of Exact Matches}}{\text{Total Questions}}$$

```
Gold: "William Shakespeare"
Prediction: "William Shakespeare" → ✅ Match (EM=1)
Prediction: "Shakespeare"         → ❌ No Match (EM=0)
```

**缺点**：过于严格。"Shakespeare" 应该算对。

### 11.3 Token F1 Score（更宽松）

把答案当词集合，计算 F1：

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

```
Gold:       "William Shakespeare"  → {william, shakespeare}
Prediction: "Shakespeare"          → {shakespeare}

TP = 1 (shakespeare)
FP = 0
FN = 1 (william)

Precision = 1/1 = 1.0
Recall    = 1/2 = 0.5
F1        = 0.67 ← 比 EM 更合理
```

### 11.4 Precision & Recall（IR 评估）

$$\text{Precision} = \frac{|\text{相关且被检索}|}{|\text{被检索}|} \quad \text{Recall} = \frac{|\text{相关且被检索}|}{|\text{所有相关}|}$$

**P-R 权衡**：
- 高 Precision：检索结果大多相关，但可能遗漏相关文档
- 高 Recall：覆盖了大多数相关文档，但包含不相关文档

### 11.5 Mean Reciprocal Rank (MRR)

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

```
Query 1: 正确答案在第 1 位 → 1/1 = 1.0
Query 2: 正确答案在第 3 位 → 1/3 = 0.33
Query 3: 正确答案在第 2 位 → 1/2 = 0.5

MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
```

**直觉**：奖励把正确答案排得越靠前的系统。

### 11.6 Mean Average Precision (MAP)

$$\text{AP} = \frac{1}{R} \sum_{k=1}^{n} P(k) \times \text{rel}(k) \quad \text{MAP} = \frac{1}{|Q|} \sum_{q} \text{AP}(q)$$

```
检索结果: [Rel, NotRel, Rel, Rel, NotRel]

P@1 = 1/1 = 1.0, rel=1
P@2 = 1/2 = 0.5, rel=0
P@3 = 2/3 = 0.67, rel=1
P@4 = 3/4 = 0.75, rel=1
P@5 = 3/5 = 0.6, rel=0

AP = (1.0×1 + 0.67×1 + 0.75×1) / 3 = 0.81
```

### 11.7 评估指标汇总

| 指标 | 任务 | 特点 |
|------|------|------|
| **EM** | QA | 严格精确匹配 |
| **Token F1** | QA | 词级别 overlap |
| **Precision/Recall** | IR | 相关文档检索率 |
| **MRR** | IR/QA | 第一个正确答案的排名 |
| **MAP** | IR | 考虑排序的综合精确率 |

---

## 12. 数据集

### 12.1 主要 QA 数据集

| 数据集 | 任务类型 | 规模 | 特点 |
|--------|----------|------|------|
| **SQuAD v1.1** | Span Extraction | 100k+ | 维基百科段落抽取，答案必须在文中 |
| **SQuAD v2.0** | Span + Unanswerable | 150k+ | 含~50%无法回答的问题 |
| **Natural Questions** | Open-Domain | 300k+ | 真实 Google 搜索查询 |
| **MS MARCO** | Passage Ranking | 1M+ | 真实 Bing 查询，自由文本答案 |
| **TriviaQA** | Open-Domain | 95k+ | 琐事问答，有证据文档 |
| **HotpotQA** | Multi-Hop | 113k+ | 需要多文档推理，含支撑事实标注 |

### 12.2 SQuAD 数据格式

```json
{
  "context": "The Louvre Museum is located in Paris, France, on the Right Bank of the Seine.",
  "question": "Where is the Louvre Museum?",
  "answers": [
    {
      "text": "Paris, France",
      "answer_start": 35
    }
  ]
}
```

### 12.3 Natural Questions 格式

```json
{
  "question": "when was the last time the eagles won the super bowl",
  "short_answer": "2018",
  "long_answer": "The Philadelphia Eagles won Super Bowl LII on February 4, 2018..."
}
```

---

## 13. 职业分析

### 13.1 AI 算法工程师视角

**核心工程问题**：

1. **速度 vs 准确率**
   - 实时系统（客服机器人）：BM25 + Bi-Encoder，<50ms
   - 高准确率（法律/医疗）：Cross-Encoder + GPT-4
   - 生产推荐：两阶段（BM25 粗筛 → Dense 精排）

2. **幻觉控制**
   - RAG 的 Retrieved context 必须足够相关
   - 加入 "I don't know" fallback（confidence 低时拒答）
   - 用 NLI 模型验证答案是否与文档一致

3. **评估管道建议**
   - 开发阶段：EM + F1 on held-out set
   - 生产监控：人工评估样本（每周抽样）
   - A/B test：MRR/MAP 在真实用户流量上

4. **技术栈**
   - 检索：Elasticsearch (BM25) + FAISS (Dense)
   - 框架：LangChain + HuggingFace Transformers
   - 模型：BERT/RoBERTa (Span QA) + GPT-4 (Generation)

### 13.2 AI 产品经理视角

**用户价值翻译**：

| 技术能力 | 产品形态 | 用户价值 |
|----------|----------|----------|
| BM25 检索 | 企业文档搜索 | 精准找到规章制度、历史记录 |
| RAG + LLM | 智能客服机器人 | 24/7 精准回答，可引用来源 |
| BERT QA | 合同审查工具 | 自动定位合同中的关键条款 |
| Multi-hop QA | 复杂研究助手 | 综合多个文档回答复杂问题 |

**何时需要 RAG？决策框架**：
```
问题类型：
  事实查询 + 需要可信来源 → 一定需要 RAG
  创意生成 + 无需事实 → 不需要 RAG
  私有数据查询 → 必须 RAG
  实时信息（新闻/股票）→ RAG + 实时数据源
```

**ROI 评估**：
- 构建 RAG 系统：文档索引 + pipeline 搭建 ≈ 1-2周工程时间
- 效果提升：幻觉率降低 60-80%（需 A/B 验证）
- 适合高价值场景：法律、医疗、金融、企业内部知识管理

### 13.3 创业者视角

**市场机会分析**：

1. **企业知识管理**（最大市场）
   - 痛点：员工找不到内部文档，每次咨询 IT/HR 浪费时间
   - 解决方案：私有化部署的 RAG 系统
   - 竞争：Notion AI、Confluence AI（开放市场，仍有差异化空间）
   - 差异化：垂直行业专业化（法律事务所、医院、制造企业）

2. **多语言问答**
   - 痛点：非英语市场 QA 准确率显著低于英语
   - 机会：中文、阿拉伯语、印地语等的专业化 QA 引擎
   - 技术路径：mBERT/XLM-RoBERTa + UD 数据

3. **教育 QA**
   - 趋势：AI Tutor 市场快速增长
   - 技术核心：Socratic questioning（问而不直接答）+ RAG from textbook
   - 数据护城河：优质教育内容的独家授权

---

## 14. 面试题

### 基础概念

**Q1: 解释 BM25 比 TF-IDF 好在哪里，举例说明。**

> **答**：BM25 有两个核心改进：
> 
> ① **词频饱和**：TF-IDF 的词频项随次数线性/对数增长，BM25 通过 $k_1$ 参数实现饱和效应——词出现 10 次和 100 次，BM25 给出的分数差距很小，防止"关键词堆砌"作弊。
> 
> ② **文档长度归一化**：TF-IDF 偏袒长文档（词绝对出现次数多），BM25 通过 $b$ 参数考虑词密度而非绝对词频，短而精的文档可以获得高分。
> 
> 例子：查询"hotel Mumbai"。500词文档中 hotel 出现5次 vs 50词文档中 hotel 出现2次——TF-IDF 给长文档打 6.41，短文档 3.94（差 63%）；BM25 给长文档 5.84，短文档 5.29（差 10%），更公平。

**Q2: 解释 RAG 的三个关键步骤，以及它如何解决 LLM 幻觉问题。**

> **答**：RAG = Retrieval + Augmented + Generation
> 
> ① **Retrieval**：用 IR（TF-IDF/BM25/Dense）从文档库检索与问题相关的 Top-K 文档。
> ② **Augmented**：将检索文档加入 LLM 的 prompt，作为回答的依据。
> ③ **Generation**：LLM 基于文档生成答案，而非凭参数记忆猜测。
> 
> 幻觉问题的根源：LLM 是概率模型，不确定时会生成"听起来合理"但错误的内容。RAG 通过"开卷考试"方式让 LLM 参考真实文档，答案有依据可引用，大幅降低幻觉率。同时还解决了数据过时和无法访问私有数据的问题。

**Q3: 什么是 Factoid Question？为什么有"Vacuous Truth"问题？**

> **答**：Factoid Question 是寻求简单事实的问题，答案通常是简短的实体或数值（"Where is the Louvre?" → "Paris"）。
> 
> Vacuous Truth 问题：同一个 factoid 问题可以有多个技术上正确但信息量不同的答案。"卢浮宫在哪里？"—— "法国"、"欧洲"、"地球上"技术上都正确，但信息量极低，是"空洞真理"。QA 系统不仅要追求"正确"，还要提供有足够信息量的答案。这也是为什么 Token F1 比 Exact Match 更合理：部分正确也有价值。

### 技术实现

**Q4: 如何用 spaCy 做 dependency parsing 来增强 QA 的答案提取？**

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_answer_entities(question: str, answer_candidate: str):
    """
    用依存分析辅助验证答案是否回应了问题中的 wh-word
    """
    q_doc = nlp(question)
    a_doc = nlp(answer_candidate)
    
    # 识别问题类型
    question_word = None
    for token in q_doc:
        if token.dep_ == "advmod" and token.text.lower() in ["where", "when", "why", "how"]:
            question_word = token.text.lower()
        elif token.dep_ == "nsubj" and token.text.lower() == "who":
            question_word = "who"
    
    # 根据问题类型验证答案
    if question_word == "where":
        # 答案应该包含 GPE (地名) 或 LOC 实体
        locations = [ent.text for ent in a_doc.ents if ent.label_ in ["GPE", "LOC"]]
        return locations if locations else None
    elif question_word == "who":
        # 答案应该包含 PERSON 或 ORG 实体
        persons = [ent.text for ent in a_doc.ents if ent.label_ in ["PERSON", "ORG"]]
        return persons if persons else None
    
    return answer_candidate
```

**Q5: 为什么 Cross-Encoder 比 Bi-Encoder 更准确，但实际系统中通常用 Bi-Encoder？**

> **答**：
> 
> **Cross-Encoder 更准确**：联合编码 (question, document) 对，Query 每个 token 都能 attend 到 Document 的每个 token（全交互），信息更充分。
> 
> **Bi-Encoder 更实用**：文档可以预先离线编码并存入向量数据库（FAISS）。查询时只需编码 query（~10ms），然后用向量近邻搜索（ANN）快速找到相似文档。Cross-Encoder 需要对每个 (query, doc) 对重新前向传播，100 万文档 = 100 万次前向传播，无法实时。
> 
> **工业实践**：两阶段 pipeline：Bi-Encoder 快速筛选 Top 1000 → Cross-Encoder 精细排序 Top 10 → LLM 生成。

**Q6: 在 SQuAD 上，BERT 是怎么处理答案不在文本中的情况（SQuAD 2.0）？**

> **答**：SQuAD 2.0 中约 50% 的问题无法从段落中找到答案。BERT 的处理方式：
> 
> 在输入序列 `[CLS] Question [SEP] Context [SEP]` 中，`[CLS]` token 被特殊处理：如果答案不存在，则最大概率的 start/end 位置应该都落在 `[CLS]` 上（代表"无答案"）。
> 
> 训练时对无答案问题，gold start = gold end = 0（CLS 位置）。
> 
> 推断时，比较 "有答案最优分数" vs "`[CLS]` 的无答案分数"，用阈值决定是否回答。

---

## 总结：知识图谱

```
Module 11: QA + IR + RAG
│
├── 信息检索 (IR)
│   ├── 基础：Document, Collection, Query, 倒排索引
│   ├── 稀疏：TF-IDF → BM25（饱和+归一化）
│   └── 密集：Bi-Encoder → Cross-Encoder → ColBERT（Late Interaction）
│
├── 检索增强生成 (RAG)
│   ├── 动机：解决 LLM 幻觉/过时/私有数据问题
│   ├── 流程：Retrieve → Augment → Generate
│   └── 进阶：Multi-hop, Two-stage, Self-RAG
│
├── 问答系统 (QA)
│   ├── 本质：P(answer | question)（序列预测特例）
│   ├── 类型：Factoid vs Non-factoid
│   ├── BERT QA：Span Extraction（start/end 分类器）
│   └── 历史：1961 Baseball → 2011 Watson → 2018 BERT → 2023 RAG+LLM
│
└── 评估
    ├── QA：EM（精确匹配）、Token F1
    └── IR：Precision/Recall、MRR、MAP
```

**与其他模块的联系**：
- [[Module09_Constituency_Parsing]] / [[Module_10_Dependency_Parsing]]：句法分析可以辅助 QA 系统理解问题结构（识别 wh-word 对应的依存关系）
- [[Module_08_Machine_Translation]] / [[Module_07_Deep_Learning]]：NER 标注帮助 QA 验证答案类型是否匹配问题类型（who→PERSON, where→GPE）
- [[Module_07_Deep_Learning]]：BERT、GPT 是 QA 和 RAG 的核心组件
