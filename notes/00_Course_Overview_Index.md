# 🧠 CS 584 NLP — 职业导向知识库
## 知识编译器索引 v1.0

> **课程**：CS 584 Natural Language Processing — Stevens Institute of Technology
> **教授**：Justo Karell | **学期**：Spring 2026
> **目标岗位**：AI 算法工程师 × AI Native 产品经理（双视角）
> **知识库创建**：2026-04-15 | **编译工具**：Claude（Karpathy 知识编译器范式）

---

## 📁 知识库文件结构

```
CS584_NLP_Notes/
├── 00_Course_Overview_Index.md          ← 你在这里（索引 + 全局视角）
├── Module_02_ML_Basics.md               ✅ 已完成
├── Module_06_Neural_Networks.md         ✅ 已完成
├── Module_07_Deep_Learning.md           ✅ 已完成（⭐⭐⭐ 最重要）
├── Module_08_Machine_Translation.md     ✅ 已完成
├── Module_09_CFG_Parsing.md             ✅ 已完成
├── Module_10_Dependency_Parsing.md      ✅ 已完成
├── Module_11_QA_IR_RAG.md              ✅ 已完成
├── Module_12_Dialogue_Systems.md        ✅ 已完成
├── Module_13_ASR.md                     ✅ 已完成
│
├── Module_01_NLP_Intro.md               ⏳ 待编译
├── Module_03_Language_Representation.md ⏳ 待编译
├── Module_04_Language_Models.md         ⏳ 待编译
└── Module_05_Vector_Semantics.md        ⏳ 待编译
```

---

## 🗺️ 全课程技术演进地图

```
基础层（Module 1-2）
├── NLP 是什么，为什么难
└── ML 基础：损失函数、梯度下降、过拟合

表示层（Module 3-5）
├── 词的表示：Token/Type/Corpus
├── 语言模型：N-gram → 神经 LM
└── 向量语义：Word2Vec、GloVe、词嵌入

建模层（Module 6-8）        ← 技术核心
├── 神经网络基础 + 序列标注
├── 深度学习：CNN / RNN / LSTM / Transformer / BERT / GPT
└── 机器翻译：SMT → Seq2Seq → Attention → Transformer

结构层（Module 9-10）
├── CFG：形式文法、CKY 算法
└── 依存句法：词间关系分析

应用层（Module 11-13）
├── 问答系统：检索 + 阅读理解
├── 对话系统：任务型 + 闲聊型 + RL 优化
└── 语音识别：ASR 基础
```

---

## 🎯 职业目标导向学习优先级

### AI 算法工程师（优先级排序）

| 优先级 | 模块 | 核心技能 |
|---|---|---|
| ⭐⭐⭐ 必掌握 | Module 7 | Transformer / BERT / GPT 架构，能手推 Attention |
| ⭐⭐⭐ 必掌握 | Module 2 | 梯度下降、过拟合、Loss 函数，ML 基础面试 |
| ⭐⭐ 重要 | Module 8 | Seq2Seq + Attention，Beam Search |
| ⭐⭐ 重要 | Module 5 | Word2Vec、词嵌入，NLP 表示基础 |
| ⭐⭐ 重要 | Module 6 | RNN/LSTM，序列标注（NER）|
| ⭐ 了解 | Module 9 | CKY 算法思路，依存 vs 成分句法 |
| ⭐ 了解 | Module 12 | RL 对话管理，DQN 应用 |

### AI Native 产品经理（优先级排序）

| 优先级 | 模块 | 核心知识 |
|---|---|---|
| ⭐⭐⭐ 必掌握 | Module 7 | BERT vs GPT 选型，预训练+微调范式 |
| ⭐⭐⭐ 必掌握 | Module 12 | 对话系统架构，槽填充设计，评估指标 |
| ⭐⭐ 重要 | Module 2 | 损失函数=业务目标，过拟合的产品影响 |
| ⭐⭐ 重要 | Module 8 | 机器翻译挑战，偏见与伦理 |
| ⭐⭐ 重要 | Module 4 | 语言模型，理解 GPT 的工作原理 |
| ⭐ 了解 | Module 1 | NLP 应用全景，六大任务 |
| ⭐ 了解 | Module 9 | 结构歧义，为什么 NLP 系统有时出错 |

---

## 💼 面试题全库（按重要度）

### Tier 1：必须能流畅回答

- **"请解释一下 Transformer 的架构"**（算法/产品都会问）
- **"BERT 和 GPT 的区别？什么时候用哪个？"**
- **"什么是过拟合？如何解决？"**
- **"Attention 机制是什么？为什么比 RNN 好？"**
- **"如何设计一个对话机器人的评估指标？"**（产品岗）

### Tier 2：需要准备

- **"从 MLE 推导出 Cross-Entropy Loss"**（算法岗）
- **"SMT 为什么用 P(s|t)·P(t) 而不直接用 P(t|s)？"**（算法岗）
- **"LSTM 如何解决梯度消失？"**（算法岗）
- **"如何判断模型是否该上线？"**（产品岗）
- **"训练集 99%，测试集 60%，怎么办？"**（两者都会问）

### Tier 3：加分项

- **"什么是 LoRA？为什么比全量微调好？"**
- **"KV Cache 是什么？怎么加速推理？"**
- **"Beam Search 的长度归一化为什么重要？"**
- **"深度强化学习如何优化对话策略？"**

---

## 📊 知识库进度追踪

| 模块 | 主题 | 笔记状态 | 职业重要度 |
|---|---|---|---|
| Course Overview | 课程总览 | ✅ 已编译（此文件）| — |
| Module 1 | NLP 简介 | ⏳ 待编译 | ⭐ |
| Module 2 | ML 基础 | ✅ 已编译 | ⭐⭐⭐ |
| Module 3 | 语言表示 | ⏳ 待编译 | ⭐⭐ |
| Module 4 | 语言模型 | ⏳ 待编译 | ⭐⭐ |
| Module 5 | 向量语义 | ⏳ 待编译 | ⭐⭐ |
| Module 6 | 神经网络与序列标注 | ✅ 已编译 | ⭐⭐ |
| Module 7 | 深度学习架构 | ✅ 已编译 | ⭐⭐⭐ |
| Module 8 | 机器翻译 | ✅ 已编译 | ⭐⭐ |
| Module 9 | CFG/句法分析 | ✅ 已编译 | ⭐ |
| Module 10 | 依存句法 | ✅ 已编译 | ⭐ |
| Module 11 | 问答/IR/RAG | ✅ 已编译 | ⭐⭐ |
| Module 12 | 对话系统 | ✅ 已编译 | ⭐⭐ |
| Module 13 | 语音识别 | ✅ 已编译 | ⭐ |

**总体进度**：10/14 已编译（71%）

---

## 🔄 知识库更新规则

### 新模块笔记来了
→ 发给 Claude，按双视角模板编译，写入新 .md 文件，更新此索引进度

### 你提问了一个概念
→ Claude 回答 + 判断是否需要补充到对应模块文件

### 发现笔记有误或不够清楚
→ 告诉 Claude：「Module X 的 Y 部分需要更新」，Claude 编辑对应文件

---

## 📅 建议学习计划

### 第一周（建立框架）
- [ ] 读 00_Course_Overview_Index.md（此文件）
- [ ] 精读 Module_07_Deep_Learning.md（最重要）
- [ ] 精读 Module_02_ML_Basics.md

### 第二周（核心技术）
- [ ] 精读 Module_08_Machine_Translation.md
- [ ] 等待 Module_04/05/06 编译完成后阅读

### 第三周（应用落地）
- [ ] 精读 Module_12_Dialogue_Systems.md
- [ ] 等待 Module_11 编译完成后阅读

### 面试前两天
- [ ] 过一遍所有模块的"双视角面试题库"
- [ ] 能流畅背出所有 Tier 1 面试题
- [ ] 复习核心公式速查卡

---

**知识库版本**：v1.0 | **下次更新**：收到新模块笔记时
