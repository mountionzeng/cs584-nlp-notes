# 🧠 CS 584 NLP — Module 9: CFG 与成分句法分析
## 职业导向笔记 v1.0

> **编译自**：Module 9 原始笔记（CFG + CNF + CKY + PARSEVAL + Span-Based Parsing）
> **目标岗位**：AI 算法工程师 × AI Native 产品经理（双视角）
> **地位**：⭐ 偏学术/理论，产品岗概念性了解，算法岗中等频率面试
> **最后更新**：2026-04-15

---

## 🗺️ 本模块地图

```
形式语言基础 → 成分（Constituency） → CFG 定义 → CNF 转换 → CKY 算法 → 评估
                                                                    ↓
                                                          PARSEVAL 指标
                                                                    ↓
                                                       现代：Span-Based Parsing
```

---

## 📌 Chapter 1：形式语言与元语言基础

### 核心概念层次

```
字母表 Σ（有限符号集合）
    ↓ 形态学
语言 L（符号串的集合）
    ↓ 文法
句子（合法的符号序列）
```

### 元语言（Metalanguage）

> **元语言是用来描述语言本身的语言。**

```
"verb" 这个词：
  → 作为普通词：英语中的一个词
  → 作为元语言词：描述 run, hit, feel 等一类词
```

**词性（POS）就是元语言词**：

| 词性 | 英文 | 缩写 | 例子 |
|---|---|---|---|
| 名词 | Noun | N | cat, book, John |
| 动词 | Verb | V | run, eat, think |
| 形容词 | Adjective | Adj | big, red, happy |
| 限定词 | Determiner | Det | the, a, this |
| 介词 | Preposition | P | on, in, with |

---

## 📌 Chapter 2：成分（Constituency）

### 定义

> **Constituent（成分）是一组作为单一语义单元行为的词。**

向成分添加或删除词，会破坏其语义完整性。

### 成分测试

**测试1：替换测试**
```
"The cat sat on the mat"
→ "The dog sat on the mat"  ✅（dog 替换 cat，整个 NP）
→ "dog sat on the mat"      ❌（不完整）
```

**测试2：移动测试**
```
原句："I saw [the man with the telescope]"
被动："[The man with the telescope] was seen by me"  ✅（整体移动）
```

### 常见成分类型

| 成分 | 缩写 | 例子 |
|---|---|---|
| 句子 | S | 完整的句子 |
| 名词短语 | NP | "the cat", "a big dog" |
| 动词短语 | VP | "sat on the mat" |
| 介词短语 | PP | "on the mat", "with a telescope" |

---

## 📌 Chapter 3：上下文无关文法（CFG）

### 四元组定义

$$G = (N, \Sigma, R, S)$$

| 符号 | 含义 | 例子 |
|---|---|---|
| **N** | 非终结符集合 | {S, NP, VP, N, V, Det} |
| **Σ** | 终结符集合 | {the, cat, sat, on, mat} |
| **R** | 产生式规则 | {S→NP VP, NP→Det N, ...} |
| **S** | 起始符号 | S（Sentence） |

### 关键概念

**终结符 vs 非终结符**：

| 特征 | 终结符 | 非终结符 |
|---|---|---|
| 可替换性 | ❌ 不能再替换 | ✅ 可以进一步展开 |
| 树中位置 | 叶子节点（具体词） | 内部节点（抽象类别） |

**上下文无关的含义**：规则可以在任何地方应用，不依赖周围上下文（S→NP VP 无论 S 在哪都能用）

### 推导示例："a flight"

```
NP ⇒ Det Nominal ⇒ a Nominal ⇒ a N ⇒ a flight

语法树：
      NP
     / \
   Det  Nominal
    |     |
    a     N
          |
        flight
```

### 文法等价

- **弱等价**：两个文法生成相同的句子集合（但树可能不同）
- **强等价**：生成相同句子集合 + 相同的树结构

---

## 📌 Chapter 4：Chomsky Normal Form（CNF）

### 三条规则

1. $A \rightarrow BC$（恰好两个非终结符）
2. $A \rightarrow a$（恰好一个终结符）
3. $S \rightarrow \epsilon$（只有起始符可推空串）

### 为什么需要 CNF？

> **CKY 算法依赖二元规则进行动态规划"二分"**
> 规则 $A \rightarrow BC$ 恰好符合"把大问题分成两个子问题"的思路

### CNF 转换三步

**Step 1：消除 ε 产生式**
```
原：NP → ε
新增：所有使用 NP 的规则加"NP 为空"的版本
     VP → V NP  →  VP → V（新增）
删除：NP → ε
```

**Step 2：消除单一产生式（A → B）**
```
原：VP → V（单一产生式，V 是非终结符）
把 V 的所有规则复制给 VP：VP → read（来自 V → read）
删除：VP → V
```

**Step 3：二元化长规则**
```
原：VP → V NP PP Adv（四个符号）
拆：VP → V X₁
   X₁ → NP X₂
   X₂ → PP Adv  ← 完成！
```

---

## 📌 Chapter 5：CKY 算法（理解思路为主）

### 核心思想

> **动态规划，自底向上填表**
> 把解析整个句子分解成解析左半部分 + 右半部分

### Fenceposts 概念

```
句子："book the flight"（3个词）

位置：  0    1    2    3
       |    |    |    |
        book  the  flight

跨度（Span）：
  [0,1]="book"  [1,2]="the"  [2,3]="flight"
  [0,2]="book the"  [1,3]="the flight"
  [0,3]="book the flight"
```

### 填表过程

```
Step 1：初始化对角线（单个词）
  table[0][1] = {V, N}    ← "book"
  table[1][2] = {Det}     ← "the"
  table[2][3] = {N}       ← "flight"

Step 2：自底向上填（长度从2到n）
  table[1][3] = {NP}      ← Det + N（"the flight"）

Step 3：查看 table[0][n]
  table[0][3] = {S, VP}   ← S ∈ table[0,3] → 合法！
```

**时间复杂度**：$O(n^3 \cdot |G|)$

---

## 📌 Chapter 6：结构歧义（Structural Ambiguity）

### 经典例子

```
"I shot an elephant in my pajamas"

解释1：我穿着睡衣射了一头大象（PP 附着 VP）
解释2：我射了一头穿着我睡衣的大象（PP 附着 NP）
→ 同一句话，两棵语法树！
```

### 消歧方法

| 方法 | 核心思想 | 准确率 |
|---|---|---|
| PCFG | 给规则加概率，选最高概率的树 | ~70-75% |
| 词汇化 CFG | 考虑具体词的附着倾向 | ~85-90% |
| **神经网络** | 端到端学习（BERT + Span） | **~95%+** |

---

## 📌 Chapter 7：评估方法（PARSEVAL）

$$\text{Precision} = \frac{\text{预测正确的成分数}}{\text{预测的成分总数}}$$

$$\text{Recall} = \frac{\text{预测正确的成分数}}{\text{金标准的成分总数}}$$

$$F_1 = \frac{2 \times P \times R}{P + R}$$

**Crossing Brackets**：两个跨度"交叉但不嵌套"，表示结构性错误，越少越好。

---

## 📌 Chapter 8：现代方法 —— Span-Based Parsing

**不依赖 CFG，直接用神经网络预测每个 span 的标签**：

```
1. BERT 生成每个词的上下文向量
2. 计算每个 fencepost 的向量
3. Span[i,j] = [fence_i; fence_j]（拼接）
4. MLP 分类器 → label
```

| 特征 | CKY | Span-Based |
|---|---|---|
| 依赖文法 | ✓ 需要 CFG+CNF | ✗ 不需要 |
| 速度 | 较慢 O(n³) | 较快 |
| 精度 | 较低 | 高（95%+ F1） |

---

## 🎯 产品经理要掌握到：

✅ **理解"句法分析"在现代 NLP 中的地位**：已被 Transformer 的隐式表示所替代，不是产品关注重点

✅ **CFG 的槽填充类比**：对话系统的意图识别和槽填充，思想上类似 CFG 推导（模板 + 填入）

✅ **结构歧义 = 用户说话的模糊性**：帮助理解为什么 NLP 系统有时解析错误

---

## 📐 算法工程师要掌握到：

✅ 能解释 CKY 的动态规划思路

✅ 理解 **依存句法（Dependency Parsing）** vs 成分句法：
- **成分句法**：树结构，关注短语层次（本模块内容）
- **依存句法**：有向图，关注词与词的直接关系（spaCy 默认使用）

✅ 知道 Span-Based Parsing 是现代标准做法

```python
# spaCy 依存句法分析（实际工作中更常用）
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The cat sat on the mat")
for token in doc:
    print(token.text, token.dep_, token.head.text)
```

---

## ⚡ 踩坑 & 经验

1. **CKY 要求文法必须是 CNF**：考试/面试时记住三步转换
2. **成分句法 ≠ 依存句法**：两者都叫"句法分析"，但结构不同，应用场景不同
3. **现代 NLP 很少直接用 CFG**：BERT 等预训练模型通过隐式表示捕捉了句法信息

---

## 💼 双视角面试题库

| 面试题 | 算法回答要点 | 产品回答要点 |
|---|---|---|
| "什么是 Context-Free Grammar？" | 四元组定义，规则不依赖上下文 | 形式化描述自然语言结构的工具 |
| "为什么 CKY 需要 CNF？" | 动态规划需要二元规则进行"二分" | 了解即可 |
| "什么是结构歧义？" | 同一句话有多个语法树，经典例子是 PP 附着 | 用户输入的模糊性，AI 解析错误的来源之一 |

---

## 🔖 核心公式速查

$$G = (N, \Sigma, R, S) \quad \text{（CFG 四元组）}$$

$$O(n^3 \cdot |G|) \quad \text{（CKY 时间复杂度）}$$

$$F_1 = \frac{2PR}{P+R} \quad \text{（PARSEVAL）}$$

---

**笔记版本**：v1.0 | **覆盖范围**：Module 9 (CFG & Constituency Parsing)
