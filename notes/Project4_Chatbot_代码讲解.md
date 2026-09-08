# 🤖 Project 4 Chatbot —— 代码逐函数讲解

> **对应作业**：CS 584 Project 4 - Basic Similarity Chatbot
> **源文件**：`chatbot_project4.py`（原代码文件当前未收录在本仓库）
> **总行数**：700 行
> **对应知识模块**：[[Module_11_QA_IR_RAG]]（TF-IDF + 余弦相似度）
> **最后更新**：2026-04-18

---

## 🎯 项目一句话总结

**用 TF-IDF + 余弦相似度做检索**：用户问一个问题 → 在语料库里找一个"最像"的问题 → 返回它对应的答案。没有神经网络、没有训练、没有 LLM。

---

## 🗺️ 文件整体结构地图

```
chatbot_project4.py (700 行)
│
├── ① 导入 & 常量区              (第 1-160 行)
│   ├── 顶部 docstring：项目说明 + 6 步架构 + 使用示例
│   ├── 依赖导入：sklearn（可选）、re、json、gzip、argparse...
│   └── 全局常量：
│       ├── BOT_NAME                     机器人名字
│       ├── DEFAULT_DATASET_CANDIDATES   数据集探测路径列表
│       ├── GENERIC_HELP_HINT            帮助提示文本
│       ├── SMALL_TALK_RESPONSES         闲聊回复字典
│       ├── SMALL_TALK_GREETING_TOKENS   打招呼关键词集合
│       ├── CAPABILITY_PHRASES           能力类问句短语
│       └── KEYWORD_STOPWORDS            停用词表
│
├── ② 工具函数区                 (第 163-378 行)
│   ├── normalize_text          轻量文本规范化（大小写+标点）
│   ├── _light_stem             词干提取（Porter 轻量版）
│   ├── preprocess_for_retrieval 完整预处理流水线（Step 3/4 核心）
│   ├── clean_answer_text       答案文本清洗
│   ├── extract_keywords        提关键词（用于 overlap 校验）
│   ├── detect_small_talk       识别闲聊
│   ├── parse_record            解析单行数据（JSON 或 Python dict）
│   ├── open_dataset            打开普通或 .gz 压缩文件
│   ├── load_corpus             加载语料库 ⭐ Step 1/2
│   ├── load_custom_qa          加载自定义 QA（覆盖用）
│   └── resolve_default_dataset 探测数据集路径
│
├── ③ 核心类：SimilarityChatbot   (第 381-586 行)
│   ├── __init__                ⭐ Step 3：向量化所有语料
│   ├── _tokenize               内部分词（调用预处理流水线）
│   ├── _build_fallback_index   手写 TF-IDF 倒排索引
│   ├── _rank_matches_fallback  手写余弦相似度检索
│   └── get_answer              ⭐ Step 4+5+6：回答用户问题
│
└── ④ 入口 & 命令行              (第 589-699 行)
    ├── parse_arguments         解析命令行参数
    └── main                    主流程控制
```

---

## 🧱 第一部分：全局常量（第 1-160 行）

### 🔧 依赖导入（第 52-61 行）

```python
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False
```

**设计点**：用 `try/except` 包装 sklearn 导入。**即使教授电脑没装 sklearn，程序也不会崩**——后面会自动走手写版 TF-IDF。这是一个**防御式编程**的典型例子。

---

### 📂 数据集路径探测（第 70-79 行）

```python
SCRIPT_DIR = Path(__file__).parent           # 脚本所在目录
DOWNLOADS_DIR = Path.home() / "Downloads"    # 用户家目录下的 Downloads

DEFAULT_DATASET_CANDIDATES = [
    DOWNLOADS_DIR / "qa_Electronics.json.gz",
    DOWNLOADS_DIR / "qa_Appliances.json.gz",
    SCRIPT_DIR / "qa_Electronics.json.gz",
    SCRIPT_DIR / "qa_Appliances.json.gz",
    SCRIPT_DIR / "sample_electronics_qa.jsonl",   ← 永远存在的兜底
]
```

**关键点**：`Path.home()` 会自动解析为**当前用户**的家目录——教授跑时就是教授的家目录，不依赖硬编码。

---

### 📝 各种常量集合（第 83-160 行）

| 常量名 | 作用 | 使用位置 |
|--------|------|---------|
| `GENERIC_HELP_HINT` | 通用引导语 | 关键词不重叠时兜底回复 |
| `SMALL_TALK_RESPONSES` | 闲聊回复字典 | `detect_small_talk` 返回值 |
| `SMALL_TALK_GREETING_TOKENS` | "hi/hello/hey"等 | 识别打招呼 |
| `CAPABILITY_PHRASES` | "what can you do"等 | 识别能力问询 |
| `KEYWORD_STOPWORDS` | 英语停用词集合 | 预处理时剔除 |

---

## 🛠️ 第二部分：工具函数逐个讲解

### 1️⃣ `normalize_text(text)` — 轻量文本规范化

**输入**：`"Does this Phone have Bluetooth??"`
**输出**：`"does this phone have bluetooth "`

**流水线**：
```
原文 → lower() → 去首尾空格 → 正则替换非字母数字为空格 → 合并多余空格
```

**代码**：
```python
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # 非字母数字 → 空格
    text = re.sub(r"\s+", " ", text)           # 多空格 → 单空格
    return text
```

**为什么要做这一步？** 检索要求"完全统一"——`"Phone"` 和 `"phone"` 如果不规范化，TF-IDF 会认为它们是两个不同的词。

**使用位置**：几乎所有函数都用它作为第一步清洗。

---

### 2️⃣ `_light_stem(token)` — 词干提取

**输入/输出**：
| 输入 | 输出 | 说明 |
|------|------|------|
| `"charging"` | `"charg"` | 去掉 `-ing` |
| `"charged"` | `"charg"` | 去掉 `-ed` |
| `"batteries"` | `"batteri"` | `ies → i` |
| `"works"` | `"work"` | 去掉 `-s` |
| `"cat"` | `"cat"` | 太短，不处理 |

**代码逻辑**：
```python
def _light_stem(token):
    if len(token) <= 3:
        return token                          # 太短不动
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "i"              # batteries → batteri
    for suffix in ("ingly", "edly", "ing", "ed", "es", "ly", "est", "er", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]      # 剥掉常见后缀
    return token
```

**为什么重要？** 没有词干提取的话：
- 语料里有 `"is this charger compatible"`
- 用户问 `"are these chargers compatible"`
- `charger` vs `chargers` 被视为不同词 → 相似度下降

**有了词干提取**：两个都变成 `charger`（或 `charg`），匹配度立刻上升。

---

### 3️⃣ `preprocess_for_retrieval(text)` — **完整预处理流水线** ⭐

这是 **Step 3 和 Step 4 的核心**，语料和用户输入都走这条流水线。

**流水线**：
```
输入: "Does this phone have Bluetooth??"
  ↓ normalize_text
  "does this phone have bluetooth "
  ↓ split()（按空格分词）
  ["does", "this", "phone", "have", "bluetooth"]
  ↓ 过滤停用词（does/this/have 都在 KEYWORD_STOPWORDS）
  ["phone", "bluetooth"]
  ↓ _light_stem 每个词
  ["phone", "bluetooth"]   ← 最终输出
```

**代码**：
```python
def preprocess_for_retrieval(text: str) -> List[str]:
    normalized = normalize_text(text)
    return [
        _light_stem(token)
        for token in normalized.split()
        if token and token not in KEYWORD_STOPWORDS
    ]
```

**关键设计**：**同一个函数同时用于语料和查询**，保证两边向量空间完全一致——这是 TF-IDF 检索正确工作的前提。

---

### 4️⃣ `clean_answer_text(text)` — 答案文本清洗

**只做两件事**：去首尾空格 + 合并多空格。

**为什么答案不做 lower/去标点？** 答案是给人看的，要**保留原格式**：`"Yes."` 和 `"No, this phone..."` 的大小写、标点都要留。

---

### 5️⃣ `extract_keywords(text)` — 提"有意义的"关键词

```python
def extract_keywords(text: str) -> set[str]:
    tokens = normalize_text(text).split()
    return {token for token in tokens if len(token) >= 3 and token not in KEYWORD_STOPWORDS}
```

**输入**：`"does this phone have bluetooth"`
**输出**：`{"phone", "bluetooth"}`

**用途**：在 `get_answer` 里做**关键词重叠校验**——如果用户问题和匹配到的语料问题**连一个关键词都不重叠**，说明检索可能"乱答"，就返回兜底提示。

---

### 6️⃣ `detect_small_talk(cleaned_query)` — 识别闲聊

**三种闲聊分类**：

| 用户说 | 返回 |
|--------|------|
| `"hi"`, `"hello"`, `"hey"` | greeting 回复 |
| `"thanks"`, `"thank you"` | thanks 回复 |
| `"what can you do"`, `"help me"` | capability 回复 |
| 其他 | `None`（继续走 TF-IDF 检索） |

**为什么要这一层？** 如果没有这层，用户说 "hi" 会被强制丢进 TF-IDF 检索，可能匹配到一个牛头不对马嘴的答案（比如"电池使用说明"）。加一层**前置分类器**是产品级聊天机器人的常规做法。

---

### 7️⃣ `parse_record(line)` — 解析一行数据

**为什么需要它？** Amazon QA 数据集的格式**不是标准 JSON**，而是 Python dict 格式：
```
{'question': 'does it work', 'answer': 'yes', ...}   ← 单引号！
```

标准 `json.loads` 会报错（JSON 必须双引号）。所以代码做了 **两级回退**：
```python
try:
    record = json.loads(line)            # 先试标准 JSON
except json.JSONDecodeError:
    pass

try:
    record = ast.literal_eval(line)      # 再试 Python 字面量解析
except (ValueError, SyntaxError):
    return None
```

**`ast.literal_eval` 的好处**：比 `eval` 安全——只能解析字面量（字符串、数字、dict、list），不会执行任意代码。

---

### 8️⃣ `open_dataset(path)` — 透明打开普通/压缩文件

```python
def open_dataset(path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")
```

**关键**：返回的对象都支持 `for line in file` 迭代，调用方不用关心是否压缩。

`errors="ignore"` 是为了跳过文件里的个别坏字符（大数据集常见）。

---

### 9️⃣ `load_corpus(path, max_rows)` — **⭐ Step 1 + Step 2**

**核心作用**：把原始数据集变成两个**索引对齐**的列表。

**流程图**：
```
打开文件
  ↓ 逐行读
parse_record(line) → dict
  ↓ 提取 question、answer
  ↓ normalize_text(question)
  ↓ Y/N 答案特殊处理（统一成 "Yes."/"No."）
questions.append(normalized_question)   ← Step 1
answers.append(answer)                  ← Step 2
```

**为什么要求 `questions[i]` 对应 `answers[i]`？** 这是**检索方案**的核心约定：找到问题的 index，同样 index 在 answers 里就是对应答案。

**`max_rows` 参数**：大数据集（电子产品类 30万+ 条）全部加载会吃光内存，默认只加载 50000 条。

---

### 🔟 `load_custom_qa(path)` — 加载自定义覆盖

**返回 3 样东西**：
```python
return questions, answers, exact_answers
```
- `questions/answers`：也加到检索池里（参与 TF-IDF）
- `exact_answers: Dict[str, str]`：**精确匹配**字典，query 规范化后如果完全命中就跳过检索

**为什么要两套机制？**
- 精确匹配：用户问的就是你写的问题 → 直接返回
- 参与检索：用户问的是相似问题 → TF-IDF 也能找到

---

## 🧠 第三部分：`SimilarityChatbot` 类 ⭐⭐⭐

这是项目的**核心引擎**。

### 🏗️ 构造函数 `__init__` — Step 3 全在这里

```python
def __init__(self, questions, answers, min_confidence=0.12, min_margin=0.02,
             use_sklearn=True, exact_answers=None):
    # ... 参数校验和保存 ...

    if self.use_sklearn:
        # === Step 3：用 sklearn TF-IDF 向量化所有语料问题 ===
        self.vectorizer = TfidfVectorizer(
            tokenizer=preprocess_for_retrieval,   # ← 关键：用我们自己的预处理
            token_pattern=None,
            preprocessor=lambda x: x,             # 不用 sklearn 默认预处理
            lowercase=False,
            ngram_range=(1, 2),                   # bigram：捕捉 "battery life" 这种短语
        )
        self.question_matrix = self.vectorizer.fit_transform(self.questions)
        # ↑ 结果是稀疏矩阵，shape = (问题数, 词表大小)
    else:
        # 手写 TF-IDF（sklearn 不可用时的兜底）
        self._build_fallback_index()
```

**两个关键点**：

1. **`tokenizer=preprocess_for_retrieval`**：把我们的预处理插入 sklearn 管线，保证语料和查询走同一条路径。
2. **`ngram_range=(1, 2)`**：TF-IDF 不只看单个词，还看**相邻两个词**的组合（bigram）。"battery life"、"warranty period" 这种两词短语会被当做特征，检索更准。

---

### 📚 `_build_fallback_index` — 手写 TF-IDF（没 sklearn 时）

这段代码实际上是在**复现 sklearn 内部做的事**，分三个阶段：

#### 阶段 1：统计文档频率（DF）
```python
for question in self.questions:
    tokens = self._tokenize(question)
    for token in set(tokens):          # set 去重，只统计"文档是否出现此词"
        doc_freq[token] += 1
```

#### 阶段 2：计算 IDF
```python
for token, df in doc_freq.items():
    self.idf_by_idx[token_index] = math.log((1.0 + doc_count) / (1.0 + df)) + 1.0
```
**IDF 公式**：$\text{idf}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$

- $N$ = 文档总数
- $\text{df}(t)$ = 词 $t$ 出现过的文档数
- **常见词 IDF 低**（如 `"phone"` 在每条都有），**罕见词 IDF 高**（如 `"esim"`）

#### 阶段 3：为每个文档计算 TF-IDF 向量 → 构建**倒排索引**
```python
for doc_index, tokens in enumerate(doc_tokens):
    # 1. 算 TF × IDF
    # 2. L2 归一化
    # 3. 存入倒排索引 postings[token_index] = [(doc_index, weight), ...]
    ...
```

**倒排索引是什么？** 反向存储：
```
正排：doc_0 → [phone, bluetooth, battery]
倒排：phone → [(doc_0, 0.3), (doc_5, 0.5), ...]
      bluetooth → [(doc_0, 0.7), (doc_12, 0.9), ...]
```

**好处**：算余弦相似度时，**只用遍历查询里的几个词**对应的 posting list，不用扫全部几万个文档——这是 IR 领域的经典加速技巧（对应 [[Module_11_QA_IR_RAG]] 里讲的 BM25 也用同样结构）。

---

### 🎯 `_rank_matches_fallback(query, top_k)` — 手写余弦检索

**流程**：
```
1. 对 query 分词 + TF-IDF 加权 + L2 归一化
2. 遍历 query 里每个词 token:
     取 postings[token] = [(doc_0, w_0), (doc_5, w_5), ...]
     对每个 (doc_i, w_i)，累加 query_weight × w_i 到 scores[doc_i]
3. 按 score 排序，取 top_k
```

**为什么这等价于余弦相似度？** 因为 query 和 doc 向量**都已经 L2 归一化**，所以余弦相似度 = 点积。点积 = 每个维度相乘相加 = 倒排索引遍历的累加结果。

数学等价：
$$\cos(q, d) = \frac{q \cdot d}{|q||d|} = \hat q \cdot \hat d = \sum_t \hat q_t \hat d_t$$

---

### 💬 `get_answer(user_query)` — **Step 4 + 5 + 6 全流程** ⭐⭐⭐

这是**每次用户提问都会调用**的函数，分五关：

```
用户输入 → [关1] 规范化 → [关2] 闲聊识别 → [关3] 精确匹配
       → [关4] TF-IDF 向量化 + 余弦相似度 → [关5] 三道保险 → 返回答案
```

#### 关 1：规范化
```python
cleaned_query = normalize_text(user_query)
if not cleaned_query:
    return "Please ask a valid question.", 0.0
```

#### 关 2：闲聊识别
```python
small_talk_response = detect_small_talk(cleaned_query)
if small_talk_response is not None:
    return small_talk_response, 1.0         # 1.0 = 最高置信度
```

#### 关 3：精确匹配
```python
exact_answer = self.exact_answers.get(cleaned_query)
if exact_answer is not None:
    return exact_answer, 1.0
```

#### 关 4：**TF-IDF 检索 —— Step 4 + Step 5 的核心**
```python
if self.use_sklearn:
    # Step 4：向量化用户查询
    query_vector = self.vectorizer.transform([cleaned_query])

    # Step 5：余弦相似度
    scores = cosine_similarity(query_vector, self.question_matrix)[0]
    # scores 形状：(N,)  一个数组，每个元素是 query vs 一个语料问题的相似度

    top_indices = scores.argsort()[::-1][:2]   # 按分数降序取前 2
    best_index = int(top_indices[0])
    best_score = float(scores[best_index])
    second_score = float(scores[top_indices[1]])
else:
    # fallback 路径
    top_matches = self._rank_matches_fallback(cleaned_query, top_k=2)
    best_index, best_score = top_matches[0]
    second_score = top_matches[1][1]
```

#### 关 5：三道保险
```python
# 保险 1：最高分太低 → 不确定
if best_score < self.min_confidence:       # 默认 0.12
    return "I could not find a close match...", best_score

# 保险 2：第 1 和第 2 太接近 → 模糊
if (best_score - second_score) < self.min_margin:   # 默认 0.02
    return "I found multiple similar questions...", best_score

# 保险 3：关键词零重叠 → 可能乱答
query_keywords = extract_keywords(cleaned_query)
matched_keywords = extract_keywords(self.questions[best_index])
if query_keywords and not query_keywords.intersection(matched_keywords):
    return "Your question seems too general...", best_score

# ✅ Step 6：返回最匹配问题对应的答案
return self.answers[best_index], best_score
```

**三道保险的意义**：防止 TF-IDF 在"没有真正匹配"时**硬塞一个答案**。工业级聊天机器人的标准做法。

---

## 🏁 第四部分：主流程 `main()`

```
main() 流程：
┌─────────────────────────────────────────────┐
│ 1. parse_arguments()：解析命令行             │
│ 2. resolve_default_dataset()：探测数据集     │
│ 3. load_corpus() → questions, answers ★S1+S2│
│ 4. load_custom_qa() → 合并入主列表           │
│ 5. SimilarityChatbot(...)  ★Step 3           │
│ 6. input("username") + 问候                  │
│ 7. while True:                               │
│      user_input = input()                    │
│      if "bye" → break                        │
│      chatbot.get_answer(user_input) ★S4+5+6  │
└─────────────────────────────────────────────┘
```

---

## 🌐 整体数据流程图（最重要的一张图）

```
                     启动阶段（只跑一次）
    ┌─────────────────────────────────────────────────────┐
    │ 数据集文件                                          │
    │    ↓ open_dataset + parse_record                    │
    │ 原始 QA 记录（dict）                                │
    │    ↓ load_corpus                                    │
    │ questions[] + answers[]   ★ Step 1 & 2              │
    │    ↓ SimilarityChatbot.__init__                     │
    │      → preprocess_for_retrieval（每条问题）         │
    │      → TF-IDF 向量化                                │
    │ question_matrix (稀疏矩阵)   ★ Step 3               │
    └─────────────────────────────────────────────────────┘

                     运行阶段（每次用户提问）
    ┌─────────────────────────────────────────────────────┐
    │ 用户输入 "does this phone have bluetooth"           │
    │    ↓ normalize_text                                 │
    │ "does this phone have bluetooth"                    │
    │    ↓ detect_small_talk → None（不是闲聊）           │
    │    ↓ exact_answers lookup → None（无精确匹配）      │
    │    ↓ vectorizer.transform  ★ Step 4                 │
    │ query_vector                                        │
    │    ↓ cosine_similarity(query_vector, matrix)  ★ S5  │
    │ scores = [0.05, 0.82, 0.11, ...]                    │
    │    ↓ argsort 取 top 2                               │
    │ best_index=1, best_score=0.82                       │
    │    ↓ 三道保险通过                                   │
    │    ↓ answers[best_index]  ★ Step 6                  │
    │ "No, this specific model does not include..."       │
    └─────────────────────────────────────────────────────┘
```

---

## 🧪 实际调用链示例

用户输入：`"does it have bluetooth"`

```
main() → while 循环
  └→ chatbot.get_answer("does it have bluetooth")
       ├→ normalize_text("does it have bluetooth")
       │     → "does it have bluetooth"
       ├→ detect_small_talk("does it have bluetooth")
       │     → None（不是闲聊）
       ├→ exact_answers.get("does it have bluetooth")
       │     → None
       ├→ self.vectorizer.transform(["does it have bluetooth"])
       │     ├→ 内部调用 preprocess_for_retrieval("does it have bluetooth")
       │     │     → ["bluetooth"]  （does/it/have 都是停用词被剥掉）
       │     └→ 稀疏向量 query_vector
       ├→ cosine_similarity(query_vector, self.question_matrix)
       │     → scores = [0.0, 0.03, 0.87, 0.01, ...]
       ├→ scores.argsort()[::-1][:2]  → [2, 1]
       ├→ best_score=0.87, second_score=0.03
       ├→ 保险 1 通过（0.87 > 0.12）
       ├→ 保险 2 通过（0.87 - 0.03 = 0.84 > 0.02）
       ├→ 保险 3 通过（关键词 {"bluetooth"} 有交集）
       └→ return self.answers[2]
            = "No, this specific model does not include Bluetooth support."
```

---

## 📊 核心知识对应 Module 11

| 代码里的概念 | Module 11 知识点 |
|------------|-----------------|
| TF-IDF 向量化 | TF-IDF 权重公式 |
| `question_matrix` 稀疏矩阵 | 倒排索引（Inverted Index）的变体 |
| 余弦相似度 | 向量空间模型的相似度度量 |
| `postings` 结构 | 倒排列表（Posting List） |
| bigram `ngram_range=(1,2)` | n-gram 特征扩展 |
| 三道保险阈值 | 检索置信度门控 |
| 整个架构 | Retrieval-only QA（RAG 的检索部分） |

---

## 💼 面试可能被问的问题

### Q1：为什么用 TF-IDF 而不是 BM25？
**A**：TF-IDF 是基线方法，更简单直观；BM25 增加了词频饱和和文档长度归一化，在长文档上更好。教学项目用 TF-IDF 足够，生产会升级到 BM25 或 Dense Retrieval（sentence-BERT）。

### Q2：为什么要做词干提取（stemming）？
**A**：没有 stemming 时，`"charger"` 和 `"chargers"` 被视为不同词，导致语料里有 `"charger"` 而用户问 `"chargers"` 时相似度下降。stemming 把它们映射成同一个 stem `"charger"`，匹配度立刻提升。

### Q3：如果语料有 100 万条，这个方案还行吗？
**A**：sklearn 的 `TfidfVectorizer + cosine_similarity` 是稠密计算，100 万条会很慢。需要改成**倒排索引 + 只算有重叠词的文档**（代码里的 fallback 路径其实已经是这个结构）。再大就要上 FAISS / Elasticsearch / Vector DB。

### Q4：三道保险里，哪一道最重要？
**A**：`min_margin`（第一名第二名分差）最重要。它防止"有很多相似但意思不同的问题"时乱答。比如用户问"warranty" 时，语料里可能有 10 条 warranty 相关问题，第一名和第二名差 0.01，这时回答其实是随机挑的。

### Q5：这个项目如何升级成完整 RAG？
**A**：
1. 检索部分不变（或升级到 Dense Retrieval）
2. 把 top-k 相似问题的答案**拼接成 context**
3. 把 context + 用户问题交给 LLM（如 GPT）
4. LLM 生成最终回答

即：**当前代码是 R（Retrieval），只需加 AG（Augmented Generation）就是完整 RAG**。

---

## 🎓 学到的设计模式

1. **try/except 可选依赖** → 程序在有/没有某库时都能运行
2. **Path.home()** → 避免硬编码路径
3. **一套预处理函数共享给语料和查询** → 保证向量空间一致
4. **多级回退解析**（JSON → ast.literal_eval） → 处理非标准格式
5. **倒排索引** → 加速稀疏向量检索
6. **前置分类器**（detect_small_talk） → 避免对无意义输入强行检索
7. **置信度 + margin + 关键词校验**三道保险 → 防止乱答

---

## 🔗 相关知识

- [[Module_11_QA_IR_RAG]]：TF-IDF、BM25、余弦相似度的完整理论
- [[Module_07_Deep_Learning]]：如果要升级到 Dense Retrieval，用 BERT/sentence-BERT
- [[Module_12_Dialogue_Systems]]：对话系统架构，这个项目是"FAQ bot"的经典范式

---

**笔记版本**：v1.0 | **适用作业**：CS584 Project 4
