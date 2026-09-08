# CS584 课程材料索引

## 一、整理口径

资料按四类区分：

| 类别 | 含义 | 仓库位置 |
|---|---|---|
| 课程文档 | Syllabus 和教授课件 | `materials/00-course-overview/` 及各 Module 目录 |
| 视频资料 | 从课堂视频导出的文字稿 PDF | 对应 Module 目录 |
| 参考书 | 课程指定教材 | `materials/reference-books/` |
| 补充资源 | 笔记中引用的教材章节、论文、视频和工具 | 本索引下方的外部链接 |

`(xx mins).pdf` 类文件是视频内容的文字导出稿，不是原始视频。本机的 CS584 资料目录里没有找到原始视频文件。

## 二、按 Module 查找

| Module | 主题 | 已有笔记 | 本地原始资料 | Syllabus 指定阅读 |
|---|---|---:|---|---|
| 01 | Introduction to NLP | 否 | 讲义 + 视频文字稿 | 无 |
| 02 | Machine Learning Basics | 是 | 讲义 + 向量/矩阵作业 | 无 |
| 03 | NLP Language Representation | 否 | 讲义 | SLP Ch. 2 |
| 04 | Language Models and Sentiment Classification | 否 | N-gram 讲义 + 视频文字稿 | SLP Ch. 3–4 |
| 05 | Regression and Vector Semantics | 否 | 本机未找到 | SLP Ch. 5–6 |
| 06 | Neural Networks and Sequence Labeling | 是 | 本机未找到 | SLP Ch. 7–8 |
| 07 | Deep Learning Architectures | 是 | 本机未找到 | SLP Ch. 9 |
| 08 | Machine Translation | 是 | 本机未找到 | SLP Ch. 10 |
| 09 | Transfer Learning and Constituency Grammars | 是 | 本机未找到 | SLP Ch. 11–12 |
| 10 | Constituency and Dependency Parsing | 是 | 本机未找到 | SLP Ch. 13–14 |
| 11 | Question Answering and Visualization Paths | 是 | 本机未找到 | SLP Ch. 23 |
| 12 | Chatbots and Dialogue Systems | 是 | 本机未找到 | SLP Ch. 24 |
| 13 | Automatic Speech Recognition | 是 | 本机未找到 | SLP Ch. 26 |

## 三、本地文件清单

### 课程总览

- [`CS 584 Syllabus.pdf`](materials/00-course-overview/CS%20584%20Syllabus.pdf)

### Module 01

- [`Module 1 What is NLP.pdf`](materials/module-01-nlp-intro/Module%201%20What%20is%20NLP.pdf)
- [`Module 1 Course Introduction - Video Transcript.pdf`](materials/module-01-nlp-intro/Module%201%20Course%20Introduction%20-%20Video%20Transcript.pdf)

### Module 02

- [`Module 2 Machine Learning Basics.pdf`](materials/module-02-ml-basics/Module%202%20Machine%20Learning%20Basics.pdf)
- [`Vectors_and_Matrices_Operations_Homework(1).ipynb`](materials/assignments/Vectors_and_Matrices_Operations_Homework(1).ipynb)

### Module 03

- [`Module 3 Natural Language Representation.pdf`](materials/module-03-language-representation/Module%203%20Natural%20Language%20Representation.pdf)

### Module 04

- [`Module 4 N Gram Language Models.pdf`](materials/module-04-language-models/Module%204%20N%20Gram%20Language%20Models.pdf)
- [`Module 4 N-Gram Language Models - Video Transcript.pdf`](materials/module-04-language-models/Module%204%20N-Gram%20Language%20Models%20-%20Video%20Transcript.pdf)

### 参考书

- [`Speech and Language Processing, 3rd edition draft`](materials/reference-books/Speech%20and%20Language%20Processing(1)%20(1).pdf), Jurafsky & Martin, 2022 draft

## 四、笔记中的补充资源

### Module 06: Neural Networks and Sequence Labeling

- [SLP Chapter 7: Neural Networks](https://web.stanford.edu/~jurafsky/slp3/7.pdf)
- [SLP Chapter 8: Sequence Labeling](https://web.stanford.edu/~jurafsky/slp3/8.pdf)
- [Stanford CS224N: RNN and LSTM](https://www.youtube.com/watch?v=0LixFSa7yts)
- [3Blue1Brown: Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)
- [spaCy named-entity recognition documentation](https://spacy.io/usage/linguistic-features#named-entities)
- [Hugging Face token-classification documentation](https://huggingface.co/docs/transformers/task_summary#token-classification)
- [CoNLL-2003 named-entity recognition dataset](https://www.clips.uantwerpen.be/conll2003/ner/)
- [PyTorch sequence-model tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

### Module 10: Dependency Parsing

- [SLP Chapter 14: Dependency Parsing](https://web.stanford.edu/~jurafsky/slp3/14.pdf)
- [Neural Network Methods for Natural Language Processing](https://www.amazon.com/dp/1627052984)
- [Stanford CS224N: Dependency Parsing](https://www.youtube.com/watch?v=nC9_RfjYwqA)
- [CMU CS11-711: Dependency Parsing](https://www.youtube.com/watch?v=Qf-3LGaMiEk)
- [Universal Dependencies](https://universaldependencies.org/)
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
- [spaCy dependency-parsing documentation](https://spacy.io/usage/linguistic-features#dependency-parse)
- [A Fast and Accurate Dependency Parser](https://aclanthology.org/D14-1082.pdf)
- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
- [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42)

### Module 11: QA, IR and RAG

- [SLP Chapter 23: Question Answering and Information Retrieval](https://web.stanford.edu/~jurafsky/slp3/23.pdf)
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [BERT](https://arxiv.org/abs/1810.04805)
- [ColBERT](https://arxiv.org/abs/2004.12832)
- [DPR](https://arxiv.org/abs/2004.04906)
- [Stanford CS224N: Question Answering](https://www.youtube.com/watch?v=NcqfHa0_YmU)
- [Pinecone RAG tutorial](https://www.youtube.com/watch?v=T-D1OfcDW1M)
- [Hugging Face question-answering documentation](https://huggingface.co/docs/transformers/task_summary#question-answering)
- [LangChain question-answering documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- [Natural Questions](https://ai.google.com/research/NaturalQuestions)
- [Elasticsearch BM25 similarity](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html)

## 五、待补资料

- Module 01、03、04、05 的独立 Markdown 笔记。
- Module 05–13 的课程原始讲义、视频或视频文字稿。
- Project 1–4 的原始题目与代码；当前只收录了 Project 4 的代码讲解笔记，未找到其中引用的 `chatbot_project4.py`。
