# **NebulaMem: 星域激活式 AI 智能体长效记忆系统架构蓝图 (Nebula Activation-Based Memory System)**

# **1. 核心愿景 (Core Vision)**

传统的 AI 记忆方案，如基于原始文本匹配的 RAG，或卡帕西提出的 [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) 静态编译方案，本质上是"检索式（Retrieval-based）"而非"认知式（Cognitive-based）"的。

在人脑中，记忆并非是以离散、静态的文件形式存储的，而是一个流动的神经通路网——像一片广袤的星域（Nebula）。当受到某个具体念头（Query）的刺激时，附近的"星点（Nodes）"率先被点亮，随后电信号沿着"星轨（Edges/Synapses）"扩散开来，将一系列互相关联的概念在脑海中重构出来。

**NebulaMem** 的核心愿景，即是将这种"激活扩散（Spreading Activation）"的生物脑运行机理，与"卡帕西 LLM Wiki"的高内聚编译哲学相结合，为个人编程和日常 AI 助手打造一套兼具联想活性、自适应演化与高能效比的下一代长效记忆框架。

# **2. 三大核心架构层 (The Three-Tier Architecture)**

## **2.1 星点层（The Nebula Node Layer — 偶发与语义事实）**

星点层是记忆的原子实体。

  - **存储形式**：每个节点（Node）代表一个精炼的原子事实（Atomic Fact），例如："用户习惯采用 Rust 语言进行微服务架构"。
  - **无模型索引（Model-Free Indexing）**：本实现**不使用任何预训练 embedding 模型**。节点通过 ① BM25 倒排索引（词法）与 ② 自研 Random Indexing（从用户自身语料增量生长的零预训练语义向量）双层定位；并自动从内容中挖掘专有名词/数字作为「实体锚点」，用于自组织地连接相关记忆。
  - **元数据跟踪**：记录其创建时间、修改时间以及"最后一次激活强度"，用以提供时间衰减权重。

## **2.2 星轨层（The Constellation Layer — 关系图谱与激活扩散）**

星轨层是点亮记忆星域的核心机制，通过概念之间的关联边（Edges）建立突触连接。

  - **动态边权值（Synaptic Plasticity）**：边上的权重（Weight, 0.0 ~ 1.0）代表两个概念的关联强度。每当两个概念被同时提及或被 LLM 推理关联时，边权重自动增强（最高 1.0）；若长期不被共同触发，则随时间自动衰减（Decay），模拟生物脑的遗忘机制。
  - **激活扩散算法（Spreading Activation Algorithm）**：  
    当用户输入一个问题时，系统按以下流程运行：
      
    1.  **种子定位**：通过 BM25 倒排检索（可叠加自研 Random Indexing 语义重排）定位最相关的 1-3 个"种子星点"，并以其归一化分值作为初始能量 $A_{0}$。此步**零模型**，且检索按需进行——只触碰 query 命中词的倒排链。
    2.  **能量扩散**：进行 $N$ 步迭代，在每步迭代中，节点上的能量顺着"星轨"（关联边）向邻近节点扩散。扩散公式为：  
        $$A_{t+1}(n) = A_t(n) + \sum_{m \in Neighbors(n)} A_t(m) \cdot W(m, n) \cdot \lambda$$  
        其中 $W(m, n)$ 是两个节点间的边权，$\lambda$ 是扩散衰减因子。
    3.  **星火点亮**：当节点能量超过特定阀值 $\tau$ 时，该节点即被判定为"已点亮（Fired）"。这组被点亮的节点共同构成了一副独特的"星座图景（Active Constellation）"，完美再现了人类在思考时产生的联想波澜。

## **2.3 星系融合层（The Nebula Fusion Layer — 编译与梦境）**

被点亮的星座图景需要被压缩、重构并呈递给大模型，这便是融合编译层的工作。

  - **异步整理（Consolidation / Dreaming）**：系统在空闲时整理图谱。本实现**不调用 LLM**，采用**抽取式**策略：在共享实体簇内用 token Jaccard 判定近重复节点并合并其连接，配合边权随时间衰减，从而"消除冗余、保持图谱清爽"。生成式改写（把多条事实重写成新句）因需模型而**有意不纳入**，以避免臆造文本。
  - **LLM Wiki 格式化生成**：最终将激活的节点拓扑排序，编译成一份高度定制、结构化的 Markdown 记忆片段，作为 System Prompt 注入 LLM 上下文。这保证了模型获得的不是冰冷的文本碎块，而是有逻辑脉络、有关联深度的"意识流"。

# **3. 技术实现框架与 Python 源码示例 (Implementation & Python Prototype)**

我们实现了一个完整的、**无模型、无 LLM**、仅依赖 `numpy` 的进程内 Python 实现：

  - **BM25 倒排索引**：按需种子定位（`nebulamem/lexical.py`）。
  - **自研 Random Indexing**：零预训练的本地语义层（`nebulamem/semantic.py`）。
  - **纯 Python 关联图**：激活扩散、侧向抑制、突触衰减、自组织桥接（`nebulamem/graph.py` + `core.py`）。
  - **SQLite / 内存内容存储**：节点内容按需懒加载（`nebulamem/store.py`）。

源码与实测均已通过：完整结构见 `nebulamem/` 包、`demo.py` 与 `tests/`；HotpotQA 多跳实测见 `docs/BENCHMARK.md`（1000 题 / CPU / 3.3 秒，两跳召回 0.42→0.71）。

# **4. 边缘情况与自适应进化机制 (Handling Edge Cases & Evolution)**

  - **事实冲突消解**：通过在星轨中引入"时序替代边（temporal_override）"，当新事实写入时（如"我改用 Rust 了"），旧事实（"我目前在写 Python"）与新事实之间会建立强抑制连接，引导扩散算法在检索时抑制旧节点，从而自动完成记忆更新。
  - **长短期记忆固化（CLS 理论）**：临时会话中的碎块事实先存储在"临时星系"中，仅当其被多次激活或经过"梦境整理周期"后，才会被提纯并升级为"永久语义星系"。

本文档引用并结合了当前先进开源记忆系统 [Mem0](https://github.com/mem0ai/mem0) 与 2026 年最新的时序知识图谱引擎 [Zep/Graphiti](https://github.com/getzep/zep) 的优秀思想，致力于提供更具人类联想特征的长效记忆体验。