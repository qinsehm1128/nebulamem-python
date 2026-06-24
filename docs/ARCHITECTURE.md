# NebulaMem 架构说明（无模型版 / Model-Free Architecture）

> 设计目标：在**不依赖任何 embedding 模型、不依赖任何 LLM** 的前提下，用「现有经典算法 + 自研算法」实现蓝图中的星域激活式记忆。全部逻辑为纯 Python + numpy，进程内运行，零外部服务、零 GPU。

## 1. 总体拓扑

```text
[ Host App (IDE / CLI / Agent) ]
            │  register_memory() / retrieve()
            ▼
┌─────────────────────────────────────────────────────────────┐
│                      NebulaMem Core                          │
│                                                              │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │  Lexical   │  │  Semantic   │  │       Graph          │  │
│  │  BM25 倒排 │  │ RandomIndex │  │  关联图 + 激活扩散    │  │
│  │ (种子定位) │  │ (自研语义)  │  │  + 侧向抑制 + 衰减     │  │
│  └────────────┘  └─────────────┘  └──────────────────────┘  │
│         │ 种子            │ 重排            │ 能量扩散          │
│         └────────┬────────┴────────┬────────┘                │
│                  ▼                 ▼                          │
│            NodeStore (sqlite/内存, 按需加载 content)         │
└─────────────────────────────────────────────────────────────┘
```

对应模块：`lexical.py` · `semantic.py` · `graph.py` · `store.py` · `core.py` · `consolidation.py` · `text.py`。

## 2. 三层映射（蓝图 → 实现）

| 蓝图层 | 蓝图原方案（需模型） | 本实现（无模型） | 代码 |
| --- | --- | --- | --- |
| 星点层 | 每节点一条预训练 embedding | 原子句节点 + 自动挖掘的实体锚点（专有名词/数字） | `text.extract_entities` |
| 星轨层·种子定位 | Query↔节点 余弦相似度（向量模型） | **BM25** 倒排检索 + 可选 **Random Indexing** 余弦重排 | `lexical.py` `semantic.py` |
| 星轨层·激活扩散 | 图上能量迭代 | 同上，纯图算法（无变化） | `core.retrieve` |
| 星轨层·侧向抑制 | 抑制边负能量 | 同上，纯图算法（无变化） | `graph` + `core` |
| 星系融合·梦境 | 轻量 LLM 抽象重写 | **抽取式**去重合并 + 边权衰减（不做生成式改写） | `consolidation.py` |
| Wiki 编译 | Markdown 拼装 | 同上 + **Token 预算**截断 | `core.compile_to_markdown` |

## 3. 种子定位：BM25 + 自研 Random Indexing

替代 embedding 模型的两层：

1. **BM25（现有算法）** — Okapi BM25 倒排索引。检索是**按需**的：只触碰 query 命中词的倒排链，开销随 query 而非语料规模增长。
2. **Random Indexing（自研，零预训练）** — 每个词有一个由 `sha256` 确定性生成的稀疏三元随机「索引向量」；词的「上下文向量」= 其滑窗共现词索引向量之和（二阶共现统计）。文档/查询向量 = 其词上下文向量之和。出现在相似上下文的词（同义/术语）向量靠近 → 余弦可捕捉 BM25 漏掉的改写。**它随用户语料自我成长，无需任何预训练权重。** 默认关闭（`semantic_weight=0`），可按需开启。

种子分数归一化后作为激活初始能量 $A_0$。

## 4. 激活扩散与侧向抑制（数学模型）

种子集合 $S$（归一化能量），迭代 $N$ 步：

- 关联 / 时序边 $(m\to n)$：$\Delta A(n) = A(m)\cdot W(m,n)\cdot \lambda$
- 抑制边 $(m\to n)$：$\Delta A(n) = -A(m)\cdot W(m,n)\cdot \beta$
- 汇总并钳位：$A_{t+1}(n)=\max(-1,\min(1, A_t(n)+\Delta A(n)))$

点火：$A(n)\ge \tau$ 的节点入选，按能量排序、`max_results` 截断后**才**按需加载 content。

**自组织关联（自研）**：`graph.auto_associate` 在写入时把新节点连到「共享实体锚点」的已有节点，权重随共享锚点数增长；并对高频锚点做文档频率剪枝（`max_anchor_df`）防止枢纽爆炸。这正是「无模型」造出**多跳桥接边**的底层机制——共同提到同一专有名词的两条记忆自动成为邻居。

**冲突消解**：`update_state_with_suppression(old,new)` 建立 `new→old` 抑制边；当 new 被点亮，负能量沿抑制边把过时的 old 压到阈值以下熄灭。

## 5. 遗忘与梦境整理（无 LLM）

- **权重衰减**：`graph.decay_weights` 让非抑制边随空闲时间衰减至下限 `MIN_WEIGHT=0.1`。
- **抽取式去重**：`consolidation.deduplicate` 在共享实体簇内，用 token Jaccard ≥ 阈值判定近重复节点，合并边、保留代表节点。**不做生成式改写**（无模型即不臆造新文本）。

## 6. 上下文安全（防止大规模召回撑爆上下文）

两道闸门，均在 `core.retrieve` / `compile_to_markdown`：

1. `max_results`：硬性限制返回节点数。
2. `token_budget`：按 `estimate_tokens` 估算累计 token，超预算即停止纳入。

无此二者时，大池召回可达**百万级 token**（见 BENCHMARK）；开启后稳定在数百 token。

## 7. 依赖与持久化

- 运行依赖：仅 `numpy`（BM25/图/抽取均为纯 Python；numpy 仅供 Random Indexing）。
- 持久化：`NodeStore` 可选 sqlite 单文件存 content，按需懒加载；图与索引为进程内结构。
- 无外部 daemon、无网络端点、无 GPU。

详见 `docs/BENCHMARK.md` 的实测召回率 / 精度 / 多跳 / 按需加载 / 上下文预算结果。
