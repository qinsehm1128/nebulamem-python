# **NebulaMem: 基于 Zvec 与 LadybugDB 的嵌入式记忆架构设计说明书**

# **1. 软件架构拓扑 (Software Topology - Local-First & In-Process)**

NebulaMem 采用原生嵌入式架构，安全、私密、高速运行，摒弃传统的 Client-Server 模式。它直接在宿主进程（如 IDE 插件或终端 CLI 进程）中以内置库形式运行。该设计旨在实现亚毫秒级的本地向量检索与图遍历，为 Agent 提供极高性能的长效记忆。

```text
[ Host Application Process (IDE / CLI) ]
          │
          ▼ [ Embedded DB Library API ]
┌──────────────────────────────────────┐
│         NebulaMem Core Module        │
│  ┌──────────────┐  ┌──────────────┐  │
│  │     Zvec     │  │  LadybugDB   │  │
│  │ (Vector Std) │  │ (Graph Rel)  │  │
│  └──────────────┘  └──────────────┘  │
└───────────┬──────────────────┬───────┘
            │ (Direct File IO) │
            ▼                  ▼
      [ zvec_data.db ]   [ ladybug_data ]
```

# **2. 数据层存储与 Schema 设计 (Data Layer & Schema Design)**

系统通过 LadybugDB（嵌入式图数据库）或高精度内置等价物存储实体间带权关系拓扑，通过 Zvec 存储高维密集向量索引。

## **2.1 LadybugDB 原生 Cypher Schema**

```cypher
CREATE NODE TABLE MemoryNode(id STRING, content STRING, type STRING, createdAt INT64, lastActivated INT64, PRIMARY KEY(id));
CREATE REL TABLE LINK(FROM MemoryNode TO MemoryNode, weight DOUBLE, type STRING, updatedAt INT64);
```

## **2.2 Zvec 向量集合 Schema**

```python
collection = zvec.createCollection({
  "name": "memory_vectors",
  "dimension": 1536,
  "metric": "cosine",
  "wal": True
})
```

# **3. 激活扩散与侧向抑制算法数学模型 (Mathematical Model)**

当输入查询向量 $Q$ 时，系统执行基于生物神经网络模拟的电信号激活逻辑：

## **3.1 寻找种子点 (Seeds)**

系统首先通过 Zvec 进行向量相似度检索，获取初始激活节点集合 $S$。随后利用 LadybugDB 的拓扑扩散接口或 Cypher 查询：

```cypher
MATCH (a:MemoryNode {id: $sourceId})-[r:LINK]->(b:MemoryNode)
RETURN b.id AS target, r.weight AS weight, r.type AS type
```

$$S = \{n \in Nodes \mid CosSim(Embedding(n), Q) \ge 0.25\}$$

## **3.2 扩散传播迭代 (Spreading Activation with Inhibitory Modulation)**

在扩散过程中，能量依据突触边权重 $W$ 传导。对于关联关系（`association` 或 `temporal_sequence`）执行正向增益传导，对于 `inhibitory`（抑制性）类型关系执行侧向抑制，防止激活范围无序泛化，实现自适应去冲突。

1.  **正向关联扩散**: 若关系 $(m \to n)$ 为 association 或 temporal_sequence：  
    $$\Delta A(n) = A(m) \cdot Weight(m \to n) \cdot \lambda$$  
    其中 $\lambda$ 为衰减因子。
2.  **抑制性关联扩散**: 若关系 $(m \to n)$ 为 inhibitory：  
    $$\Delta A(n) = - A(m) \cdot Weight(m \to n) \cdot \beta$$  
    其中 $\beta$ 为抑制系数。
3.  **汇总更新**:  
    $$A_{t+1}(n) = \max\left(-1.0, \min\left(1.0, A_t(n) + \Delta A(n)\right)\right)$$

# **4. 突触衰减与后台遗忘机制 (Synaptic Decay & Forgetting)**

在非活动周期，系统通过后台线程遍历所有突触连接执行边权重衰减，模拟生物脑的遗忘曲线：

```cypher
MATCH ()-[r:LINK]->()
SET r.weight = r.weight * exp(-decay_rate * (current_time - r.updatedAt))
```

在系统内存实现中，衰减公式简化为：  
`weight = Math.max(0.1, weight - idle_time * decay_rate)`

# **5. 梦境整理（Memory Consolidation）与 Wiki 生成**

激活后的"点亮星座图景"（Fired Constellation）将根据权重与类型被分类编译，直接用于智能体提示词注入：

  - `### ✦ STATE MEMORIES`: 记录状态相关的动态记忆。
  - `### ✦ FACT MEMORIES`: 记录事实性的静态知识。

该方案具有高健壮性，完全支持无损的降级适配。