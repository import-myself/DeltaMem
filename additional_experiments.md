# DeltaMem 补充实验方案

> 围绕系统核心贡献点构建完整的论文逻辑链，在现有主实验和消融实验之外，从多个维度补充验证 DeltaMem 的设计选择。

---

## 一、现有消融实验（Ablation Study，已有）

### 1.1 记忆模式消融（Memory Mode Ablation）

**脚本：** `ablation/run_memory_mode_ablation.py`

**对比组：** `task_only`（仅任务树注入 Prompt）vs `env_only`（仅环境树注入 Prompt）vs 完整双树

**覆盖 Benchmark：** ALFWorld / ScienceWorld / Mind2Web

**验证点：** 任务树和环境树各自的独立贡献，证明双树协作的必要性。

---

### 1.2 检索阈值消融（Threshold Ablation）

**脚本：** `ablation/run_threshold_ablation.py`

**参数网格：**
- 任务树基础阈值 `task_base ∈ {0.70, 0.75, 0.80, 0.85, 0.90}`，深度步长 `task_step ∈ {0.01, 0.02, 0.03, 0.04, 0.05}`
- 环境树基础阈值 `env_base ∈ {0.82, 0.85, 0.88, 0.91, 0.94}`，深度步长 `env_step ∈ {0.01, 0.02, 0.03, 0.04}`

**覆盖 Benchmark：** ALFWorld（主）

**验证点：** 动态深度阈值超参的敏感性，支撑最优配置选择。

---

### 1.3 联合阈值搜索（Joint Threshold Search）

**脚本：** `ablation/run_joint_threshold_search.py`

**方法：** 双树阈值联合网格搜索（粗粒度 3×3×3×3 = 81 组，再在最优点附近细化）

**覆盖 Benchmark：** ALFWorld / ScienceWorld / Mind2Web

**验证点：** 跨 Benchmark 的最优阈值配置，为 Implementation Details 中的超参表提供实验依据。

---

## 二、新增补充实验

### 2.1 快慢双路路由消融（Fast/Slow Path Routing Ablation）

**脚本：** `ablation/run_routing_ablation.py`（待实现）

**动机：**
DeltaMem 的核心架构贡献是快慢双系统路由：SkillCache O(1) 命中走快路，DFS 残差检索走慢路。目前没有实验直接对比两条路径的性能差异，读者无法判断 SkillCache 对整体成功率的实际增益，也无法看出两条路径何时互补。

**对比组（4 组）：**

| 组别 | 描述 |
|------|------|
| `skill_only` | 只用 SkillCache 快路；未命中时退化为无记忆 |
| `episodic_only` | 只用慢路 DFS 残差检索；不编译 Skill |
| `dual_routing` | SkillCache 优先，未命中回退慢路（完整系统） |
| `no_memory` | 无记忆注入（参考组，复用主实验数据） |

**实验设置：**
- Benchmark：ALFWorld `eval_in_distribution`（样本 140，成本可控）
- 固定 `icl_num=1`，复用已有离线记忆库

**统计指标：**
- 主：`success_rate`
- 辅：`skill_hit_rate`（快路命中率）、`avg_prompt_token_count`（token 消耗）、`avg_retrieval_latency`（检索延迟）

**期望结论：**
`dual_routing` > `episodic_only` > `skill_only` > `no_memory`。`skill_only` 在技能库稀疏时明显低于 `episodic_only`，证明慢路对冷启动场景不可或缺；随记忆积累快路命中率上升，两路径形成互补。

---

### 2.2 训练集离线构建记忆 → 测试集评估（Train-to-Test Memory Transfer）

**脚本：** `ablation/run_train_to_test.py`（待实现）

**动机：**
现有主实验采用在线学习设定（边跑边积累记忆），Synapse / AWM / ReasoningBank 等基线也基于同样设定。但更贴近真实部署的问题是：能否用训练集经验构建离线记忆库，直接迁移到测试集？该实验同时验证 DeltaMem 记忆的跨-split 泛化能力，并通过**记忆库大小**指标衡量各方法的存储效率。

**Benchmark 与 Split：**

| Benchmark | 记忆构建 split | 评估 split |
|-----------|---------------|------------|
| ALFWorld | `train`（~3500 个任务） | `eval_in_distribution`（140） |
| ScienceWorld | `train` | `dev`（194） |

Mind2Web 不纳入（train/test 按网站/域划分，任务类型差异显著，不适合此设定）。

**实验流程：**

```
train split → run agent → collect trajectories
    → build offline memory（各方法各自构建）→ freeze
    → eval on test split（冻结记忆，不在评估时更新）
```

所有方法使用同一批 train 轨迹（DeltaMem 在 train 上跑一遍收集，其他方法从这批轨迹中提取记忆），评估时一律冻结记忆，确保对比公平。

**对比方法（5 组）：**

| 方法 | 记忆形式 | 构建脚本 |
|------|---------|---------|
| `DeltaMem` | 双树（TaskTree + EnvTree）+ SkillCache | 参考 `ALFWorld/agent_alfworld_dual.py` 在线流程，改为离线 build |
| `Synapse` | FAISS 向量索引 + exemplar | `ALFWorld/build_synapse_memory.py` / `ScienceWorld/build_synapse_memory.py` |
| `AWM` | 自由文本规则库 | 补写 offline build 脚本（参考现有 AWM 在线流程） |
| `ReasoningBank` | 结构化推理记忆 | 补写 offline build 脚本 |
| `no_memory` | — | — |

**统计指标：**

主要性能指标：
- `success_rate`（ALFWorld / ScienceWorld 主指标）
- `task_hit_rate` / `env_hit_rate`（DeltaMem 专属，反映 train→test 检索命中率）
- ScienceWorld 按 `task_type` 细分成功率（观察哪些类型跨-split 迁移效果好/差）

记忆库大小（所有方法均需统计，用于存储效率对比）：

| 方法 | 度量方式 |
|------|---------|
| DeltaMem | 树节点总数（`task_tree_total_nodes + env_tree_total_nodes`）；磁盘大小（JSON，MB） |
| Synapse | FAISS 索引向量数（exemplar 条数）；磁盘大小（MB） |
| AWM | 规则条数；磁盘大小（MB） |
| ReasoningBank | 推理记忆条数；磁盘大小（MB） |

> 用 `success_rate vs. memory_size` 散点图呈现各方法的"存储-性能效率"，可作为 Analysis 小节的核心可视化。

**期望结论：**
1. DeltaMem 在 train→test 设定下成功率优于或持平其他基线，证明 PR-Tree 层次结构对跨-split 知识迁移更鲁棒。
2. 所有方法在 train→test 设定下成功率普遍低于在线设定，这一 gap 侧面说明在线学习的价值。
3. DeltaMem 跨-split 性能衰减最小：树状残差结构对分布偏移容忍度更高（未命中时可回退到父节点经验），而 Synapse / AWM 等方法衰减更明显。
4. DeltaMem 在相近成功率下节点数显著少于 Synapse 全量 exemplar，体现残差压缩（多个相似任务共享 Root 节点，只存 Delta）的存储优势。

---

### 2.3 Skill 固化阈值消融（Consolidation Threshold Ablation）

**脚本：** `ablation/run_consolidation_ablation.py`（待实现）

**动机：**
Skill 固化的核心超参是 `CONSOLIDATION_THRESHOLD`：节点 `success_count` 达到阈值后才将残差链异步编译为 ProceduralSkillPatch 存入 SkillCache。阈值太低导致 Skill 过拟合单次经验（质量差）；阈值太高导致固化速度太慢（快路长期空置）。这是双系统路由得以有效运转的关键设计选择，必须用实验支撑。

**参数网格：** `threshold ∈ {1, 2, 3, 5, 8}`

**实验设置：**
- Benchmark：ALFWorld `eval_in_distribution`
- 以相同的初始记忆库（离线构建，仅保留树结构，清空 SkillCache）为起点
- 顺序执行评估集，触发固化后记录 SkillCache 状态

**统计指标：**
- 主：`success_rate`
- 辅：`skill_cache_size`（固化出多少 Skill）、`skill_avg_reuse_count`（平均每个 Skill 被命中次数）、`skill_hit_rate`（快路命中率）

**期望结论：**
`threshold=3` 在成功率上取得最优或次优；`skill_avg_reuse_count` 随 threshold 增大而提高（高阈值的 Skill 质量更高但数量稀少）；呈现 precision-recall 式的权衡曲线，为默认值 $K_{\text{cons}}=3$ 提供实验依据。

---

## 三、优先级建议

| 优先级 | 实验 | 理由 |
|--------|------|------|
| ★★★ | 2.1 快慢路由消融 | 直接验证核心架构贡献（双系统路由），论文逻辑链的关键环节 |
| ★★★ | 2.2 Train→Test 迁移 | 验证跨-split 泛化能力 + 存储效率，贴近真实部署场景，对比更公平 |
| ★★☆ | 2.3 Skill 固化阈值消融 | 支撑 $K_{\text{cons}}$ 默认值选择，给出 precision-recall 式权衡曲线，适合放 Appendix |
