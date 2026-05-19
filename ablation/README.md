# DeltaMem — Ablation & Supplementary Experiments

本目录包含 DeltaMem 所有消融实验与补充实验的脚本、运行说明和结果索引。

---

## 实验总览

| # | 实验名称 | 核心问题 | Benchmark | 脚本 | 结果 CSV |
|---|---------|---------|-----------|------|---------|
| 1 | 记忆模式消融 | 任务树 / 环境树各自的独立贡献？ | ALF · Sci · M2W | `run_memory_mode_ablation.py` | `results/memory_mode_ablation.csv` |
| 2 | 检索阈值消融 | 动态深度阈值超参敏感性？ | ALFWorld | `run_threshold_ablation.py` | `results/threshold_ablation.csv` |
| 3 | 联合阈值网格搜索 | 各 benchmark 最优阈值组合？ | ALF · Sci · M2W | `run_joint_threshold_search.py` | `results/joint_threshold_search_all_*.csv` |
| 4 | 快慢双路路由消融 | SkillCache 快路 vs DFS 慢路 各自增益？ | ALF · Sci · M2W | `run_routing_ablation.py` | `results/routing_ablation.csv` |
| 5 | Train→Test 记忆迁移 | 离线记忆跨-split 泛化与存储效率？ | ALF · Sci | `run_train_to_test.py` | `results/train_to_test.csv` |
| 6 | Skill 固化阈值消融 | K_cons 默认值 3 的 precision-recall 权衡？ | ALFWorld | `run_consolidation_ablation.py` | `results/consolidation_ablation.csv` |

---

## 前置条件

### 环境变量

所有 Shell 脚本已内嵌以下变量，修改后直接运行即可：

```bash
export ALFWORLD_DATA='/hdd/REDACTED_USER/DeltaMem/ALFWorld/data/alfworld'
export DEEPSEEK_API_KEY='sk-xxxx'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
```

### 离线记忆库

多数实验需要预先构建离线 PRTree 记忆库，路径约定如下：

| Benchmark | 默认路径 |
|-----------|---------|
| ALFWorld | `../ALFWorld/storage/prtree_dual_offline` |
| ScienceWorld | `../ScienceWorld/storage/prtree_sciworld_offline` |
| Mind2Web | `../Mind2web/storage/prtree_mind2web_offline` |

---

## 实验 1：记忆模式消融（Memory Mode Ablation）

**核心问题：** 任务树（TaskTree）和环境树（EnvTree）各自对成功率的独立贡献，证明双树协作的必要性。

**对比组：**

| 组别 | 描述 |
|------|------|
| `task_only` | 只将任务树记忆注入 Prompt；环境树仍写入但不注入 |
| `env_only` | 只将环境树记忆注入 Prompt；任务树仍写入但不注入 |

完整双树（`both`）的结果见主实验表。

**运行：**

```bash
cd /hdd/REDACTED_USER/DeltaMem/ablation

# 全量三个 benchmark（顺序执行）
bash run_memory_mode_ablation.sh

# 单独运行某一个 benchmark
bash run_memory_mode_alfworld.sh     # ALFWorld  eval_in_distribution（140 episodes）
bash run_memory_mode_sciworld.sh     # ScienceWorld  test split
bash run_memory_mode_mind2web.sh     # Mind2Web  test_task split
```

自定义参数示例：

```bash
python run_memory_mode_ablation.py \
    --benchmark            alfworld \
    --memory-modes         task_only,env_only \
    --model                deepseek-v4-flash \
    --alfworld-split       eval_in_distribution \
    --alfworld-load-memory ../ALFWorld/storage/prtree_dual_offline \
    --output-csv           results/memory_mode_ablation.csv
```

**统计指标：** `success_rate`, `avg_steps`, `task_hit_rate`, `env_hit_rate`, `avg_task_retrieval_len_*`, `task_tree_total_nodes`, `env_tree_total_nodes`

---

## 实验 2：检索阈值消融（Threshold Ablation）

**核心问题：** 任务树和环境树的动态深度阈值（base + depth × step）对召回精度和成功率的影响，为超参默认值提供实验依据。

**参数网格（单树扫描，固定另一棵树的默认值）：**

- 任务树：`task_base ∈ {0.70, 0.75, 0.80, 0.85, 0.90}`，`task_step ∈ {0.01, 0.02, 0.03, 0.04, 0.05}`
- 环境树：`env_base ∈ {0.82, 0.85, 0.88, 0.91, 0.94}`，`env_step ∈ {0.01, 0.02, 0.03, 0.04}`

**运行：**

```bash
bash run_threshold_ablation.sh
```

自定义参数示例：

```bash
python run_threshold_ablation.py \
    --mode         task \
    --model        deepseek-v4-flash \
    --split        eval_in_distribution \
    --load-memory  ../ALFWorld/storage/prtree_dual_offline \
    --max-episodes 50 \
    --output-csv   results/threshold_ablation.csv
```

**统计指标：** `task_base_threshold`, `task_depth_step`, `success_rate`, `avg_steps`, `task_hit_rate`, `avg_task_retrieval_len_hit`

---

## 实验 3：联合阈值网格搜索（Joint Threshold Search）

**核心问题：** 双树阈值的最优联合配置，支撑 Implementation Details 中的超参数表。

**搜索策略：**
- 粗粒度：`3×3×3×3 = 81` 组（约 10h per benchmark）
- 细粒度：在粗搜索最优点附近加密（脚本内注释中提供替换配置）

**运行：**

```bash
# 修改脚本内的 benchmark 变量（alfworld / sciworld / mind2web），然后
bash run_joint_threshold_search.sh
```

自定义参数示例：

```bash
python run_joint_threshold_search.py \
    --mode                  grid \
    --benchmark             alfworld \
    --alfworld-load-memory  ../ALFWorld/storage/prtree_dual_offline \
    --task-base-thresholds  "0.75,0.80,0.85" \
    --task-depth-steps      "0.01,0.03,0.05" \
    --env-base-thresholds   "0.82,0.88,0.94" \
    --env-depth-steps       "0.01,0.02,0.04" \
    --output-csv            results/joint_threshold_search_all
```

**统计指标：** `task_base`, `task_step`, `env_base`, `env_step`, `success_rate`, `task_hit_rate`, `env_hit_rate`

---

## 实验 4：快慢双路路由消融（Fast/Slow Path Routing Ablation）

**核心问题：** SkillCache O(1) 快路 vs DFS 残差慢路各自对成功率的贡献，以及两条路径何时互补。

**对比组：**

| 组别 | 描述 |
|------|------|
| `no_memory` | 无记忆注入（参考基线） |
| `skill_only` | 只走 SkillCache 快路；未命中退化为无记忆 |
| `episodic_only` | 只走 DFS 残差慢路；SkillCache 被 patch 禁用 |
| `dual_routing` | SkillCache 优先，未命中回退慢路（完整系统） |

**期望结论：** `dual_routing` > `episodic_only` > `skill_only` > `no_memory`

**前置要求：** 需预先构建含 SkillCache 的离线 PRTree 记忆库。

**运行：**

```bash
bash run_routing_ablation.sh
```

自定义参数示例：

```bash
# 仅跑 ALFWorld，快速验证
python run_routing_ablation.py \
    --benchmark            alfworld \
    --routing-modes        no_memory,skill_only,episodic_only,dual_routing \
    --alfworld-split       eval_in_distribution \
    --alfworld-load-memory ../ALFWorld/storage/prtree_dual_offline \
    --output-csv           results/routing_ablation.csv

# 全量三个 benchmark
python run_routing_ablation.py \
    --benchmark            all \
    --alfworld-load-memory ../ALFWorld/storage/prtree_dual_offline \
    --sciworld-load-memory ../ScienceWorld/storage/prtree_sciworld_offline \
    --mind2web-load-memory ../Mind2web/storage/prtree_mind2web_offline \
    --output-csv           results/routing_ablation.csv
```

**统计指标：** `success_rate`, `skill_hit_rate`, `skill_cache_size`, `task_hit_rate`, `env_hit_rate`, `avg_prompt_tokens`, `avg_retrieval_len_*`

---

## 实验 5：Train→Test 记忆迁移（Train-to-Test Memory Transfer）

**核心问题：** 用训练集离线积累记忆，冻结后迁移到测试集——验证跨-split 泛化能力与存储效率。

**Benchmark 与 split：**

| Benchmark | 记忆构建 split | 评估 split |
|-----------|--------------|-----------|
| ALFWorld | `train`（~3553 episodes） | `eval_in_distribution`（140） |
| ScienceWorld | `train` | `dev`（194） |

（Mind2Web 不纳入：train/test 按网站/域划分，任务类型差异过大。）

**对比方法：**

| 方法 | 记忆形式 |
|------|---------|
| `deltamem` | 双树 PRTree + SkillCache |
| `synapse` | FAISS 向量索引 exemplar（需预先用 `build_synapse_memory.py` 构建） |
| `no_memory` | 无记忆基线 |

**`--phase` 参数说明：**

| 值 | 含义 |
|----|------|
| `build` | 只在 train split 运行 agent 积累记忆并保存（可以单独预跑） |
| `eval` | 只加载已有冻结记忆，在 test split 评估（不更新记忆） |
| `all` | 先 build 后 eval（一站式，适合首次运行） |

**运行：**

```bash
bash run_train_to_test.sh
```

分阶段运行（已有离线记忆时直接 eval，跳过 build）：

```bash
python run_train_to_test.py \
    --benchmark              alfworld \
    --method                 deltamem \
    --phase                  eval \
    --alfworld-load-memory   ../ALFWorld/storage/prtree_train_offline \
    --output-csv             results/train_to_test.csv
```

同时跑多个方法：

```bash
python run_train_to_test.py \
    --benchmark              alfworld \
    --method                 deltamem,no_memory \
    --phase                  eval \
    --alfworld-load-memory   ../ALFWorld/storage/prtree_train_offline \
    --output-csv             results/train_to_test.csv
```

**统计指标：** `success_rate`, `avg_steps`, `skill_hit_rate`, `task_hit_rate`, `env_hit_rate`, `task_tree_total_nodes`, `env_tree_total_nodes`, `memory_disk_mb`

> `memory_disk_mb` 用于绘制**存储-性能效率散点图**（success_rate vs memory_size），直观对比各方法的压缩效率。

---

## 实验 6：Skill 固化阈值消融（Consolidation Threshold Ablation）

**核心问题：** `CONSOLIDATION_THRESHOLD`（节点 success_count 达到阈值才编译为 ProceduralSkillPatch）的大小对 Skill 质量与快路命中率的权衡，为默认值 K=3 提供实验依据。

**参数网格：** `K ∈ {1, 2, 3, 5, 8}`

**期望结论：** K=3 在成功率上取得最优或次优；`skill_avg_reuse_count` 随 K 增大而提升（高阈值 Skill 质量更高但数量更少）；呈现 precision-recall 式权衡曲线。

**前置要求：** 预构建的离线 PRTree（有树节点即可；SkillCache 由脚本自动清空，从零开始积累）。

**运行：**

```bash
bash run_consolidation_ablation.sh
```

自定义参数示例：

```bash
python run_consolidation_ablation.py \
    --thresholds  1,2,3,5,8 \
    --load-memory ../ALFWorld/storage/prtree_dual_offline \
    --split       eval_in_distribution \
    --model       deepseek-v4-flash \
    --output-csv  results/consolidation_ablation.csv
```

**统计指标：** `threshold`, `success_rate`, `avg_steps`, `skill_cache_size`, `skill_hit_rate`, `skill_avg_reuse_count`, `task_hit_rate`, `env_hit_rate`

---

## 结果文件索引

| CSV 文件 | 对应实验 |
|---------|---------|
| `results/memory_mode_ablation.csv` | 实验 1 |
| `results/threshold_ablation.csv` | 实验 2 |
| `results/joint_threshold_search_all_alfworld.csv` | 实验 3（ALFWorld） |
| `results/joint_threshold_search_all_sciworld.csv` | 实验 3（ScienceWorld） |
| `results/joint_threshold_search_all_mind2web.csv` | 实验 3（Mind2Web） |
| `results/routing_ablation.csv` | 实验 4 |
| `results/train_to_test.csv` | 实验 5 |
| `results/consolidation_ablation.csv` | 实验 6 |

所有 CSV 均为**追加写入**，支持断点续跑，多次运行不覆盖已有行。

---

## 常用命令

```bash
# 格式化查看 CSV
column -t -s, results/routing_ablation.csv | less -S

# 实时查看日志
tail -f logs/routing_ablation-alfworld-deepseek-v4-flash.log

# 统计已完成行数
wc -l results/*.csv
```
