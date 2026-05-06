"""
awm_memory.py  —  PRTree 通用 AWM (Autonomous Workflow Memory) 后端
====================================================================
实现"执行-评估-诱导-迭代"闭环：
  1. 智能体执行任务，生成完整轨迹
  2. 评估模块判断成功/失败
  3. 成功时 LLM 将轨迹抽象为可复用 workflow（变量化具体参数）
  4. Workflow 写入记忆库，作为下一任务的 system prompt 上下文
  5. 新轨迹在已有 workflow 基础上成功时，触发层次化合并（雪球效应）

每类任务（task_type / website）维护独立 workflow 文件，按类别分类存储。

用法
----
from awm_memory import AWMMemory

store = AWMMemory(
    memory_path="storage/awm_memory",
    llm_client=llm_client,       # PRTree LLMClient 实例
    benchmark="alfworld",        # "alfworld" | "sciworld" | "mind2web"
)

# 每 episode 前：获取当前 workflow 字符串注入 prompt
workflow_str = store.get_workflow(task_type)

# 每 episode 后（成功时）：诱导新 workflow
store.induce_and_update(
    task_type=task_type,
    task_description=task_instruction,
    trajectory=trajectory_list,   # list of str: ["Obs: ...", "Action: ...", ...]
    success=True,
)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

# --- ALFWorld ---
ALFWORLD_INDUCTION_INSTRUCTION = """\
You are given a successful trajectory of an agent completing a household task.
Your job is to extract a reusable workflow from this trajectory.

Rules:
1. Abstract concrete object names / locations into descriptive variable names like {object-name}, {target-location}.
2. Keep the high-level action structure, removing irrelevant exploration steps.
3. The workflow must have at least 2 steps.
4. Output format:
## <workflow_name>
Given <context description>, this workflow <what it does>.
Step 1: <action>
Step 2: <action>
...

Output ONLY the workflow block(s). Do not include any other text."""

ALFWORLD_MERGE_INSTRUCTION = """\
You have an existing workflow and a new successful trajectory for the same task type.
Merge them into a single, more complete workflow that covers both cases.

Rules:
1. Keep abstract variable names like {object-name}.
2. If the new trajectory adds new steps or alternatives, incorporate them.
3. Output format: same as existing workflow (## name, docstring, steps).
4. Output ONLY the merged workflow block. Do not include any other text."""

# --- ScienceWorld ---
SCIWORLD_INDUCTION_INSTRUCTION = """\
You are given a successful trajectory of an agent completing a science experiment task.
Your job is to extract a reusable workflow from this trajectory.

Rules:
1. Abstract specific object names / measurements into variables like {object-name}, {target-value}.
2. Keep the essential experimental procedure steps.
3. The workflow must have at least 2 steps.
4. Output format:
## <workflow_name>
Given <context description>, this workflow <what it does>.
Step 1: <action>
Step 2: <action>
...

Output ONLY the workflow block(s). Do not include any other text."""

SCIWORLD_MERGE_INSTRUCTION = """\
You have an existing workflow and a new successful trajectory for the same science task type.
Merge them into a single, more complete workflow.

Rules:
1. Keep abstract variable names.
2. Incorporate new steps or alternatives from the new trajectory.
3. Output format: same as existing workflow.
4. Output ONLY the merged workflow block. Do not include any other text."""

# --- Mind2web ---
MIND2WEB_INDUCTION_INSTRUCTION = """\
Given a successful web navigation task trajectory, extract a reusable workflow.

Rules:
1. Abstract specific values (URLs, names, dates) into descriptive variables like {product-name}, {target-date}.
2. Find the repeatable sub-sequence of actions.
3. Each workflow must have at least 2 steps.
4. Output format:
## <workflow_name>
Given <context>, this workflow <what it does>.
[element_type]  element_description -> ACTION: {variable}
...

Output ONLY the workflow block(s). Do not include any other text."""

MIND2WEB_MERGE_INSTRUCTION = """\
You have an existing workflow and a new successful web navigation trajectory for the same website.
Merge them into a single, more comprehensive workflow.

Rules:
1. Keep abstract variable names.
2. Add new action steps or branches from the new trajectory.
3. Output ONLY the merged workflow block. Do not include any other text."""

INDUCTION_INSTRUCTIONS = {
    "alfworld":  ALFWORLD_INDUCTION_INSTRUCTION,
    "sciworld":  SCIWORLD_INDUCTION_INSTRUCTION,
    "mind2web":  MIND2WEB_INDUCTION_INSTRUCTION,
}

MERGE_INSTRUCTIONS = {
    "alfworld":  ALFWORLD_MERGE_INSTRUCTION,
    "sciworld":  SCIWORLD_MERGE_INSTRUCTION,
    "mind2web":  MIND2WEB_MERGE_INSTRUCTION,
}


# ============================================================
# AWMMemory
# ============================================================

class AWMMemory:
    """
    AWM (Autonomous Workflow Memory) 核心类。

    每类任务（task_type）维护一个 workflow 文件：
      {memory_path}/{benchmark}/{task_type}.txt

    Parameters
    ----------
    memory_path : str
        记忆库根目录。
    llm_client : any
        PRTree LLMClient 实例（需有 .chat(messages, temperature) 方法）。
    benchmark : str
        "alfworld" | "sciworld" | "mind2web"
    induction_every : int
        每 N 个成功 episode 触发一次诱导（默认 1 = 每次成功都诱导）。
    max_workflow_tokens : int
        workflow 注入 prompt 时的最大字符数（防止 context 过长）。
    """

    def __init__(
        self,
        memory_path: str,
        llm_client,
        benchmark: str = "alfworld",
        induction_every: int = 1,
        max_workflow_tokens: int = 2000,
    ):
        self.memory_path   = Path(memory_path)
        self.benchmark     = benchmark.lower()
        self.llm_client    = llm_client
        self.induction_every = induction_every
        self.max_workflow_tokens = max_workflow_tokens
        self._lock         = threading.Lock()

        # 每类任务的成功轨迹缓存（等待诱导）
        self._pending: Dict[str, List[dict]] = {}
        # 成功计数
        self._success_counts: Dict[str, int] = {}

        # 建目录
        self._wf_dir = self.memory_path / self.benchmark
        self._wf_dir.mkdir(parents=True, exist_ok=True)

        # 加载已有 workflow 统计
        existing = list(self._wf_dir.glob("*.txt"))
        logger.info(f"✅ AWMMemory initialized: benchmark={benchmark}, "
                    f"path={self._wf_dir}, existing={len(existing)} workflow files")

    # ----------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------

    def get_workflow(self, task_type: str) -> Optional[str]:
        """
        获取指定任务类型的当前 workflow 字符串（用于注入 system prompt）。
        若无 workflow 返回 None。
        """
        wf_file = self._workflow_file(task_type)
        if not wf_file.exists():
            return None
        text = wf_file.read_text(encoding="utf-8").strip()
        if not text:
            return None
        # 截断防止 token 超限
        if len(text) > self.max_workflow_tokens:
            text = text[: self.max_workflow_tokens] + "\n...[truncated]"
        return text

    def induce_and_update(
        self,
        task_type: str,
        task_description: str,
        trajectory: List[str],
        success: bool,
    ) -> None:
        """
        episode 结束后调用。
        - 若 success=False，丢弃轨迹（防止错误污染）。
        - 若 success=True，将轨迹加入 pending 缓存，触发诱导条件时诱导 workflow。
        """
        if not success:
            logger.debug(f"[AWM] Episode failed for '{task_type}', discarding trajectory.")
            return

        with self._lock:
            if task_type not in self._pending:
                self._pending[task_type] = []
                self._success_counts[task_type] = 0
            self._pending[task_type].append({
                "task_description": task_description,
                "trajectory": trajectory,
            })
            self._success_counts[task_type] += 1
            count = self._success_counts[task_type]

        if count % self.induction_every == 0:
            self._induce(task_type)

    def save(self) -> None:
        """强制将所有 pending 轨迹诱导并保存（运行结束时调用）。"""
        for task_type in list(self._pending.keys()):
            if self._pending[task_type]:
                self._induce(task_type)
        logger.info(f"💾 AWMMemory saved all workflows → {self._wf_dir}")

    def list_workflows(self) -> Dict[str, str]:
        """返回所有已有 workflow 的 {task_type: content} 字典。"""
        result = {}
        for f in self._wf_dir.glob("*.txt"):
            task_type = f.stem
            result[task_type] = f.read_text(encoding="utf-8").strip()
        return result

    # ----------------------------------------------------------
    # 内部：诱导
    # ----------------------------------------------------------

    def _induce(self, task_type: str) -> None:
        """从 pending 轨迹中诱导 workflow，并与已有 workflow 合并。"""
        with self._lock:
            pending = self._pending.pop(task_type, [])
        if not pending:
            return

        existing_wf = self.get_workflow(task_type)

        # 格式化轨迹
        traj_text = self._format_trajectories(pending)

        try:
            if existing_wf:
                # 已有 workflow → 合并（雪球效应）
                new_wf = self._call_merge(task_type, existing_wf, traj_text)
                logger.info(f"[AWM] Merged workflow for '{task_type}' "
                            f"({len(pending)} new trajectories)")
            else:
                # 冷启动 → 首次诱导
                new_wf = self._call_induction(task_type, traj_text)
                logger.info(f"[AWM] Induced new workflow for '{task_type}' "
                            f"({len(pending)} trajectories)")

            if new_wf and new_wf.strip():
                self._save_workflow(task_type, new_wf.strip())
        except Exception as e:
            logger.warning(f"[AWM] Induction failed for '{task_type}': {e}")
            # 失败时把 pending 放回（下次重试）
            with self._lock:
                if task_type not in self._pending:
                    self._pending[task_type] = []
                self._pending[task_type] = pending + self._pending[task_type]

    def _call_induction(self, task_type: str, traj_text: str) -> str:
        """调用 LLM 从轨迹中诱导 workflow。"""
        instruction = INDUCTION_INSTRUCTIONS.get(self.benchmark, ALFWORLD_INDUCTION_INSTRUCTION)
        prompt = (
            f"{instruction}\n\n"
            f"Task Type: {task_type}\n\n"
            f"{traj_text}\n\n"
            f"## Extracted Workflow:"
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages, temperature=0.0)
        return response.strip()

    def _call_merge(self, task_type: str, existing_wf: str, traj_text: str) -> str:
        """调用 LLM 合并已有 workflow 和新轨迹。"""
        instruction = MERGE_INSTRUCTIONS.get(self.benchmark, ALFWORLD_MERGE_INSTRUCTION)
        prompt = (
            f"{instruction}\n\n"
            f"Task Type: {task_type}\n\n"
            f"=== Existing Workflow ===\n{existing_wf}\n\n"
            f"=== New Successful Trajectories ===\n{traj_text}\n\n"
            f"## Merged Workflow:"
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages, temperature=0.0)
        return response.strip()

    # ----------------------------------------------------------
    # 内部：格式化
    # ----------------------------------------------------------

    def _format_trajectories(self, pending: List[dict]) -> str:
        """将 pending 轨迹列表格式化为 prompt 可读文本。"""
        parts = []
        for i, item in enumerate(pending, 1):
            traj_str = "\n".join(item["trajectory"])
            parts.append(
                f"## Trajectory {i}\n"
                f"Task: {item['task_description']}\n"
                f"{traj_str}"
            )
        return "\n\n".join(parts)

    # ----------------------------------------------------------
    # 内部：文件 I/O
    # ----------------------------------------------------------

    def _workflow_file(self, task_type: str) -> Path:
        safe_name = task_type.replace("/", "_").replace(" ", "_").replace(":", "_")
        return self._wf_dir / f"{safe_name}.txt"

    def _save_workflow(self, task_type: str, content: str) -> None:
        wf_file = self._workflow_file(task_type)
        wf_file.write_text(content, encoding="utf-8")
        logger.info(f"[AWM] Workflow saved: {wf_file} ({len(content)} chars)")

    # ----------------------------------------------------------
    # 特殊方法
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        n = len(list(self._wf_dir.glob("*.txt")))
        return f"AWMMemory(benchmark={self.benchmark}, path={self._wf_dir}, n_workflows={n})"
