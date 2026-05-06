"""
Trajectory Logger Module

保存运行轨迹的模块，支持：
- 单个 episode 轨迹
- 批量 episodes 轨迹
- JSON 格式持久化
- 详细的运行信息记录
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """轨迹记录器"""
    
    def __init__(self, output_dir: str = "trajectories"):
        """
        初始化轨迹记录器
        
        Args:
            output_dir: 轨迹保存目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories: List[Dict[str, Any]] = []
        
        logger.info(f"📝 Trajectory logger initialized: {self.output_dir}")
    
    def log_episode(
        self,
        episode_idx: int,
        task_instruction: str,
        task_type: str,
        result: Dict[str, Any],
        split: str = "train",
        mode: str = "offline"
    ):
        """
        记录单个 episode 的轨迹
        
        Args:
            episode_idx: Episode 索引
            task_instruction: 任务指令
            task_type: 任务类型
            result: run_episode() 返回的结果字典
            split: 数据集分割
            mode: 运行模式
        """
        trajectory_data = {
            "episode_idx": episode_idx,
            "mode": mode,
            "split": split,
            "task_type": task_type,
            "task_instruction": task_instruction,
            "success": result.get("success", False),
            "steps": result.get("steps", 0),
            "memory_used": result.get("memory_used", False),
            "reflection": result.get("reflection", ""),
            "trajectory": result.get("trajectory", []),
            "timestamp": datetime.now().isoformat()
        }
        
        self.trajectories.append(trajectory_data)
    
    def save(
        self,
        filepath: Optional[str] = None,
        mode: str = "offline",
        split: str = "train"
    ):
        """
        保存所有轨迹到文件
        
        Args:
            filepath: 保存路径（如不指定，自动生成）
            mode: 运行模式
            split: 数据集分割
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"{mode}_{split}_{timestamp}.json"
        else:
            filepath = Path(filepath)
        
        # 准备保存数据
        save_data = {
            "mode": mode,
            "split": split,
            "total_episodes": len(self.trajectories),
            "success_count": sum(1 for t in self.trajectories if t["success"]),
            "trajectories": self.trajectories,
            "saved_at": datetime.now().isoformat()
        }
        
        # 保存到文件
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(self.trajectories)} trajectories to: {filepath}")
        return str(filepath)
    
    def clear(self):
        """清空当前轨迹"""
        self.trajectories = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取轨迹统计信息"""
        if not self.trajectories:
            return {
                "total": 0,
                "success": 0,
                "failure": 0,
                "success_rate": 0.0,
                "avg_steps": 0.0,
                "memory_hit_rate": 0.0
            }
        
        total = len(self.trajectories)
        success = sum(1 for t in self.trajectories if t["success"])
        memory_used = sum(1 for t in self.trajectories if t["memory_used"])
        total_steps = sum(t["steps"] for t in self.trajectories)
        
        return {
            "total": total,
            "success": success,
            "failure": total - success,
            "success_rate": success / total if total > 0 else 0.0,
            "avg_steps": total_steps / total if total > 0 else 0.0,
            "memory_hit_rate": memory_used / total if total > 0 else 0.0
        }


def load_trajectories(filepath: str) -> Dict[str, Any]:
    """
    加载保存的轨迹文件
    
    Args:
        filepath: 轨迹文件路径
    
    Returns:
        包含所有轨迹的字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"📂 Loaded {data['total_episodes']} trajectories from: {filepath}")
    return data


def analyze_trajectories(filepath: str) -> Dict[str, Any]:
    """
    分析轨迹文件，生成统计报告
    
    Args:
        filepath: 轨迹文件路径
    
    Returns:
        统计分析结果
    """
    data = load_trajectories(filepath)
    trajectories = data["trajectories"]
    
    total = len(trajectories)
    if total == 0:
        return {"error": "No trajectories found"}
    
    # 基础统计
    success_count = sum(1 for t in trajectories if t["success"])
    memory_used_count = sum(1 for t in trajectories if t["memory_used"])
    total_steps = sum(t["steps"] for t in trajectories)
    
    # 按任务类型统计
    task_type_stats = {}
    for t in trajectories:
        task_type = t["task_type"]
        if task_type not in task_type_stats:
            task_type_stats[task_type] = {
                "count": 0,
                "success": 0,
                "total_steps": 0
            }
        task_type_stats[task_type]["count"] += 1
        if t["success"]:
            task_type_stats[task_type]["success"] += 1
        task_type_stats[task_type]["total_steps"] += t["steps"]
    
    # 计算比率
    for stats in task_type_stats.values():
        stats["success_rate"] = stats["success"] / stats["count"] if stats["count"] > 0 else 0
        stats["avg_steps"] = stats["total_steps"] / stats["count"] if stats["count"] > 0 else 0
    
    # With/Without Memory 对比
    with_memory = [t for t in trajectories if t["memory_used"]]
    without_memory = [t for t in trajectories if not t["memory_used"]]
    
    return {
        "mode": data["mode"],
        "split": data["split"],
        "total_episodes": total,
        "success_rate": success_count / total,
        "avg_steps": total_steps / total,
        "memory_hit_rate": memory_used_count / total,
        "task_type_breakdown": task_type_stats,
        "with_memory": {
            "count": len(with_memory),
            "success_rate": sum(1 for t in with_memory if t["success"]) / len(with_memory) if with_memory else 0,
            "avg_steps": sum(t["steps"] for t in with_memory) / len(with_memory) if with_memory else 0
        },
        "without_memory": {
            "count": len(without_memory),
            "success_rate": sum(1 for t in without_memory if t["success"]) / len(without_memory) if without_memory else 0,
            "avg_steps": sum(t["steps"] for t in without_memory) / len(without_memory) if without_memory else 0
        }
    }
