"""
记忆质量修复脚本
修复问题：
1. content_body 中含 [数字] 格式的 Element ID（跨 episode 无效，误导 Agent）
2. content_body 是 Python dict 字符串（JSON 解析失败遗留）
3. 统计修复前后数据
"""

import json
import re
import sys
import shutil
from pathlib import Path

ELEMENT_ID_RE = re.compile(r'\[(\d+)\]')
PYTHON_DICT_RE = re.compile(r"^\{.*'memory_description'.*'content_body'.*\}", re.DOTALL)


def clean_content_body(content: str) -> str:
    """移除 [数字] 格式的 Element ID，替换为语义占位符"""
    return ELEMENT_ID_RE.sub('[<element_id>]', content)


def fix_python_dict_node(content: str) -> str:
    """
    如果 content_body 是 Python dict 字符串（JSON 解析失败残留），
    尝试提取其中的 content_body 值。
    """
    if not content.startswith('{'):
        return content

    # 尝试用 ast.literal_eval 解析
    import ast
    try:
        d = ast.literal_eval(content)
        if isinstance(d, dict) and 'content_body' in d:
            return d['content_body']
    except Exception:
        pass

    # fallback: 正则提取
    m = re.search(r"'content_body'\s*:\s*'(.*?)'(?:\s*\}|,\s*')", content, re.DOTALL)
    if m:
        return m.group(1).replace("\\'", "'")

    return content  # 无法修复，保留原文


def fix_tree_file(filepath: str) -> dict:
    """修复单个树文件，返回统计信息"""
    path = Path(filepath)
    if not path.exists():
        print(f"[SKIP] 文件不存在: {filepath}")
        return {}

    # 备份
    backup = path.with_suffix('.json.bak')
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[BACKUP] {backup}")

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get('nodes', [])
    stats = {
        'total': len(nodes),
        'element_id_fixed': 0,
        'python_dict_fixed': 0,
        'already_clean': 0,
    }

    for node in nodes:
        payload = node.get('payload', {})
        content = payload.get('content_body', '')

        if not content or 'GLOBAL_ROOT_PLACEHOLDER' in payload.get('scenario_description', ''):
            stats['already_clean'] += 1
            continue

        changed = False

        # Fix 1: Python dict 字符串
        if content.startswith('{'):
            fixed = fix_python_dict_node(content)
            if fixed != content:
                payload['content_body'] = fixed
                content = fixed
                changed = True
                stats['python_dict_fixed'] += 1

        # Fix 2: Element IDs
        if ELEMENT_ID_RE.search(content):
            payload['content_body'] = clean_content_body(content)
            changed = True
            stats['element_id_fixed'] += 1

        if not changed:
            stats['already_clean'] += 1

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return stats


def main():
    base = Path('/hdd/REDACTED_USER/DeltaMem/Mind2web/storage')
    files = {
        'task_tree': str(base / 'prtree_dual_mind2web_task.json'),
        'env_tree':  str(base / 'prtree_dual_mind2web_env.json'),
        'online_task': str(base / 'online-test_task-Qwen3-14B-prtree_task.json'),
        'online_env':  str(base / 'online-test_task-Qwen3-14B-prtree_env.json'),
    }

    for name, fp in files.items():
        print(f"\n=== 处理 {name}: {fp} ===")
        stats = fix_tree_file(fp)
        if stats:
            print(f"  总节点: {stats['total']}")
            print(f"  Element ID 修复: {stats['element_id_fixed']}")
            print(f"  Python dict 修复: {stats['python_dict_fixed']}")
            print(f"  无需修复: {stats['already_clean']}")

    print("\n✅ 完成")


if __name__ == '__main__':
    main()
