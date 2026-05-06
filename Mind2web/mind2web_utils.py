"""
Mind2Web Utility Functions (v4.0)
整合自 DREM/mind2web/scripts/mind2web_utils.py
提供 HTML 处理、指标计算、数据加载等工具函数
"""

import copy
import json
import os
import pickle as pkl
import re
import string
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import tiktoken
from lxml import etree

# =================================================================
# Token 计数
# =================================================================

MAX_TOKENS = {
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
}


def num_tokens_from_messages(messages: List[Dict], model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


# =================================================================
# 动作解析与评估
# =================================================================

def parse_act_str(act_str: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """解析动作字符串，返回 (op, element_id, value)"""
    if not act_str:
        return None, None, None
    pattern = re.compile(r"(?:^|\s)(CLICK|SELECT|TYPE)?\s?\[(.+?)\](?:\s\[(.+?)\])?")
    match = pattern.search(act_str)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def construct_act_str(op: Optional[str], val: Optional[str]) -> str:
    if op is None:
        return " " if val is None else " " + val
    if op == "CLICK" or val is None:
        return op + " "
    return f"{op} {val}"


def calculate_f1(pred: str, label: str) -> float:
    pred_set  = set(w for w in pred.strip().split() if w not in string.punctuation)
    label_set = set(w for w in label.strip().split() if w not in string.punctuation)
    if not pred_set and not label_set:
        return 1.0
    if not pred_set or not label_set:
        return 0.0
    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_from_response(response: str, backtick: str = "```") -> str:
    """从 LLM 响应中提取反引号包裹的内容"""
    if backtick == "```":
        pattern = r"```(?:[a-zA-Z]*)\n?(.*?)\n?```"
    elif backtick == "`":
        pattern = r"`(.*?)`"
    else:
        raise ValueError(f"Unknown backtick: {backtick}")
    match = re.search(pattern, response, re.DOTALL)
    return match.group(1) if match else ""


# =================================================================
# HTML / DOM 处理
# =================================================================

def get_descendants(node, max_depth: int, current_depth: int = 0):
    if current_depth > max_depth:
        return []
    descendants = []
    for child in node:
        descendants.append(child)
        descendants.extend(get_descendants(child, max_depth, current_depth + 1))
    return descendants


def get_attribute_repr(node, max_value_length: int = 5, max_length: int = 20):
    attr_values_set = set()
    attr_values = ""
    for attr in [
        "role", "aria_role", "type", "alt", "aria_description", "aria_label",
        "label", "title", "name", "text_value", "value", "placeholder",
        "input_checked", "input_value", "option_selected", "class",
    ]:
        if attr in node.attrib and node.attrib[attr] is not None:
            value = node.attrib[attr].lower()
            if value in ["hidden", "none", "presentation", "null", "undefined"] or value.startswith("http"):
                continue
            value = " ".join(v for v in value.split() if len(v) < 15)
            if value and value not in attr_values_set:
                attr_values_set.add(value)
                attr_values += value + " "
    uid = node.attrib.get("backend_node_id", "")
    node.attrib.clear()
    if uid:
        node.attrib["id"] = uid
    if attr_values:
        node.attrib["meta"] = " ".join(attr_values.split()[:max_length])


def prune_tree(dom_tree, candidate_set, max_depth=5, max_children=50, max_sibling=3):
    nodes_to_keep = set()
    for candidate_id in candidate_set:
        candidate_node = dom_tree.xpath(f'//*[@backend_node_id="{candidate_id}"]')[0]
        nodes_to_keep.add(candidate_node.attrib["backend_node_id"])
        nodes_to_keep.update(
            x.attrib.get("backend_node_id", "")
            for x in candidate_node.xpath("ancestor::*")
        )
        nodes_to_keep.update(
            x.attrib.get("backend_node_id", "")
            for x in get_descendants(candidate_node, max_depth)
        )
        parent = candidate_node.getparent()
        if parent is not None:
            siblings = [x for x in parent.getchildren() if x.tag != "text"]
            idx = siblings.index(candidate_node)
            nodes_to_keep.update(
                x.attrib.get("backend_node_id", "")
                for x in siblings[max(0, idx - max_sibling): idx + max_sibling + 1]
            )
    new_tree = copy.deepcopy(dom_tree)
    for node in new_tree.xpath("//*")[::-1]:
        if node.tag != "text":
            is_keep = node.attrib.get("backend_node_id", "") in nodes_to_keep
            is_cand = node.attrib.get("backend_node_id", "") in candidate_set
        else:
            is_keep = node.getparent().attrib.get("backend_node_id", "") in nodes_to_keep
            is_cand = node.getparent().attrib.get("backend_node_id", "") in candidate_set
        if not is_keep and node.getparent() is not None:
            node.getparent().remove(node)
        else:
            if not is_cand or node.tag == "text":
                node.attrib.pop("backend_node_id", None)
            if (
                len(node.attrib) == 0
                and not any(x.tag == "text" for x in node.getchildren())
                and node.getparent() is not None
                and node.tag != "text"
                and len(node.getchildren()) <= 1
            ):
                for child in node.getchildren():
                    node.addprevious(child)
                node.getparent().remove(node)
    return new_tree


def get_tree_repr(tree, max_value_length=5, max_length=20, id_mapping=None, keep_html_brackets=False):
    if id_mapping is None:
        id_mapping = {}
    if isinstance(tree, str):
        tree = etree.fromstring(tree)
    else:
        tree = copy.deepcopy(tree)
    for node in tree.xpath("//*"):
        if node.tag != "text":
            if "backend_node_id" in node.attrib:
                if node.attrib["backend_node_id"] not in id_mapping:
                    id_mapping[node.attrib["backend_node_id"]] = len(id_mapping)
            get_attribute_repr(node, max_value_length, max_length)
        else:
            node.text = " ".join(node.text.split()[:max_length])
    tree_repr = etree.tostring(tree, encoding="unicode")
    tree_repr = tree_repr.replace('"', " ")
    tree_repr = tree_repr.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
    tree_repr = re.sub(r"<text>(.*?)</text>", r"\1", tree_repr)
    if not keep_html_brackets:
        tree_repr = tree_repr.replace("/>", "$/$>")
        tree_repr = re.sub(r"</(.+?)>", r")", tree_repr)
        tree_repr = re.sub(r"<(.+?)>", r"(\1", tree_repr)
        tree_repr = tree_repr.replace("$/$", ")")
    html_escape = [
        ("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&nbsp;", " "), ("&ndash;", "-"), ("&rsquo;", "'"), ("&lsquo;", "'"),
        ("&ldquo;", '"'), ("&rdquo;", '"'), ("&#39;", "'"), ("&#40;", "("), ("&#41;", ")"),
    ]
    for k, v in html_escape:
        tree_repr = tree_repr.replace(k, v)
    tree_repr = re.sub(r"\s+", " ", tree_repr).strip()
    return tree_repr, id_mapping


def get_target_obs(dom_tree, target_element_ids: List[str]) -> str:
    pruned = prune_tree(dom_tree, target_element_ids)
    tree_repr, _ = get_tree_repr(pruned, id_mapping={}, keep_html_brackets=True)
    return tree_repr


def get_target_act(example: Dict, target_element_id: str) -> str:
    action_op  = example["operation"]["op"]
    action_val = example["operation"]["value"]
    target_action = f"{action_op} [{target_element_id}]"
    if action_op != "CLICK":
        target_action += f" [{action_val}]"
    return target_action


def get_target_obs_and_act(example: Dict) -> Tuple[str, str]:
    if len(example["pos_candidates"]) == 0:
        dom_tree  = etree.fromstring(example["raw_html"])
        gt_elem   = dom_tree.xpath(f"//*[@data_pw_testid_buckeye='{example['action_uid']}']")
        elem_id   = gt_elem[0].get("backend_node_id")
        raw_obs   = get_target_obs(dom_tree, [elem_id])
        start_idx     = raw_obs.find(f"id={elem_id}")
        start_tag_idx = raw_obs.rfind("<", 0, start_idx)
        end_tag_idx   = raw_obs.find(">", start_idx)
        tag_name      = raw_obs[start_tag_idx + 1: end_tag_idx].split()[0]
        open_count = close_count = 0
        search_idx = start_tag_idx
        while True:
            next_open  = raw_obs.find(f"<{tag_name}", search_idx)
            next_close = raw_obs.find(f"</{tag_name}>", search_idx)
            if next_open == -1 and next_close == -1:
                break
            if next_open != -1 and (next_open < next_close or next_close == -1):
                open_count += 1
                search_idx = raw_obs.find(">", next_open) + 1
            else:
                close_count += 1
                search_idx = next_close + len(f"</{tag_name}>")
            if open_count == close_count:
                break
        o = f"<html> {raw_obs[start_tag_idx:search_idx]} </html>"
        a = get_target_act(example, elem_id)
    else:
        dom_tree = etree.fromstring(example["cleaned_html"])
        elem_id  = example["pos_candidates"][0]["backend_node_id"]
        o = get_target_obs(dom_tree, [elem_id])
        a = get_target_act(example, elem_id)
    return o, a


def get_top_k_obs(s: Dict, top_k: int, use_raw: bool = True) -> Tuple[str, List[str]]:
    pos_candidates = s["pos_candidates"]
    pos_ids = [c["backend_node_id"] for c in pos_candidates][:1]
    neg_candidates = sorted(s["neg_candidates"], key=lambda c: c["rank"])[: top_k - 1]
    neg_ids = [c["backend_node_id"] for c in neg_candidates]
    all_candidates = pos_ids + neg_ids
    obs = get_target_obs(etree.fromstring(s["cleaned_html"]), all_candidates)

    if len(s["pos_candidates"]) == 0:
        assert use_raw
        dom_tree = etree.fromstring(s["raw_html"])
        gt_elem  = dom_tree.xpath(f"//*[@data_pw_testid_buckeye='{s['action_uid']}']")
        elem_id  = gt_elem[0].get("backend_node_id")
        raw_obs  = get_target_obs(dom_tree, [elem_id])
        start_idx     = raw_obs.find(f"id={elem_id}")
        start_tag_idx = raw_obs.rfind("<", 0, start_idx)
        end_tag_idx   = raw_obs.find(">", start_idx)
        tag_name      = raw_obs[start_tag_idx + 1: end_tag_idx].split()[0]
        open_count = close_count = 0
        search_idx = start_tag_idx
        while True:
            next_open  = raw_obs.find(f"<{tag_name}", search_idx)
            next_close = raw_obs.find(f"</{tag_name}>", search_idx)
            if next_open == -1 and next_close == -1:
                break
            if next_open != -1 and (next_open < next_close or next_close == -1):
                open_count += 1
                search_idx = raw_obs.find(">", next_open) + 1
            else:
                close_count += 1
                search_idx = next_close + len(f"</{tag_name}>")
            if open_count == close_count:
                break
        obs = obs.replace("</html>", f"{raw_obs[start_tag_idx:search_idx]} </html>")
    return obs, all_candidates


# =================================================================
# 数据加载
# =================================================================

def add_scores(
    examples: List[Dict],
    candidate_results: Optional[Dict] = None,
    score_path: Optional[str] = None,
) -> List[Dict]:
    """为候选元素附加预测分数和排名"""
    if candidate_results is None:
        if score_path is None:
            score_path = str(Path(__file__).parent / "data" / "scores_all_data.pkl")
        with open(score_path, "rb") as f:
            candidate_results = pkl.load(f)
    for sample in examples:
        for s in sample["actions"]:
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    cid = candidate["backend_node_id"]
                    candidate["score"] = candidate_results["scores"][sample_id][cid]
                    candidate["rank"]  = candidate_results["ranks"][sample_id][cid]
    return examples


def load_json_data(data_dir: str, folder_name: str) -> List[Dict]:
    """加载指定文件夹下的所有 JSON 文件"""
    folder_path = os.path.join(data_dir, folder_name)
    print(f"Data path: {folder_path}")
    data_paths = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    samples = []
    for dp in data_paths:
        with open(dp, "r") as f:
            samples.extend(json.load(f))
    print(f"# of samples: {len(samples)}")
    return samples


def get_all_combinations(samples: List[Dict]) -> List[Tuple[str, str, str]]:
    """提取所有唯一的 (domain, subdomain, website) 组合"""
    combinations = set()
    for s in samples:
        combinations.add((s.get("domain", ""), s.get("subdomain", ""), s.get("website", "")))
    return sorted(list(combinations))


def save_results(results: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to {output_path}")


def calculate_metrics(results: List[Dict]) -> Dict[str, Any]:
    if not results:
        return {}
    total = len(results)
    success_count = sum(1 for r in results if r.get("success", False))
    all_elem_acc  = [v for r in results for v in r.get("element_acc", [])]
    all_act_f1    = [v for r in results for v in r.get("action_f1", [])]
    all_step_sr   = [v for r in results for v in r.get("step_success", [])]
    import numpy as np
    return {
        "total_tasks":        total,
        "success_count":      success_count,
        "task_success_rate":  success_count / total,
        "element_acc":        float(np.mean(all_elem_acc)) if all_elem_acc else 0.0,
        "action_f1":          float(np.mean(all_act_f1))  if all_act_f1  else 0.0,
        "step_success_rate":  float(np.mean(all_step_sr)) if all_step_sr  else 0.0,
        "memory_usage_rate":  sum(1 for r in results if r.get("memory_used", False)) / total,
    }
