def get_prompt_key(is_root, is_success):
    if is_root:
        return "root_success" if is_success else "root_failure"
    else:
        return "node_success" if is_success else "node_failure"