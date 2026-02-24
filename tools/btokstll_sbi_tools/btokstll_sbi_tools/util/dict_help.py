
def get_nodes_nested_dict(
    nested:dict,
):
    nodes = {}
    def recurse(dict_):
        for k, v in dict_.items():
            if not isinstance(v, dict):
                nodes[k] = v
            else: 
                recurse(v)
    recurse(nested)
    return nodes