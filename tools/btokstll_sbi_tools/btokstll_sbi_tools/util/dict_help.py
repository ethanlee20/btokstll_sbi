
from typing import Any


def access_nested_dict(
    d:dict, 
    *keys:Any
) -> Any:
    
    for k in keys: 
        d = d[k] 
    return d
    

def flatten_dict(
    d:dict, 
    *keys:str,
) -> dict:
    
    if not isinstance(
        v := access_nested_dict(d, *keys), 
        dict
    ):
        return {"_".join(keys): v}
    
    d_ = {}
    for k in v:
        d_ = d_ | flatten_dict(d, *keys, k)

    return d_






if __name__ == "__main__":

    d = {"a": {"d": 4, "e": {"p": 1, "g":2}}, "b": 2, "c": 3}
    print(flatten_dict(d))