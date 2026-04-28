
from typing import Any


def are_instance(
    objs:Any, 
    classinfo:Any,
)-> bool:
    for o in objs:
        if not isinstance(o, classinfo):
            return False
    return True




