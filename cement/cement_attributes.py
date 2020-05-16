from typing import *


class CementAttributes(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.__dict__
        else:
            raise TypeError(f'{type(item)} is not implemented.')

    def add(self, k, v):
        self.__dict__[k] = v

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CementAttributes':
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
        }
