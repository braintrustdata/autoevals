import dataclasses
import json
from typing import Union, get_origin


class SerializableDataClass:
    def as_dict(self):
        """Serialize the object to a dictionary."""
        return dataclasses.asdict(self)

    def as_json(self, **kwargs):
        """Serialize the object to JSON."""
        return json.dumps(self.as_dict(), **kwargs)

    def __getitem__(self, item: str):
        return getattr(self, item)

    @classmethod
    def from_dict(cls, d: dict):
        """Deserialize the object from a dictionary. This method
        is shallow and will not call from_dict() on nested objects."""
        fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in fields}
        return cls(**filtered)

    @classmethod
    def from_dict_deep(cls, d: dict):
        """Deserialize the object from a dictionary. This method
        is deep and will call from_dict_deep() on nested objects."""
        fields = {f.name: f for f in dataclasses.fields(cls)}
        filtered = {}
        for k, v in d.items():
            if k not in fields:
                continue

            if (
                isinstance(v, dict)
                and isinstance(fields[k].type, type)
                and issubclass(fields[k].type, SerializableDataClass)
            ):
                filtered[k] = fields[k].type.from_dict_deep(v)
            elif get_origin(fields[k].type) == Union:
                for t in fields[k].type.__args__:
                    if t == type(None) and v is None:
                        filtered[k] = None
                        break
                    if isinstance(t, type) and issubclass(t, SerializableDataClass) and v is not None:
                        try:
                            filtered[k] = t.from_dict_deep(v)
                            break
                        except TypeError:
                            pass
                else:
                    filtered[k] = v
            elif (
                isinstance(v, list)
                and get_origin(fields[k].type) == list
                and len(fields[k].type.__args__) == 1
                and isinstance(fields[k].type.__args__[0], type)
                and issubclass(fields[k].type.__args__[0], SerializableDataClass)
            ):
                filtered[k] = [fields[k].type.__args__[0].from_dict_deep(i) for i in v]
            else:
                filtered[k] = v
        return cls(**filtered)
