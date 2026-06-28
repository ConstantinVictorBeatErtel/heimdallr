from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class FloatList(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, value: _Iterable[float] | None = ...) -> None: ...

class Feature(_message.Message):
    __slots__ = ("float_list",)
    FLOAT_LIST_FIELD_NUMBER: _ClassVar[int]
    float_list: FloatList
    def __init__(self, float_list: FloatList | _Mapping | None = ...) -> None: ...

class Features(_message.Message):
    __slots__ = ("feature",)

    class FeatureEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Feature
        def __init__(
            self, key: str | None = ..., value: Feature | _Mapping | None = ...
        ) -> None: ...

    FEATURE_FIELD_NUMBER: _ClassVar[int]
    feature: _containers.MessageMap[str, Feature]
    def __init__(self, feature: _Mapping[str, Feature] | None = ...) -> None: ...

class Example(_message.Message):
    __slots__ = ("features",)
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    features: Features
    def __init__(self, features: Features | _Mapping | None = ...) -> None: ...
