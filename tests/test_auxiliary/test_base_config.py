"""The module provides configs for custom layers."""
from dataclasses import dataclass, field

from tf_extensions.auxiliary.base_config import BaseConfig


@dataclass
class C1(BaseConfig):
    a1: int = 0
    a2: int = 0


@dataclass
class C2(BaseConfig):
    c1: C1 = field(default_factory=C1)
    a3: int = 0
    a4: int = 0


@dataclass
class C3(BaseConfig):
    c2: C2 = field(default_factory=C2)
    a5: int = 0
    a6: int = 0


class TestBaseConfigC1:

    def test_init_default(self) -> None:
        config = C1()
        assert config.a1 == 0
        assert config.a2 == 0

    def test_init_non_default(self) -> None:
        a1 = 1
        config = C1(a1=a1)
        assert config.a1 == a1
        assert config.a2 == 0

    def test_as_dict_default(self) -> None:
        config = C1()
        assert config.as_dict() == {'a1': 0, 'a2': 0}

    def test_as_dict_non_default(self) -> None:
        a1 = 1
        config = C1(a1=a1)
        assert config.as_dict() == {'a1': a1, 'a2': 0}

    def test_from_dict(self) -> None:
        a1 = 1
        config = C1.from_dict(properties={'a1': a1})
        assert config.a1 == a1
        assert config.a2 == 0

    def test_from_kwargs(self) -> None:
        a1 = 1
        config = C1.from_kwargs(a1=a1)
        assert config.a1 == a1
        assert config.a2 == 0


class TestBaseConfigC2:

    def test_init_default(self) -> None:
        config = C2()
        assert config.c1.a1 == 0
        assert config.c1.a2 == 0
        assert config.a3 == 0
        assert config.a4 == 0

    def test_init_non_default(self) -> None:
        a1 = 1
        a3 = 1
        config = C2(
            c1=C1(a1=a1),
            a3=a3,
        )
        assert config.c1.a1 == a1
        assert config.c1.a2 == 0
        assert config.a3 == a3
        assert config.a4 == 0

    def test_as_dict_default(self) -> None:
        config = C2()
        assert config.as_dict() == {
            'c1': {'a1': 0, 'a2': 0},
            'a3': 0,
            'a4': 0,
        }

    def test_as_dict_non_default(self) -> None:
        a1 = 1
        a3 = 1
        config = C2(
            c1=C1(a1=a1),
            a3=a3,
        )
        assert config.as_dict() == {
            'c1': {'a1': a1, 'a2': 0},
            'a3': a3,
            'a4': 0,
        }

    def test_from_dict(self) -> None:
        a1 = 1
        a3 = 1
        config = C2.from_dict(
            properties={
                'c1': {'a1': a1},
                'a3': a3,
            },
        )
        assert config.c1.a1 == a1
        assert config.c1.a2 == 0
        assert config.a3 == a3
        assert config.a4 == 0

    def test_from_kwargs(self) -> None:
        a1 = 1
        a3 = 1
        config = C2.from_kwargs(
            c1={'a1': a1},
            a3=a3,
        )
        assert config.c1.a1 == a1
        assert config.c1.a2 == 0
        assert config.a3 == a3
        assert config.a4 == 0


class TestBaseConfigC3:

    def test_init_default(self) -> None:
        config = C3()
        assert config.c2.c1.a1 == 0
        assert config.c2.c1.a2 == 0
        assert config.c2.a3 == 0
        assert config.c2.a4 == 0
        assert config.a5 == 0
        assert config.a6 == 0

    def test_init_non_default(self) -> None:
        a1 = 1
        a3 = 1
        a5 = 1
        config = C3(
            c2=C2(
                c1=C1(a1=a1),
                a3=a3,
            ),
            a5=a5,
        )
        assert config.c2.c1.a1 == a1
        assert config.c2.c1.a2 == 0
        assert config.c2.a3 == a3
        assert config.c2.a4 == 0
        assert config.a5 == a5
        assert config.a6 == 0

    def test_as_dict_default(self) -> None:
        config = C3()
        assert config.as_dict() == {
            'c2': {
                'c1': {'a1': 0, 'a2': 0},
                'a3': 0,
                'a4': 0,
            },
            'a5': 0,
            'a6': 0,
        }

    def test_as_dict_non_default(self) -> None:
        a1 = 1
        a3 = 1
        a5 = 1
        config = C3(
            c2=C2(
                c1=C1(a1=a1),
                a3=a3,
            ),
            a5=a5,
        )
        assert config.as_dict() == {
            'c2': {
                'c1': {'a1': a1, 'a2': 0},
                'a3': a3,
                'a4': 0,
            },
            'a5': a5,
            'a6': 0,
        }

    def test_from_dict(self) -> None:
        a1 = 1
        a3 = 1
        a5 = 1
        config = C3.from_dict(
            properties={
                'c2': {
                    'c1': {'a1': a1},
                    'a3': a3,
                },
                'a5': a5,
            }
        )
        assert config.c2.c1.a1 == a1
        assert config.c2.c1.a2 == 0
        assert config.c2.a3 == a3
        assert config.c2.a4 == 0
        assert config.a5 == a5
        assert config.a6 == 0

    def test_from_kwargs(self) -> None:
        a1 = 1
        a3 = 1
        a5 = 1
        config = C3.from_kwargs(
            c2={
                'c1': {'a1': a1},
                'a3': a3,
            },
            a5=a5,
        )
        assert config.c2.c1.a1 == a1
        assert config.c2.c1.a2 == 0
        assert config.c2.a3 == a3
        assert config.c2.a4 == 0
        assert config.a5 == a5
        assert config.a6 == 0
