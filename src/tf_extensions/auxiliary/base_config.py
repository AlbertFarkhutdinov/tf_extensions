"""The module contains a base configuration class."""
from contextlib import suppress
from dataclasses import Field, asdict, dataclass, fields
from inspect import signature
from typing import Optional, TypeVar, Union

BaseConfigType = TypeVar('BaseConfigType', bound='BaseConfig')

JSONField = Union[
    bool,
    int,
    float,
    str,
    list['JSONField'],
    dict[str, 'JSONField'],
    None,
]


@dataclass
class BaseConfig:
    """
    A base configuration class.

    This class provides methods for converting between dictionary
    representations and class instances, and for handling keyword arguments.

    Methods
    -------
    as_dict() -> dict
        Converts the instance of the class to a dictionary.

    from_dict(cls, properties: dict) -> BaseConfigType
        Creates an instance of the class from a dictionary.

    from_kwargs(cls, ``**kwargs``) -> BaseConfigType
        Creates an instance of the class from keyword arguments.

    """

    def as_dict(self) -> dict[str, JSONField]:
        """
        Convert the instance of the class to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the class instance.

        """
        # noinspection PyTypeChecker
        return asdict(self)

    @classmethod
    def from_dict(cls, properties: dict[str, JSONField]) -> BaseConfigType:
        """
        Create an instance of the class from a dictionary.

        Parameters
        ----------
        properties : dict
            A dictionary containing the properties to set on the instance.

        Returns
        -------
        BaseConfigType
            An instance of the class with the properties from the dictionary.

        """
        kwargs = {}
        # noinspection PyTypeChecker
        for cls_field in fields(cls):
            if cls_field.name in properties:
                kwargs[cls_field.name] = cls._get_config_property(
                    cls_field=cls_field,
                    properties=properties,
                )
        # noinspection PyArgumentList
        return cls(**kwargs)

    @classmethod
    def from_kwargs(cls, **kwargs) -> BaseConfigType:
        """
        Create an instance of the class from keyword arguments.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the class constructor.

        Returns
        -------
        BaseConfigType
            An instance of the class with the keyword arguments.

        """
        native_args, new_args = {}, {}
        for name, kwarg in kwargs.items():
            if name in signature(cls).parameters:
                native_args[name] = kwarg
            else:
                new_args[name] = kwarg
        # noinspection PyTypeChecker
        for cls_field in fields(cls):
            with suppress(AttributeError):
                native_args[cls_field.name] = (
                    cls_field.default_factory.from_kwargs(**new_args)
                )
        # noinspection PyArgumentList
        return cls(**native_args)

    @classmethod
    def _get_config_property(
        cls,
        cls_field: Field,
        properties: dict[str, JSONField],
    ) -> Optional[Union[BaseConfigType, JSONField]]:
        """
        Return a proper config field from a dictionary.

        Parameters
        ----------
        cls_field : Field
            A field of the config.
        properties : dict
            A dictionary containing the properties to set on the instance.

        Returns
        -------
        any
            A proper config field from a dictionary.

        Raises
        ------
        ValueError
            If a default factory of a `BaseConfigType` attribute is undefined.

        """
        if 'config' in cls_field.name:
            default_factory = cls_field.default_factory
            try:
                return default_factory.from_dict(
                    properties=properties[cls_field.name],
                )
            except AttributeError as exc:
                msg = 'Default factory for `{field}` is undefined.'.format(
                    field=cls_field.name,
                )
                raise ValueError(msg) from exc
        else:
            return properties[cls_field.name]
