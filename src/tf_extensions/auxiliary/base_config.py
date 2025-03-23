"""The module contains a base configuration class."""
from dataclasses import asdict, dataclass, fields
from typing import TypeVar, Union

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
            if isinstance(properties, BaseConfig):
                properties_dict = properties.as_dict()
            else:
                properties_dict = {**properties}
            prop = properties_dict.get(cls_field.name)
            if prop is not None:
                default_factory = cls_field.default_factory
                try:
                    kwargs[cls_field.name] = default_factory.from_dict(prop)
                except AttributeError:
                    kwargs[cls_field.name] = prop

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
        return cls.from_dict(kwargs)
