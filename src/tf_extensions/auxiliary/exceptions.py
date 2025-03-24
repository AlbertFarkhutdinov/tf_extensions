"""
Custom exception classes for error handling.

This module defines custom exception classes to provide more informative
error messages in various situations, such as invalid argument values,
unspecified arguments, or incorrect data types.

Classes
-------
EvenKernelError
    Raised when an even-sized kernel is detected.
UnsupportedArgumentError
    Raised when an unsupported argument value is provided.
UnspecifiedArgumentError
    Raised when a required argument is missing.
NonEqualValuesError
    Raised when expected values are not equal.
WrongNumberError
    Raised when a number-related error occurs.
WrongTypeError
    Raised when a type mismatch occurs.

"""


class EvenKernelError(ValueError):
    """Exception raised when an even-sized kernel is found."""

    def __init__(self, kernel_size: tuple[int, ...]) -> None:
        """
        Initialize EvenKernelError.

        Parameters
        ----------
        kernel_size : tuple of int
            The size of the kernel that caused the error.

        """
        super().__init__(f'Even kernel is found: {kernel_size}.')


class UnsupportedArgumentError(ValueError):
    """Exception raised for unsupported argument values."""

    def __init__(self, arg_name: str, arg_value: str) -> None:
        """
        Initialize UnsupportedArgumentError.

        Parameters
        ----------
        arg_name : str
            The name of the argument.
        arg_value : str
            The unsupported value of the argument.

        """
        super().__init__(f'Unsupported value of "{arg_name}": "{arg_value}".')


class UnspecifiedArgumentError(ValueError):
    """Exception raised when a required argument is unspecified."""

    def __init__(self, arg_name: str) -> None:
        """
        Initialize UnspecifiedArgumentError.

        Parameters
        ----------
        arg_name : str
            The name of the missing argument.

        """
        super().__init__(f'Unspecified argument: "{arg_name}".')


class NonEqualValuesError(ValueError):
    """Exception raised when expected values are not equal."""

    def __init__(self, *, msg: str) -> None:
        """
        Initialize NonEqualValuesError.

        Parameters
        ----------
        msg : str
            A message describing the error.

        """
        super().__init__(msg)


class WrongNumberError(ValueError):
    """Exception raised when a number-related error occurs."""

    def __init__(self, *, msg: str) -> None:
        """
        Initialize WrongNumberError.

        Parameters
        ----------
        msg : str
            A message describing the error.

        """
        super().__init__(msg)


class WrongTypeError(TypeError):
    """Exception raised when a type mismatch occurs."""

    def __init__(self, *, msg: str) -> None:
        """
        Initialize WrongTypeError.

        Parameters
        ----------
        msg : str
            A message describing the error.

        """
        super().__init__(msg)
