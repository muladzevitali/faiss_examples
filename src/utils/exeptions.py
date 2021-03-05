class LengthError(Exception):
    """
    Exception on iterable length error
    """

    def __init__(self, message: str = 'Incorrect length') -> None:
        self.message = message


class DimensionError(Exception):
    """
    Exception on incorrect array dimension
    """
    def __init__(self, message: str = 'Incorrect length') -> None:
        self.message = message
