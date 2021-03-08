import json
from pathlib import Path
from typing import (Union, Any)

from src.utils.numpy_json_encoder import (NumpyArrayDecoder, NumpyArrayEncoder)


class IndicesMapper:
    def __init__(self, mapper_path: Union[str, Path]):
        self.__mapper: dict = dict()
        self.mapper_path: Path = Path(mapper_path)

        if self.mapper_path.is_file():
            self._load_mapper_file(mapper_path)

    def _load_mapper_file(self, file_path: Union[str, Path]) -> None:
        """
        Load numpy array containing json file with custom decoder handler
        :param file_path: json file path
        """
        with open(str(file_path)) as input_stream:
            self.__mapper = json.load(input_stream, cls=NumpyArrayDecoder)

    def save(self) -> bool:
        """
        Write mapper to disk with numpy array serializer
        :return: status of writing
        """
        with open(str(self.mapper_path), 'w') as output_stream:
            json.dump(self.__mapper, output_stream, cls=NumpyArrayEncoder)

        return True

    def __getitem__(self, item: Any):
        # keys are always strings
        item = str(item)
        return self.__mapper[item]

    def __setitem__(self, key, value):
        # keys are always string
        key = str(key)
        self.__mapper[key] = value
