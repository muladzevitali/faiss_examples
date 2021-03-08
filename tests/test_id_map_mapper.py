import unittest
from pathlib import Path
import numpy
from tempfile import TemporaryDirectory
from src.mappers import IndicesMapper


class IDMapMapperClassTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.test_mapper_path: Path = Path(self.temporary_directory.name).joinpath('test.json')
        self.mapper = IndicesMapper(mapper_path=self.test_mapper_path)

    def test_set_get_methods(self):
        vector = numpy.random.random((1, 512))
        self.mapper[0] = vector

        self.assertIs(self.mapper[0], vector)

    def test_save_method(self):
        vector = numpy.random.random((3, 4))

        self.mapper[0] = vector
        self.mapper.save()

        loaded_mapper = IndicesMapper(self.test_mapper_path)

        assert self.test_mapper_path.is_file()
        assert numpy.array_equal(vector, loaded_mapper[0])
