import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from src.indices.id_map import FaissIDMapIndex


class TestFaissIDMapDatabaseInsert(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.index_path = Path(self.temporary_directory.name).joinpath('test.index')
        self.dimension = 32
        self.index = FaissIDMapIndex(self.index_path, self.dimension)

    def test_index_creation(self):
        index = FaissIDMapIndex(self.index_path, dimension=self.dimension)
        self.assertIs(len(index), 0)

    def test_insert_one_method(self):
        test_vector = numpy.random.random(self.dimension)
        self.index.insert_one(test_vector)

        self.assertIs(len(self.index), 1)

    def test_insert_many_method(self):
        test_vectors = numpy.random.random((6, self.dimension))

        self.index.insert_many(test_vectors)
        self.assertIs(len(self.index), 6)

    def test_index_save_method(self):
        test_vectors = numpy.random.random((6, self.dimension))

        self.index.insert_many(test_vectors)
        self.index.save()

        assert self.index_path.is_file()




class TestFaissIDMapDatabaseFindAndUpdate(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.index_path = Path(self.temporary_directory.name).joinpath('test.index')
        self.dimension = 32
        self.index = FaissIDMapIndex(self.index_path, self.dimension)
        self.test_vectors = numpy.random.random((100, self.dimension))
        self.index.insert_many(self.test_vectors)

    def test_find_one_method(self):
        indices, distances = self.index.find_one(self.test_vectors[3])
        self.assertIs(indices[0], 3)
        self.assertIs(int(distances[0]), 0)

    def test_find_many_method(self):
        indices, distances = self.index.find_many(self.test_vectors[:4])
        for index_ in range(4):
            self.assertIs(indices[index_][0], index_)
            self.assertIs(int(distances[index_][0]), 0)

    def test_update_one_method(self):
        new_vector = numpy.random.random(self.dimension)
        self.index.update_one(new_vector, 15)

        indices, distances = self.index.find_one(new_vector)

        self.assertIs(indices[0], 15)
        self.assertIs(int(distances[0]), 0)

    def test_update_many_method(self):
        new_vectors = numpy.random.random((5, self.dimension))
        new_indices = numpy.arange(3, 8)
        self.index.update_many(new_vectors, new_indices)

        indices, distances = self.index.find_many(new_vectors)
        for index_ in range(1, 5):
            self.assertIs(indices[index_][0], index_ + 3)
            self.assertIs(int(distances[index_][0]), 0)
