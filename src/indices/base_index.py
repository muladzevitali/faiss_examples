from pathlib import Path
from typing import (Union, Tuple, List)

import numpy


class FaissBaseClass:
    def insert_one(self, vector: numpy.array) -> int:
        """
        Insert vector to faiss database
        :param vector: vector of shape (d, )
        :return: index occupied by inserted vector
        """

        raise NotImplementedError

    def insert_many(self, vectors: numpy.array) -> list:
        """
        Insert vectors with batches
        :param vectors: vectors of shape (n, d)
        :return: indices occupied by inserted vectors
        """
        raise NotImplementedError

    def update_many(self, new_vectors: numpy.array, indices: numpy.array) -> list:
        """
        Update index ids with new values
        :param new_vectors: vectors to change value to
        :param indices: indices to change value on
        :return: new indices of vectors
        """
        raise NotImplementedError

    def update_one(self, new_vector: numpy.array, index: int) -> int:
        """
        Update index value to new vector
        :param new_vector: vector to change value to
        :param index: index to change value on
        :return: index of the new vector
        """
        raise NotImplementedError

    def find_one(self, vector: numpy.array, n_results: int = 10) -> Tuple[list, list]:
        """
        Search similar vectors to current vector
        :param vector: vectors of shape (d, )
        :param n_results: number of results
        :return: indices of the results and distances sorted increasingly
        """
        raise NotImplementedError

    def find_many(self, vectors: numpy.array, n_results: int = 10) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Search similar vectors to given vectors
        :param vectors: vectors of shape (n, d)
        :param n_results: number of results
        :return: indices of the results and distances sorted increasingly for each vector
        """

        raise NotImplementedError

    def to_disk(self, index_path: Union[str, Path]) -> str:
        """
        Write the index to disk
        :param index_path: Path to the index folder
        :return: status if writing
        """
        raise NotImplementedError

    def __len__(self):
        """Get number of vectors in the database"""
        raise NotImplementedError
