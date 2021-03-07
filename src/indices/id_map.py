"""Faiss index class for creating/loading and using the faiss id map database"""
__author__ = "Vitali Muladze"

import os
from pathlib import Path
from typing import (Union, Tuple, List)

import faiss
import numpy
from faiss import IDSelectorBatch
from .base_index import FaissBaseClass


class FaissIDMapIndex(FaissBaseClass):
    def __init__(self, index_path: Union[str, Path] = None, dimension: int = 2048) -> None:
        """
        Initialize the the faiss database
        """
        super().__init__()
        self.dimension = dimension
        # Check if the file is faiss index
        if os.path.isfile(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.index_factory(dimension, 'IDMap,Flat')

    def insert_many(self, vectors: numpy.array) -> list:
        """
        Insert vectors with batches
        :param vectors: vectors of shape (n, d)
        :return: indices occupied by inserted vectors
        """
        indices: numpy.array = numpy.arange(self.index.ntotal, self.index.ntotal + vectors.shape[0])
        vectors: numpy.array = vectors.astype('float32')
        # Insert values into the index
        self.index.add_with_ids(vectors, indices)

        return indices.tolist()

    def insert_one(self, vector: numpy.array) -> int:
        """
        Insert vector to faiss database
        :param vector: vector of shape (d, )
        :return: index occupied by inserted vector
        """
        index: numpy.array = numpy.arange(self.index.ntotal, self.index.ntotal + 1)
        vectors: numpy.array = vector.reshape(1, -1).astype('float32')
        self.index.add_with_ids(vectors, index)

        return index[0]

    def update_many(self, new_vectors: numpy.array, indices: numpy.array) -> list:
        """
        Update index ids with new values
        :param new_vectors: vectors to change value to
        :param indices: indices to change value on
        :return: new indices of vectors
        """
        indices = indices.astype(numpy.int64)
        new_vectors = new_vectors.astype('float32')

        id_selector = IDSelectorBatch(indices.shape[0], faiss.swig_ptr(indices))
        self.index.remove_ids(id_selector)

        self.index.add_with_ids(new_vectors, indices)

        return indices.tolist()

    def update_one(self, new_vector: numpy.array, index: int) -> int:
        """
        Update index value to new vector
        :param new_vector: vector to change value to
        :param index: index to change value on
        :return: index of the new vector
        """
        index = numpy.array([index]).astype(numpy.int64)
        new_vector = new_vector.reshape(1, -1).astype('float32')

        index_selector = IDSelectorBatch(index.shape[0], faiss.swig_ptr(index))

        self.index.remove_ids(index_selector)

        self.index.add_with_ids(new_vector, index)

        return index[0]

    def find_one(self, vector: numpy.array, n_results: int = 10) -> Tuple[list, list]:
        """
        Search similar vectors to current vector
        :param vector: vectors of shape (d, )
        :param n_results: number of results
        :return: indices of the results and distances sorted increasingly
        """
        vector = vector.reshape(1, -1).astype('float32')
        # Search for similarities
        distances, result_indices = self.index.search(vector, n_results)

        return result_indices.tolist()[0], distances.tolist()[0]

    def find_many(self, vectors: numpy.array, n_results: int = 10) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Search similar vectors to given vectors
        :param vectors: vectors of shape (n, d)
        :param n_results: number of results
        :return: indices of the results and distances sorted increasingly for each vector
        """
        vectors = vectors.astype('float32')
        distances, result_indices = self.index.search(vectors, n_results)

        return result_indices.tolist(), distances.tolist()

    def to_disk(self, index_path: Union[str, Path]) -> str:
        """
        Write the index to disk
        :param index_path: Path to the index folder
        :return: status if writing
        """
        # Create path to the index if not exists
        if not os.path.isdir(os.path.dirname(index_path)):
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
        # Write the index to disk
        faiss.write_index(self.index, index_path)

        return True

    def __len__(self):
        """Get number of vectors in the database"""
        return self.index.ntotal
