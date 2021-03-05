import faiss
import numpy

d = 64
nb = 100
xb = numpy.random.random((nb, d)).astype('float32')
print(xb.shape)
indices = numpy.arange(5, nb +5).astype(numpy.int64)
search_vector = xb[1:5]

index = faiss.index_factory(d, 'IDMap,Flat')

a = index.add_with_ids(xb, indices)
print('inserted: ', index.ntotal)
_, i = index.search(search_vector, 1)

print('fount at ', i[0], 'index')

indices_two_remove = numpy.array([1, 2]).astype(numpy.int64)
id_selector = faiss.IDSelectorBatch(indices_two_remove.shape[0], faiss.swig_ptr(indices_two_remove))
index.remove_ids(id_selector)
d, i = index.search(search_vector, 10)
print(d.tolist(), i.tolist())
print('fount at ', i[0], 'index')