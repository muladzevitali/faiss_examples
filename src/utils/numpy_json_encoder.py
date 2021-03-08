import json

import numpy


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return dict(__type__='numpy_array', __value__=obj.tolist())
        if isinstance(obj, numpy.integer):
            return int(obj)

        elif isinstance(obj, numpy.floating):
            return float(obj)

        return super().default(obj)


class NumpyArrayDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(obj):
        if obj.get('__type__') == 'numpy_array':
            return numpy.array(obj['__value__'])

        return obj
