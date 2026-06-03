

def report_deprecated(func):
    print(f' !!!: "{func.__name__}" is deprecated. will be removed in a future release. !!!')

class kwargs_helper:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.allowed_keys = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key in self.kwargs.keys():
            if key not in self.allowed_keys:
                raise ValueError(f' !!!: "{key}" is not a valid argument. !!!')

    def get(self, key, default_value):
        self.allowed_keys.add(key)
        return self.kwargs.get(key, default_value)