class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setattr__
    __delattr__ = dict.__delitem__



class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]
