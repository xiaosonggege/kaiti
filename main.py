from collections.abc import Callable, Iterable, Iterator, Generator
import main2
class Ade:
    def __init__(self, name:str):
        self._name = name
    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

class A:
    def __init__(self):
        self._l = [1, 2, 3, 4]
        self._item = 0
    def __iter__(self):
        return self
    def __next__(self):
        l = [1, 2, 3, 4]
        for i in l:
            yield(i)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
    def fun(self, a:str, b:int)->int:
        return 3
class B(Exception):
    name = 'yun'
    def __new__(cls, ann:str):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        elif ann != cls.instance._ann:
            cls.instance = super().__new__(cls)
        return cls.instance
    def __init__(self, ann:str):
        self._ann = ann
        # print(locals())
    def __str__(self):
        return self._ann
    def __del__(self):
        print('woaibingqian')

if __name__ == '__main__':
    import sys
    import io
    a = io.StringIO()
    a.write('song')
    a.seek(2)
    a.write('yun')
    print(a.getvalue())
