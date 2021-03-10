from collections.abc import Callable, Iterable, Iterator, Generator
import main2
import contextlib
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':
    os.system('export DISPLAY=:0.0')
    fig, ax = plt.subplots()
    fig.show()
    a = [1, 2, 3]
    a.append()