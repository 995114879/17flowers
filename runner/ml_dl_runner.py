import os
import sys

print(sys.path)

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


def t0():
    from ml_dl import iris_demo
    # iris_demo.training_v0()
    iris_demo.training_v1()

if __name__ == '__main__':
    t0()
