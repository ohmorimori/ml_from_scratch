import sys
dir_str = ".."
if (dir_str not in sys.path):
    sys.path.append(dir_str)
import numpy as np
from utils.data_manipulation import sigmoid

class Sigmoid():
    def __init__(self):
        self.out
        pass

    def forward(self, x):
        self.out = sigmoid(x)
        return

    def backward(self, dout):
        return dout * (self.out * (1 - self.out))



