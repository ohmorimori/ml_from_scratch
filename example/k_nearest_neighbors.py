import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

dir_str = ".."
if dir_str not in sys.path:
	sys.path.append(dir_str)

from utils import train_test_split, normalize, accuracy_score, euclidean_distance, Plot
from supervised_learning import Knn

