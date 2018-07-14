from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

binary = np.array([[92, 20, 71],
                   [9, 92, 6],
                   [4, 2, 31]])

fig, ax = plot_confusion_matrix(conf_mat=binary)
plt.show()