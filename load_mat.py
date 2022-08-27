import scipy.io
import numpy as np

dict = scipy.io.loadmat('frame.mat')
data = dict["frame"]
np.set_printoptions(threshold=np.inf)
print(data[0, 0][0].shape)
a = np.zeros(2)
# print(a)