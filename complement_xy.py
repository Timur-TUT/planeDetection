import numpy as np

data = np.array([[1394, 1397, 1397, 1400, 1404, 1404, 1407, 1407, 1407, 1407],
       [1391, 1394, 1394, 1397, 1397, 1400, 1400, 1404, 1404, 1404],
       [1388, 1391, 1391, 1394, 1397, 1397, 1397, 1400, 1400, 1400],
       [1388, 1391, 1391, 1391, 1394, 1394, 1394, 1397, 1397, 1397],
       [1385, 1388, 1391, 1391, 1391, 1394, 1394, 1394, 1394, 1394],
       [1385, 1388, 1388, 1388, 1388, 1391, 1391, 1391, 1391, 1391],
       [1385, 1388, 1388, 1388, 1391, 1391, 1391, 1391, 1391, 1388],
       [1385, 1388, 1388, 1388, 1391, 1391, 1391, 1391, 1388, 1388],
       [1385, 1385, 1388, 1388, 1388, 1388, 1388, 1388, 1388, 1385],
       [1385, 1388, 1388, 1388, 1388, 1388, 1388, 1388, 1388, 1385]])
data = data[..., np.newaxis]

indices = []
for i in range(10):
    for j in range(10):
        indices.append([j, i])

indices = np.reshape(indices, (10, 10, 2))

indices = np.concatenate([indices, data], axis=2)

print(indices)