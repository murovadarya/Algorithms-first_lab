import numpy as np
import time

n = 200

X = np.random.rand(n, n)
Y = np.random.rand(n, n)

result = np.empty((n, n))


start_time = time.time()
for i in range(len(X)):
    # iterate through columns of Y
    for j in range(len(Y[0])):
        # iterate through rows of Y
        for k in range(len(Y)):
            result[i][j] += X[i][k] * Y[k][j]
finish_time = time.time()
result_time = finish_time - start_time

print(f'Time for finding product: {result_time}')
