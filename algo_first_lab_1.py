import numpy as np
import time
import statistics as stat
import pandas as pd
from typing import NamedTuple
from collections import namedtuple
import matplotlib.pyplot as mp


class Task_one():

    def const(self, vector):
        return 1

    def sum(self, vector):
        sum = 0
        for i in vector:
            sum += i
        return sum

    def product(self, vector):
        
        product = 1
        for i in vector:
            product *= i
            # print(product)
        return product

    def polynomial_naive(self, vector):
        p = 0
        x = 1.5
        for i, a in enumerate(vector):
            p += (x ** i) * a
            # print(f'p = {p}')
        # print(p)
        return p
            # print(f'step {n}')
            # print(f'product_sum {product_sum}')

    def polinom_Horner(self, vector):
        x = 1.5
        p = vector[-1]
        i = len(vector) - 2
        while i >= 0:
            p = p * x + vector[i]
            i -= 1
        # print(f'poli {p}')
        return p

        
    def bubbleSort(self, array):
        for i in range(len(array)):
            for j in range(0, len(array) - i - 1):
                if array[j] > array[j + 1]:
                    temp = array[j]
                    array[j] = array[j+1]
                    array[j+1] = temp

    def quick_sort(self, vector):
        if len(vector) < 2:
            return vector
        else:
            pivot = vector[0]
            less = [i for i in vector[1:] if i <= pivot]
            greater = [i for i in vector[1:] if i > pivot]

            return self.quick_sort(less) + [pivot] + self.quick_sort(greater)

    # Python3 program to perform basic timSort
    MIN_MERGE = 32


    def calcMinRun(self, n):
        """Returns the minimum length of a
        run from 23 - 64 so that
        the len(array)/minrun is less than or
        equal to a power of 2.

        e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33,
        ..., 127=>64, 128=>32, ...
        """
        r = 0
        while n >= self.MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def insertionSort(self, arr, left, right):
        for i in range(left + 1, right + 1):
            j = i
            while j > left and arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                j -= 1

    def merge(self, arr, l, m, r):

        len1, len2 = m - l + 1, r - m
        left, right = [], []
        for i in range(0, len1):
            left.append(arr[l + i])
        for i in range(0, len2):
            right.append(arr[m + 1 + i])

        i, j, k = 0, 0, l

        while i < len1 and j < len2:
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1

            else:
                arr[k] = right[j]
                j += 1

            k += 1

        while i < len1:
            arr[k] = left[i]
            k += 1
            i += 1

        while j < len2:
            arr[k] = right[j]
            k += 1
            j += 1

    def timSort(self, arr):
        n = len(arr)
        minRun = self.calcMinRun(n)

        for start in range(0, n, minRun):
            end = min(start + minRun - 1, n - 1)
            self.insertionSort(arr, start, end)

        size = minRun
        while size < n:

            for left in range(0, n, 2 * size):

                mid = min(n - 1, left + size - 1)
                right = min((left + 2 * size - 1), (n - 1))


                if mid < right:
                    self.merge(arr, left, mid, right)

                size = 2 * size

    def conduct_experiment_const(self):
        Constant = namedtuple('Constant', ['run_number', 'vector_size', 'time'])
        results_list = []
        df = pd.DataFrame
        for i in range (1, 6):  
            for n in range(1, 2001):
                shape = (1, n)
                vector = self.vector = np.random.randint(0, 100, shape)
                vector_size = vector.shape[1]
                start_time = time.time_ns()
                print(start_time)
                algorithm = self.const(vector)
                finish_time = time.time_ns()
                time_result = round((finish_time - start_time), 5)
                
                result = Constant(i, vector_size, time_result)
                results_list.append(result)
                n += 20
            i += 1

        df = pd.DataFrame(results_list, columns=results_list[0]._fields)
        print(list(df.columns.values))
        print(df)

        df_wide = pd.pivot_table(df, index=['vector_size', ], columns=['run_number'])
        print(list(df_wide.columns.values))

        df_wide['mean'] = (df_wide['time', 1]
                           + df_wide['time', 2]
                           + df_wide['time', 3]
                           + df_wide['time', 4]
                           + df_wide['time', 5]) / 5
        df_wide['n'] = df['vector_size']
        print(list(df_wide.columns.values))
        print(df_wide['mean'].max())
        print(df_wide['mean'].min())
        print(df_wide['mean'].mean())
        print(df_wide)
        arr = np.ndarray(2000, 1)
        print(arr)
        vector_one = self.vector = np.random.randint(9268.4, 9268.4, shape)
        np.polyfit(vector_one)
        mp.plot(df_wide['n'], df_wide['mean'])
        mp.plot(df_wide['n'], 1)
        mp.ylim(950, 30000)
        mp.xlabel("Размер вектора V")
        mp.ylabel("Время (наносекунды)")
        mp.show()
        
        
q = Task_one()
res = q.conduct_experiment_const()
