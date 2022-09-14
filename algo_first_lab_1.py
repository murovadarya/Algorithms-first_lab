from multiprocessing.dummy import Array
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
            p += (x ** i) * (a - 1)
            # print(f'p = {p}')
        # print(p)
        return p

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

            return self.quick_sort(less
                                   + [pivot]
                                   + self.quick_sort(greater))

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

    def delete_noize(self, data):

        for x in ['mean']:
            q75, q25 = np.percentile(data.loc[:, x], [75, 25])
            intr_qr = q75-q25

            max = q75+(1.5*intr_qr)
            min = q25-(1.5*intr_qr)

            data.loc[data[x] < min, x] = np.nan
            data.loc[data[x] > max, x] = np.nan

        data.isnull().sum()
        data = data.dropna(axis=0)
        # print(data.isnull().sum())
        return data

    def conduct_experiment_const(self):
        Constant = namedtuple('Constant', ['run_number', 'vector_size', 'time'])
        results_list = []
        df = pd.DataFrame

        # запускаем 5 раз алгоритм
        for i in range(1, 6):

            # generate vector of N scale
            for n in range(1, 2001):
                shape = (1, n)
                vector = self.vector = np.random.randint(0, 100, shape)
                vector_size = vector.shape[1]

                # start algorithms with this vector, measure time
                start_time = time.time_ns()
                print(start_time)
                algorithm = self.const(vector)
                finish_time = time.time_ns()
                time_result = (finish_time - start_time)

                # write down the results of experiment
                result = Constant(i, vector_size, time_result)
                results_list.append(result)
                n += 1
            i += 1

        df = pd.DataFrame(results_list, columns=results_list[0]._fields)
        # print(list(df.columns.values))
        print(df)

        # reshare dataframe
        df_wide = pd.pivot_table(df, index=['vector_size', ],
                                 columns=['run_number'])
        print(list(df_wide.columns.values))

        # find mean time of 5 runs
        df_wide['mean'] = (df_wide['time', 1]
                           + df_wide['time', 2]
                           + df_wide['time', 3]
                           + df_wide['time', 4]
                           + df_wide['time', 5]) / 5
        df_wide['n'] = df['vector_size']

        # clean outliers
        cleaned_data = self.delete_noize(df_wide)

        # check mean and median
        mean = cleaned_data['mean'].mean()
        median = cleaned_data['mean'].median()
        print(f'Mean: {mean}')
        print(f'Median: {median}')

        # plot the results

        mp.plot(cleaned_data['n'], cleaned_data['mean'],
                label='Эксперементальные результаты')
        mp.hlines(y=mean, xmin=1, xmax=2000, colors='red',
                  label='Аппроксимация на основе теоретических оценок')
        mp.ylim(950, 5000)
        mp.xlabel("Размер вектора V")
        mp.ylabel("Время (наносекунды)")
        mp.legend()
        mp.show()

    def conduct_experiment_sum(self):
        Constant = namedtuple('Sum', ['run_number', 'vector_size', 'time'])
        results_list = []
        df = pd.DataFrame

        # запускаем 5 раз алгоритм
        for i in range(1, 6):
            # generate vector of N scale
            for n in range(1, 2001):
                shape = (1, n)
                vector = self.vector = np.random.randint(0, 100, shape)
                vector_size = vector.shape[1]

                # start algorithms with this vector, measure time
                start_time = time.time_ns()
                print(start_time)
                algorithm = self.sum(vector)
                finish_time = time.time_ns()
                time_result = (finish_time - start_time)

                # write down the results of experiment
                result = Constant(i, vector_size, time_result)
                results_list.append(result)
                n += 1
            i += 1

        df = pd.DataFrame(results_list, columns=results_list[0]._fields)
        # print(list(df.columns.values))
        print(df)

        # reshare dataframe
        df_wide = pd.pivot_table(df, index=['vector_size', ],
                                 columns=['run_number'])
        print(list(df_wide.columns.values))

        # find mean time of 5 runs
        df_wide['mean'] = (df_wide['time', 1]
                           + df_wide['time', 2]
                           + df_wide['time', 3]
                           + df_wide['time', 4]
                           + df_wide['time', 5]) / 5
        df_wide['n'] = df['vector_size']

        # clean outliers
        cleaned_data = self.delete_noize(df_wide)

        # check mean and median 
        mean = cleaned_data['mean'].mean()
        median = cleaned_data['mean'].median()
        print(f'Mean: {mean}')
        print(f'Median: {median}')
        cleaned_data.drop(
            cleaned_data.tail(1).index, inplace=True)
        print(cleaned_data)
        # m, b = np.polyfit(cleaned_data['mean'], cleaned_data['time'], 1)
        # mp.plot(cleaned_data['mean'], m*cleaned_data['mean'] + b)
        # plot the results

        y = 0.5 * cleaned_data['n'] + cleaned_data.iloc[0, 5]

        mp.plot(cleaned_data['n'], cleaned_data['mean'],
                label='Эксперементальные результаты')
        mp.plot(cleaned_data['n'], y, label='Теоретические результаты')
        mp.xlabel("Размер вектора V")
        mp.ylabel("Время (наносекунды)")
        mp.legend()
        mp.show()

    def conduct_experiment_product(self):
        Constant = namedtuple('Product', ['run_number', 'vector_size', 'time'])
        results_list = []
        df = pd.DataFrame

        # запускаем 5 раз алгоритм
        for i in range(1, 6):

            # generate vector of N scale
            for n in range(1, 2001):
                shape = (1, n)
                vector = self.vector = np.random.randint(0, 100, shape)
                vector_size = vector.shape[1]

                # start algorithms with this vector, measure time
                start_time = time.time_ns()
                print(start_time)
                algorithm = self.product(vector)
                finish_time = time.time_ns()
                time_result = (finish_time - start_time)

                # write down the results of experiment
                result = Constant(i, vector_size, time_result)
                results_list.append(result)
                n += 1
            i += 1

        df = pd.DataFrame(results_list, columns=results_list[0]._fields)
        # print(list(df.columns.values))
        print(df)

        # reshare dataframe
        df_wide = pd.pivot_table(df, index=['vector_size', ],
                                 columns=['run_number'])
        print(list(df_wide.columns.values))

        # find mean time of 5 runs
        df_wide['mean'] = (df_wide['time', 1]
                           + df_wide['time', 2]
                           + df_wide['time', 3]
                           + df_wide['time', 4]
                           + df_wide['time', 5]) / 5
        df_wide['n'] = df['vector_size']

        # clean outliers
        cleaned_data = self.delete_noize(df_wide)
        mean = cleaned_data['mean'].mean()
        median = cleaned_data['mean'].median()

        print(f'Mean: {mean}')
        print(f'Median: {median}')
        cleaned_data.drop(cleaned_data.tail(1).index, inplace=True)
        print(cleaned_data)

        y = 0.5 * cleaned_data['n'] + cleaned_data.iloc[2, 5]

        mp.plot(cleaned_data['n'], cleaned_data['mean'],
                label='Эксперементальные результаты')
        mp.plot(cleaned_data['n'], y, label='Теоретические результаты')
        mp.xlabel("Размер вектора V")
        mp.ylabel("Время (наносекунды)")
        mp.legend()
        mp.show()

    def conduct_experiment_polynomial_naive(self):
        Constant = namedtuple('polynomial_naive', ['run_number', 'vector_size', 'time'])
        results_list = []
        df = pd.DataFrame

        # запускаем 5 раз алгоритм
        for i in range(1, 6):

            # generate vector of N scale
            for n in range(1, 2001):
                shape = (1, n)
                vector = self.vector = np.random.randint(0, 100, shape)
                vector_size = vector.shape[1]

                # start algorithms with this vector, measure time
                start_time = time.time_ns()
                print(start_time)
                algorithm = self.polynomial_naive(vector)
                finish_time = time.time_ns()
                time_result = (finish_time - start_time)

                # write down the results of experiment
                result = Constant(i, vector_size, time_result)
                results_list.append(result)
                n += 1
            i += 1

        df = pd.DataFrame(results_list, columns=results_list[0]._fields)
        # print(list(df.columns.values))
        print(df)

        # reshare dataframe
        df_wide = pd.pivot_table(df, index=['vector_size', ],
                                 columns=['run_number'])
        print(list(df_wide.columns.values))

        # find mean time of 5 runs
        df_wide['mean'] = (df_wide['time', 1]
                           + df_wide['time', 2]
                           + df_wide['time', 3]
                           + df_wide['time', 4]
                           + df_wide['time', 5]) / 5
        df_wide['n'] = df['vector_size']

        # clean outliers
        cleaned_data = self.delete_noize(df_wide)
        mean = cleaned_data['mean'].mean()
        median = cleaned_data['mean'].median()

        print(f'Mean: {mean}')
        print(f'Median: {median}')
        cleaned_data.drop(cleaned_data.tail(1).index, inplace=True)
        print(cleaned_data)

        # fit
        print(cleaned_data['n'].shape, cleaned_data['mean'].shape)
        fit = np.polyfit(cleaned_data['n'], cleaned_data['mean'], 2)
        equation = np.poly1d(fit)
        print("The fit coefficients are a = {0:.4f}, b = {1:.4f} c = {2:.4f}".format(*fit))
        print(equation)
        print(fit)

        # theoretical curve 
        y = fit[0] * cleaned_data['n'] ** 2 + fit[1] * cleaned_data['n'] + fit[2]

        mp.plot(cleaned_data['n'], cleaned_data['mean'],
                label='Эксперементальные результаты')
        mp.plot(cleaned_data['n'], y, label='Теоретические результаты')
        # mp.ylim(3000, 10000)
        mp.xlabel("Размер вектора V")
        mp.ylabel("Время (наносекунды)")
        mp.legend()
        mp.show()

    def conduct_experiment_polinom_Horner(self):
        Constant = namedtuple('polinom_Horner',
                              ['run_number', 'vector_size', 'time'])
        results_list = []
        df = pd.DataFrame

        # запускаем 5 раз алгоритм
        for i in range(1, 6):

            # generate vector of N scale
            for n in range(1, 2001):
                shape = (1, n)
                vector = self.vector = np.random.randint(0, 100, shape)
                vector_size = vector.shape[1]

                # start algorithms with this vector, measure time
                start_time = time.time_ns()
                print(start_time)
                algorithm = self.polinom_Horner(vector)
                finish_time = time.time_ns()
                time_result = (finish_time - start_time)

                # write down the results of experiment
                result = Constant(i, vector_size, time_result)
                results_list.append(result)
                n += 1
            i += 1

        df = pd.DataFrame(results_list, columns=results_list[0]._fields)
        # print(list(df.columns.values))
        print(df)

        # reshare dataframe
        df_wide = pd.pivot_table(df, index=['vector_size', ],
                                 columns=['run_number'])
        print(list(df_wide.columns.values))

        # find mean time of 5 runs
        df_wide['mean'] = (df_wide['time', 1]
                           + df_wide['time', 2]
                           + df_wide['time', 3]
                           + df_wide['time', 4]
                           + df_wide['time', 5]) / 5
        df_wide['n'] = df['vector_size']

        # clean outliers
        cleaned_data = self.delete_noize(df_wide)
        mean = cleaned_data['mean'].mean()
        median = cleaned_data['mean'].median()

        print(f'Mean: {mean}')
        print(f'Median: {median}')
        cleaned_data.drop(cleaned_data.tail(1).index,inplace=True)
        print(cleaned_data)

        y = 0.5 * cleaned_data['n'] + cleaned_data.iloc[1, 5]

        mp.plot(cleaned_data['n'], cleaned_data['mean'],
                label='Эксперементальные результаты')
        mp.plot(cleaned_data['n'], y, label='Теоретические результаты')
        mp.xlabel("Размер вектора V")
        mp.ylabel("Время (наносекунды)")
        mp.legend()
        mp.show()

    def conduct_experiment_bubbleSort(self):
        Constant = namedtuple('bubbleSort',
                              ['run_number', 'vector_size', 'time'])
        results_list = []
        df = pd.DataFrame

        # запускаем 5 раз алгоритм
        for i in range(1, 6):

            # generate vector of N scale
            for n in range(1, 2001):
                shape = (1, n)
                vector = self.vector = np.random.randint(0, 100, shape)
                vector_size = vector.shape[1]

                # start algorithms with this vector, measure time
                start_time = time.time_ns()
                print(start_time)
                algorithm = self.bubbleSort(vector)
                finish_time = time.time_ns()
                time_result = (finish_time - start_time)

                # write down the results of experiment
                result = Constant(i, vector_size, time_result)
                results_list.append(result)
                n += 1
            i += 1

        df = pd.DataFrame(results_list, columns=results_list[0]._fields)
        # print(list(df.columns.values))
        print(df)

        # reshare dataframe
        df_wide = pd.pivot_table(df,
                                 index=['vector_size', ],
                                 columns=['run_number'])
        print(list(df_wide.columns.values))

        # find mean time of 5 runs 
        df_wide['mean'] = (df_wide['time', 1]
                           + df_wide['time', 2]
                           + df_wide['time', 3]
                           + df_wide['time', 4]
                           + df_wide['time', 5]) / 5
        df_wide['n'] = df['vector_size']

        # clean outliers 
        cleaned_data = self.delete_noize(df_wide)
        mean = cleaned_data['mean'].mean()
        median = cleaned_data['mean'].median()

        print(f'Mean: {mean}')
        print(f'Median: {median}')
        cleaned_data.drop(cleaned_data.tail(1).index, inplace=True)
        print(cleaned_data)

        # fit
        print(cleaned_data['n'].shape, cleaned_data['mean'].shape)
        fit = np.polyfit(cleaned_data['n'], cleaned_data['mean'], 2)
        equation = np.poly1d(fit)
        print("The fit coefficients are a = {0:.4f}, b = {1:.4f} c = {2:.4f}".format(*fit))
        print(equation)
        print(fit)

        # theoretical curve
        y = fit[0] * cleaned_data['n'] ** 2 + fit[1] * cleaned_data['n'] + fit[2]

        mp.plot(cleaned_data['n'], cleaned_data['mean'],
                label='Эксперементальные результаты')
        mp.plot(cleaned_data['n'], y, label='Теоретические результаты')
        mp.ylim(1000, 5000)
        mp.xlabel("Размер вектора V")
        mp.ylabel("Время (наносекунды)")
        mp.legend()
        mp.show()

    def conduct_experiment_quick_sort(self):
        Constant = namedtuple(
            'quick_sort', ['run_number', 'vector_size', 'time'])
        results_list = []
        df = pd.DataFrame

        # запускаем 5 раз алгоритм
        for i in range(1, 6):
            # generate vector of N scale
            for n in range(1, 2001):
                shape = (1, n)
                vector = self.vector = np.random.randint(0, 100, shape)
                vector_size = vector.shape[1]

                # start algorithms with this vector, measure time
                start_time = time.time_ns()
                print(start_time)
                algorithm = self.quick_sort(vector)
                finish_time = time.time_ns()
                time_result = (finish_time - start_time)

                # write down the results of experiment
                result = Constant(i, vector_size, time_result)
                results_list.append(result)
                n += 1
            i += 1

        df = pd.DataFrame(results_list, columns=results_list[0]._fields)
        # print(list(df.columns.values))
        print(df)

        # reshare dataframe
        df_wide = pd.pivot_table(df,
                                 index=['vector_size', ],
                                 columns=['run_number'])
        print(list(df_wide.columns.values))

        # find mean time of 5 runs
        df_wide['mean'] = (df_wide['time', 1]
                           + df_wide['time', 2]
                           + df_wide['time', 3]
                           + df_wide['time', 4]
                           + df_wide['time', 5]) / 5
        df_wide['n'] = df['vector_size']

        # clean outliers 
        cleaned_data = self.delete_noize(df_wide)
        mean = cleaned_data['mean'].mean()
        median = cleaned_data['mean'].median()

        print(f'Mean: {mean}')
        print(f'Median: {median}')
        cleaned_data.drop(cleaned_data.tail(1).index, inplace=True)
        # print(cleaned_data)
        coeffs = np.polyfit(np.log(cleaned_data['n']), cleaned_data['mean'], 1)
        fit = np.poly1d(coeffs)
        print(fit)
        equation = np.poly1d(fit)
        print(equation)

        mp.plot(cleaned_data['n'],
                cleaned_data['mean'],
                label='Эксперементальные результаты')
        mp.plot(cleaned_data['n'], fit(np.log(cleaned_data['n'])),
                label='Теоретические результаты')
        mp.ylim(1000, 5000)
        mp.xlabel("Размер вектора V")
        mp.ylabel("Время (наносекунды)")
        mp.legend()
        mp.show()

    def conduct_experiment_timSort(self):
        Constant = namedtuple('timSort', ['run_number', 'vector_size', 'time'])
        results_list = []
        df = pd.DataFrame

        # запускаем 5 раз алгоритм
        for i in range(1, 6):

            # generate vector of N scale
            for n in range(1, 2001):
                shape = (1, n)
                vector = self.vector = np.random.randint(0, 100, shape)
                vector_size = vector.shape[1]

                # start algorithms with this vector, measure time
                start_time = time.time_ns()
                print(start_time)
                algorithm = self.timSort(vector)
                finish_time = time.time_ns()
                time_result = (finish_time - start_time)

                # write down the results of experiment
                result = Constant(i, vector_size, time_result)
                results_list.append(result)
                n += 1
            i += 1

        df = pd.DataFrame(results_list, columns=results_list[0]._fields)
        # print(list(df.columns.values))
        print(df)

        # reshare dataframe
        df_wide = pd.pivot_table(df, index=['vector_size', ],
                                 columns=['run_number'])
        print(list(df_wide.columns.values))

        # find mean time of 5 runs
        df_wide['mean'] = (df_wide['time', 1]
                           + df_wide['time', 2]
                           + df_wide['time', 3]
                           + df_wide['time', 4]
                           + df_wide['time', 5]) / 5
        df_wide['n'] = df['vector_size']

        # clean outliers 
        cleaned_data = self.delete_noize(df_wide)
        mean = cleaned_data['mean'].mean()
        median = cleaned_data['mean'].median()

        print(f'Mean: {mean}')
        print(f'Median: {median}')
        cleaned_data.drop(cleaned_data.tail(1).index, inplace=True)
        # print(cleaned_data)

        coeffs = np.polyfit(np.log(cleaned_data['n']), cleaned_data['mean'], 1)
        fit = np.poly1d(coeffs)
        equation = np.poly1d(fit)
        print(equation)

        mp.plot(cleaned_data['n'], cleaned_data['mean'],
                label='Эксперементальные результаты')
        mp.plot(cleaned_data['n'], fit(np.log(cleaned_data['n'])),
                label='Теоретические результаты')
        mp.ylim(1000, 5000)
        mp.xlabel("Размер вектора V")
        mp.ylabel("Время (наносекунды)")
        mp.legend()
        mp.show()


q = Task_one()

# const = q.conduct_experiment_const()
# sum = q.conduct_experiment_sum()
# product = q.conduct_experiment_product()
# polynomial_naive = q.conduct_experiment_polynomial_naive()
# polinom_Horner = q.conduct_experiment_polinom_Horner()
# bubbleSort = q.conduct_experiment_bubbleSort()
# quick_sort = q.conduct_experiment_quick_sort()
# timSort = q.conduct_experiment_timSort()

