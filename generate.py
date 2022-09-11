import numpy as np
import argparse
import pickle
from train import Model

"""
-    Загрузить модель.
-    Инициализировать её каким-нибудь сидом.
-    Сгенерировать последовательность нужной длины.
-    Вывести её на экран.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #parser.add_argument('--model', dest='model', required=True)
    #parser.add_argument('--prefix', dest='prefix', nargs='*')
    #parser.add_argument('--length', dest='length', type=int, required=True)

    #args = parser.parse_args()

    gen = pickle.load(open('model_11-09-2022-19-33-13.pkl', 'rb'))
    l = gen.generate(5, 'by the good housewives')
    print(l['good'])
