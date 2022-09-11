from unittest import result
import numpy as np
import re
import argparse
import sys
import os
import pickle
from datetime import datetime
"""
-    Считать входные данные из файлов.
+    Очистить тексты: выкидывать неалфавитные символы, приводить к lowercase.
+    Разбить тексты на слова (в ML это называется токенизацией).
+    Сохранить модель в каком-нибудь формате, который позволяет восстановить её в утилите генерации.
"""


class Model:
    def __init__(self, n=2):
        self.n = n

        # Список слов, которые могут появиться после заданного контекста
        self.context = {}

        # Сколько раз ngram появлялся в тексте
        self.ngram_counter = {}

    #Токеназацию делаем по предложениям (не самое лучшее решение)
    def __tokenize(self, text: str) -> list:
        """
        :param text: Получаем весь текст
        :return: Токенизированные предложения
        """
        text = re.sub(r'[^a-zA-z0-9\s]', ' ', text).lower().strip()
        token = []
        for sentence in text.split('.'):
            token += sentence.split()
        return token

    def __get_ngrams(self, n: int, tokens: list) -> list:
        """
        :param n: n-грамм
        :param tokens: Токенизированные предложения
        :return: лист с n-грамми
        """

        tokens = np.append((n-1)*['<>'], tokens)
        l = [(tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i])
             for i in range(n-1, len(tokens))]
        return l

    def fit(self, text: str) -> "Model":
        """
        Обновляем языковую модель для генерации
        :param text: Весь текст (Все тексты)
        :return: self
        """

        n = self.n
        ngrams = self.__get_ngrams(n, self.__tokenize(text))
        for ngram in ngrams:
            if ngram in self.ngram_counter:
                self.ngram_counter[ngram] += 1
            else:
                self.ngram_counter[ngram] = 1

            prev_words, target_word = ngram
            if prev_words in self.context:
                self.context[prev_words].append(target_word)
            else:
                self.context[prev_words] = [target_word]

        return self

    def generate(self, token_count:int, text:str = '') -> str:
        n = self.n
        context_queue = (n - 1) * ['<>']
        result = []
        tokens_input_text = self.__get_ngrams(n, self.__tokenize(text))

        token_count -= len(tokens_input_text)
        for _ in range(token_count):
            obj = np.random.choice

        return tokens_input_text

    def print_parameters(self):
        #print(self.context)
        print(self.ngram_counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    #parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--input-dir', dest='input')
    parser.add_argument('--model', dest='model', required=True)

    args = parser.parse_args()

    text = ''
    if not os.path.exists(args.model):
        print('The path does not exist') 
        exit()
    if args.input != None:
        for filename in os.listdir(args.input):
            with open(os.path.join(args.input, filename)) as f:
                text = text + '\n' + f.read()
        final_tokens = Model()
        final_tokens.fit(text=text)

        pickle.dump(final_tokens, open(f'{args.model}/model_' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '.pkl', 'wb'))
    # Придумать, как реализвать нормально stdin. Ломается на символах ' ;
    else:
        print(args)
        print('Error')
        exit()
