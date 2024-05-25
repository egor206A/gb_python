#Тема “Вычисления с помощью Numpy”

#1 задание
import numpy as np

# Создание массива Numpy
a = np.array([[1, 6],
              [2, 8],
              [3, 11],
              [3, 10],
              [1, 7]])

# Нахождение среднего значения по каждому признаку
mean_a = np.mean(a, axis=0)

print("Массив a:")
print(a)

print("\nСреднее значение по каждому признаку (mean_a):")
print(mean_a)

#2 задание
import numpy as np

# Создание массива Numpy
a = np.array([[1, 6],
              [2, 8],
              [3, 11],
              [3, 10],
              [1, 7]])

# Нахождение среднего значения по каждому признаку
mean_a = np.mean(a, axis=0)

# Вычисление массива a_centered
a_centered = a - mean_a

print("Массив a_centered:")
print(a_centered)

#3 задание 
import numpy as np

# Создание массива a_centered
a_centered = np.array([[-1. , -2.4],
                       [ 0. ,  0.6],
                       [ 1. ,  3.6],
                       [ 1. ,  2.6],
                       [-1. , -0.4]])

# Нахождение скалярного произведения столбцов массива a_centered
a_centered_sp = np.dot(a_centered[:, 0], a_centered[:, 1])

# Нахождение числа наблюдений N
N = a_centered.shape[0]

# Деление скалярного произведения на N-1
result = a_centered_sp / (N - 1)

print("Значение _centered_sp:", result)

#Тема “Работа с данными в Pandas”
#1 задание
import pandas as pd

# Создание датафрейма authors
authors = pd.DataFrame({
    'author_id': [1, 2, 3],
    'author_name': ['Тургенев', 'Чехов', 'Островский']
})

# Создание датафрейма books
books = pd.DataFrame({
    'author_id': [1, 1, 1, 2, 2, 3, 3],
    'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
    'price': [500, 400, 300, 350, 450, 600, 200]
})

# Вывод созданных датафреймов
print("Датафрейм authors:")
print(authors)

print("\nДатафрейм books:")
print(books)

#2 задание
import pandas as pd

# Создание датафрейма authors
authors = pd.DataFrame({
    'author_id': [1, 2, 3],
    'author_name': ['Тургенев', 'Чехов', 'Островский']
})

# Создание датафрейма books
books = pd.DataFrame({
    'author_id': [1, 1, 1, 2, 2, 3, 3],
    'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
    'price': [500, 400, 300, 350, 450, 600, 200]
})

# Объединение датафреймов по полю author_id
authors_price = pd.merge(authors, books, on='author_id')

# Вывод объединенного датафрейма authors_price
print("Объединенный датафрейм authors_price:")
print(authors_price)

#3 задание
# Сортировка датафрейма authors_price по убыванию цены книг
sorted_authors_price = authors_price.sort_values(by='price', ascending=False)

# Выбор пяти самых дорогих книг
top5 = sorted_authors_price.head(5)

# Вывод датафрейма top5
print("Датафрейм top5 с пятью самыми дорогими книгами:")
print(top5)

#4 задание
import pandas as pd

# Группировка по автору и вычисление минимальной, максимальной и средней цены
authors_stat = authors_price.groupby('author_name').agg({'price': ['min', 'max', 'mean']}).reset_index()

# Переименование столбцов для удобства
authors_stat.columns = ['author_name', 'min_price', 'max_price', 'mean_price']

# Вывод датафрейма authors_stat
print("Датафрейм authors_stat с информацией о ценах книг по каждому автору:")
print(authors_stat)

