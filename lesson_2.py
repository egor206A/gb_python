#Тема “Визуализация данных в Matplotlib”
#1 задание

import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]

plt.plot(x, y)
plt.show()

Для построения диаграммы рассеяния (scatter plot) в следующей ячейке можно использовать следующий код:
plt.scatter(x, y)
plt.show()

#2 задание
 import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 51)
f = np.cos(t)

plt.plot(t, f, color='green')
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.xlim(0.5, 9.5)
plt.ylim(-2.5, 2.5)
plt.show()

#3 задание

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 51)
y1 = x**2
y2 = 2 * x + 0.5
y3 = -3 * x - 1.5
y4 = np.sin(x)

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

ax[0, 0].plot(x, y1)
ax[0, 0].set_title('График y1')
ax[0, 0].set_xlim(-5, 5)

ax[0, 1].plot(x, y2)
ax[0, 1].set_title('График y2')

ax[1, 0].plot(x, y3)
ax[1, 0].set_title('График y3')

ax[1, 1].plot(x, y4)
ax[1, 1].set_title('График y4')

plt.show()

#4 задание
import pandas as pd
import matplotlib.pyplot as plt

# Импортируем данные
data = pd.read_csv('creditcard.csv')

# Устанавливаем стиль графиков "fivethirtyeight"
plt.style.use('fivethirtyeight')

# Строим столбчатую диаграмму для количества наблюдений каждого значения целевой переменной Class
data['Class'].value_counts().plot(kind='bar')
plt.title('Количество наблюдений для каждого значения Class')
plt.show()

# Строим столбчатую диаграмму с логарифмическим масштабом
data['Class'].value_counts().plot(kind='bar')
plt.yscale('log')
plt.title('Количество наблюдений для каждого значения Class (логарифмический масштаб)')
plt.show()

# Строим гистограммы для признака V1 по значениям Class
plt.hist(data[data['Class'] == 0]['V1'], bins=20, density=True, alpha=0.5, color='gray', label='Class 0')
plt.hist(data[data['Class'] == 1]['V1'], bins=20, density=True, alpha=0.5, color='red', label='Class 1')
plt.xlabel('V1')
plt.legend()
plt.show()
