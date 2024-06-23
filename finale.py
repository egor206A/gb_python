from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Предположим, что у вас есть данные X и y
# X - ваши признаки
# y - ваши целевые значения

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализируйте и обучите модель линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Сделайте предсказания на тестовом наборе данных
y_pred = model.predict(X_test)

# Вычислите коэффициент детерминации R2
r2 = r2_score(y_test, y_pred)

if r2 > 0.6:
    print("Коэффициент детерминации R2 больше 0.6:", r2)
else:
    print("Коэффициент детерминации R2 меньше или равен 0.6:", r2)
