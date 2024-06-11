#Тема “Обучение с учителем”
#задание 1

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Загрузка данных
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Разделение на признаки и целевую переменную
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Разбиение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение модели линейной регрессии
lr = LinearRegression()
lr.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = lr.predict(X_test)

#задание 2

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Создание и обучение модели RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
model.fit(X_train, y_train.values[:, 0])

# Предсказание на тестовых данных
y_pred_rf = model.predict(X_test)

# Вычисление R2
r2_rf = r2_score(y_test, y_pred_rf)

print(f'R2 для модели RandomForestRegressor: {r2_rf}')
print(f'R2 для модели LinearRegression из предыдущего задания: {r2_score(y_test, y_pred)}')

#задание 3

from sklearn.ensemble import RandomForestRegressor

# Вывод документации для класса RandomForestRegressor
help(RandomForestRegressor)

# Создание и обучение модели RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
model.fit(X_train, y_train.values[:, 0])

# Нахождение суммы всех показателей важности
feature_importance_sum = sum(model.feature_importances_)

print(f'Сумма всех показателей важности: {feature_importance_sum}')

# Нахождение индексов двух признаков с наибольшей важностью
top_feature_indices = model.feature_importances_.argsort()[-2:][::-1]

print(f'Индексы двух признаков с наибольшей важностью: {top_feature_indices}')

#задание 4
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# Загрузка датасета и создание датафрейма df
df = pd.read_csv('creditcard.csv')

# Убедимся, что выборка несбалансирована
print(df['Class'].value_counts(normalize=True))

# Проверим типы данных и наличие пропусков
print(df.info())

# Настройка для просмотра всех столбцов датафрейма
pd.options.display.max_columns = 100

# Просмотр первых 10 строк датафрейма
print(df.head(10))

# Создание датафрейма X и объекта Series y
X = df.drop('Class', axis=1)
y = df['Class']

# Разбиение на тренировочный и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

# Просмотр информации о форме данных
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Параметры для поиска по сетке
parameters = [{'n_estimators': [10, 15], 'max_features': np.arange(3, 5), 'max_depth': np.arange(4, 7)}]

# Создание модели GridSearchCV
model = GridSearchCV(estimator=RandomForestClassifier(random_state=100), param_grid=parameters, scoring='roc_auc', cv=3)

# Обучение модели на тренировочном наборе данных
model.fit(X_train, y_train)

# Просмотр параметров лучшей модели
print(model.best_params_)

# Предсказание вероятностей классов с помощью полученной модели и метода predict_proba
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Вычисление AUC на тестовых данных
auc_test = roc_auc_score(y_test, y_pred_proba)
print(f'AUC на тестовых данных: {auc_test}')
