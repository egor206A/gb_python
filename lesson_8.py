import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Загрузка данных
boston = load_boston()
data = boston.data
target = boston.target
feature_names = boston.feature_names

# Создание датафреймов X и y
X = pd.DataFrame(data, columns=feature_names)
y = pd.DataFrame(target, columns=['PRICE'])

# Разбиение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Построение модели TSNE
tsne = TSNE(n_components=2, learning_rate=250, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)

# Построение диаграммы рассеяния
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])
plt.title('TSNE Scatter Plot')
plt.show()

# Использование KMeans для разбиения на кластеры
kmeans = KMeans(n_clusters=3, max_iter=100, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)

# Построение диаграммы рассеяния с раскраской точек из разных кластеров
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=clusters)
plt.title('TSNE Scatter Plot with KMeans Clusters')
plt.show()

# Вычисление средних значений price и CRIM в разных кластерах
X_train['Cluster'] = clusters
cluster_means = X_train.groupby('Cluster').agg({'PRICE': 'mean', 'CRIM': 'mean'})
print(cluster_means)

# Применение модели KMeans к данным из тестового набора
test_clusters = kmeans.predict(X_test_scaled)

# Вычисление средних значений price и CRIM в разных кластерах на тестовых данных
X_test['Cluster'] = test_clusters
test_cluster_means = X_test.groupby('Cluster').agg({'PRICE': 'mean', 'CRIM': 'mean'})
print(test_cluster_means)
