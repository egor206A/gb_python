from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Генерируем случайные данные
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Создаем экземпляр модели KMeans с количеством кластеров равным 4
kmeans = KMeans(n_clusters=4)

# Обучаем модель на данных
kmeans.fit(X)

# Получаем предсказанные метки кластеров
y_kmeans = kmeans.predict(X)

# Визуализируем результаты
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)

plt.show()