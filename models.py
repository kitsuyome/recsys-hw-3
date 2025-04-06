import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class HeuristicModel:
    def __init__(self, default_score=None):
        """
        Инициализация эвристической модели.
        Если default_score не указан, он будет установлен в глобальное среднее значение target.
        """
        self.default_score = default_score
        self.item_popularity = {}
        self.global_mean = None

    def fit(self, df: pd.DataFrame):
        """
        Обучение модели на переданном DataFrame.
        Ожидается, что df содержит столбец 'target'.
        Вычисляем популярность каждого item как сумму target.
        Также вычисляем глобальное среднее, которое используется как дефолтное значение.
        """
        
        self.item_popularity = df.groupby('item_id')['target'].sum().to_dict()
        
        self.global_mean = df['target'].mean()
        
        if self.default_score is None:
            self.default_score = self.global_mean

    def predict(self, user_id, item_id):
        """
        Предсказывает релевантность для заданной пары (user_id, item_id).
        Поскольку модель эвристическая, для пользователя ничего не делаем — возвращаем популярность айтема.
        Если item_id отсутствует в данных обучения, возвращается дефолтное значение.
        """
        return self.item_popularity.get(item_id, self.default_score)

    def recommend(self, user_id, k=10):
        """
        Для заданного пользователя возвращает топ-k самых популярных айтемов.
        В данной реализации рекомендация не зависит от конкретного пользователя, а основана только на общей популярности.
        """
        if not self.item_popularity:
            raise Exception("Модель не обучена. Сначала вызовите метод fit().")
        
        sorted_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
        top_items = [item for item, score in sorted_items[:k]]
        return top_items



class MatrixFactorization:
    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.1, n_epochs=10, random_state=42, default_score=0.0):
        """
        Инициализация модели матричной факторизации.
        
        :param n_factors: Размерность скрытых факторов.
        :param learning_rate: Скорость обучения.
        :param reg: Коэффициент регуляризации.
        :param n_epochs: Количество эпох обучения.
        :param random_state: Начальное состояние для генератора случайных чисел (для воспроизводимости).
        :param default_score: Дефолтное значение прогноза, если пользователь или айтем отсутствует в обучении.
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.default_score = default_score
        
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        
        self.user_to_index = {}
        self.item_to_index = {}
        self.index_to_user = {}
        self.index_to_item = {}

    def fit(self, df: pd.DataFrame):
        """
        Обучение модели на переданном DataFrame.
        Ожидается наличие столбцов 'user_id', 'item_id' и 'target'.
        Используется стохастический градиентный спуск (SGD) для оптимизации параметров.
        """
        
        user_ids = df['user_id'].unique()
        item_ids = df['item_id'].unique()
        self.user_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
        self.item_to_index = {item_id: i for i, item_id in enumerate(item_ids)}
        self.index_to_user = {i: user_id for user_id, i in self.user_to_index.items()}
        self.index_to_item = {i: item_id for item_id, i in self.item_to_index.items()}
        
        n_users = len(user_ids)
        n_items = len(item_ids)
        
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(n_items, self.n_factors))
        
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        self.global_mean = df['target'].mean()
        
        for epoch in range(self.n_epochs):
            df_shuffled = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            for _, row in df_shuffled.iterrows():
                u = self.user_to_index[row['user_id']]
                i = self.item_to_index[row['item_id']]
                rating = row['target']
                
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i] + np.dot(self.user_factors[u], self.item_factors[i])
                error = rating - pred
                
                self.user_bias[u] += self.learning_rate * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (error - self.reg * self.item_bias[i])
                
                user_factor_old = self.user_factors[u].copy()
                self.user_factors[u] += self.learning_rate * (error * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.learning_rate * (error * user_factor_old - self.reg * self.item_factors[i])

    def predict(self, user_id, item_id):
        """
        Предсказывает релевантность для заданной пары (user_id, item_id).
        Если пользователь или айтем отсутствуют в обучении, возвращается дефолтное значение.
        """
        u_index = self.user_to_index.get(user_id, None)
        i_index = self.item_to_index.get(item_id, None)
        if u_index is None or i_index is None:
            return self.default_score
        score = self.global_mean + self.user_bias[u_index] + self.item_bias[i_index] + np.dot(self.user_factors[u_index], self.item_factors[i_index])
        return score

    def recommend(self, user_id, k=10):
        """
        Для заданного пользователя возвращает топ-k айтемов с наивысшими предсказанными оценками.
        Если пользователь отсутствует в обучении, возвращается пустой список.
        """
        if user_id not in self.user_to_index:
            return []
        u_index = self.user_to_index[user_id]
        predictions = self.global_mean + self.user_bias[u_index] + self.item_bias + np.dot(self.item_factors, self.user_factors[u_index])
        top_k_indices = np.argsort(predictions)[::-1][:k]
        top_k_items = [self.index_to_item[i] for i in top_k_indices]
        return top_k_items

class NeuralRecSys(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=20, hidden_dim=64):
        """
        Нейронная модель для рекомендательной системы.
        
        :param n_users: количество уникальных пользователей
        :param n_items: количество уникальных айтемов
        :param embedding_dim: размерность эмбеддингов
        :param hidden_dim: размер скрытого слоя
        """
        super(NeuralRecSys, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, user_indices, item_indices):
        """
        Прямой проход модели.
        :param user_indices: тензор с индексами пользователей
        :param item_indices: тензор с индексами айтемов
        :return: предсказание релевантности (скалярное значение)
        """
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        x = torch.cat([user_emb, item_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze()
        return x + self.global_bias

class NeuralRecSysModel:
    def __init__(self, embedding_dim=20, hidden_dim=64, lr=0.01, n_epochs=10, batch_size=1024, default_score=0.0, device=None):
        """
        Обёртка для нейронной модели рекомендаций.
        
        :param embedding_dim: размерность эмбеддингов
        :param hidden_dim: размер скрытого слоя
        :param lr: скорость обучения
        :param n_epochs: число эпох обучения
        :param batch_size: размер батча
        :param default_score: значение, возвращаемое при отсутствии пользователя или айтема
        :param device: устройство для вычислений (cpu или cuda)
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.default_score = default_score
        self.device = device if device is not None else torch.device("cpu")
        
        self.model = None
        self.optimizer = None
        
        self.user_to_index = {}
        self.item_to_index = {}
        self.index_to_user = {}
        self.index_to_item = {}

    def fit(self, df: pd.DataFrame):
        """
        Обучает модель на DataFrame с колонками 'user_id', 'item_id' и 'target'.
        """
        
        user_ids = df['user_id'].unique()
        item_ids = df['item_id'].unique()
        self.user_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_to_index = {iid: idx for idx, iid in enumerate(item_ids)}
        self.index_to_user = {idx: uid for uid, idx in self.user_to_index.items()}
        self.index_to_item = {idx: iid for iid, idx in self.item_to_index.items()}
        
        n_users = len(user_ids)
        n_items = len(item_ids)
        
        self.model = NeuralRecSys(n_users, n_items, self.embedding_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        user_indices = df['user_id'].map(self.user_to_index).values
        item_indices = df['item_id'].map(self.item_to_index).values
        ratings = df['target'].values.astype(np.float32)
        
        user_tensor = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        ratings_tensor = torch.tensor(ratings, dtype=torch.float, device=self.device)
        
        dataset = TensorDataset(user_tensor, item_tensor, ratings_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for batch_users, batch_items, batch_ratings in dataloader:
                self.optimizer.zero_grad()
                preds = self.model(batch_users, batch_items)
                loss = nn.MSELoss()(preds, batch_ratings)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_ratings.size(0)
            total_loss /= len(dataset)

    def predict(self, user_id, item_id):
        """
        Возвращает предсказанный скор релевантности для пары (user_id, item_id).
        Если пользователь или айтем не найдены, возвращается default_score.
        """
        if user_id not in self.user_to_index or item_id not in self.item_to_index:
            return self.default_score
        
        self.model.eval()
        with torch.no_grad():
            u_idx = torch.tensor([self.user_to_index[user_id]], dtype=torch.long, device=self.device)
            i_idx = torch.tensor([self.item_to_index[item_id]], dtype=torch.long, device=self.device)
            pred = self.model(u_idx, i_idx).item()
        return pred

    def recommend(self, user_id, k=10):
        """
        Для заданного пользователя возвращает список топ-k рекомендованных айтемов.
        Если пользователь не найден, возвращается пустой список.
        """
        if user_id not in self.user_to_index:
            return []
        
        self.model.eval()
        u_idx = self.user_to_index[user_id]
        n_items = len(self.item_to_index)
        with torch.no_grad():
            user_tensor = torch.tensor([u_idx] * n_items, dtype=torch.long, device=self.device)
            all_items = torch.arange(n_items, dtype=torch.long, device=self.device)
            preds = self.model(user_tensor, all_items)
            preds = preds.cpu().numpy()
            top_k_indices = preds.argsort()[::-1][:k]
            recommended_items = [self.index_to_item[idx] for idx in top_k_indices]
        return recommended_items