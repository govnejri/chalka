

**Цель:** Перестать смотреть на `.fit()` как на магию. Понять математику под капотом. Научиться валидировать модели, чтобы не опозориться в продакшене.

---

### 5.1 Math Foundation (The "Why")

Не надо учить всю высшую математику. Нужен конкретный сабсет для ML [web:64][web:69].

**Linear Algebra (Матрицы — это данные)**
*   Векторы, скалярное произведение (dot product).
*   Матричное умножение (основа всех нейронок).
*   Тензоры (многомерные матрицы).

**Calculus (Градиент — это обучение)**
*   Производная и градиент (направление спуска).
*   Chain Rule (как ошибка течет через слои).

**Statistics (Валидация — это правда)**
*   Mean, Median, Variance, Standard Deviation.
*   Распределения (Normal, Uniform).
*   Bias vs Variance (недообучение vs переобучение).

**Задачи (Numpy Only)**
*   [ ] Реализовать матричное умножение на чистом Python (циклами), потом на Numpy. Сравнить скорость.
*   [ ] Написать функцию расчета MSE (Mean Squared Error) руками.

---

### 5.2 Classic ML Algorithms

Алгоритмы, которые решают 80% задач в бизнесе быстрее и дешевле, чем нейронки [web:65][web:70].

**Темы**
1.  **Preprocessing:** Scaling (MinMax, Standard), Encoding (One-Hot, Label), пропуски данных.
2.  **Regression:** Линейная регрессия. Метод наименьших квадратов.
3.  **Classification:** Логистическая регрессия, k-NN, Decision Trees.
4.  **Clustering:** K-Means (без учителя).
5.  **Ensembles:** Random Forest, Gradient Boosting (XGBoost/CatBoost) — короли табличных данных.

**Задачи**
*   [ ] **Titanic EDA:** Взять датасет Титаника. Заполнить пропуски (средним/медианой). Визуализировать корреляции.
*   [ ] **House Prices:** Обучить Linear Regression для предсказания цены. Посчитать метрику R2.
*   [ ] **Classifier:** Сравнить Decision Tree и Random Forest на задаче классификации ирисов. Построить Confusion Matrix.

---

### 5.3 ML Operations (MLOps Lite)

Как жить с моделью после обучения.

**Темы**
*   **Metrics:** Accuracy не работает на дисбалансе. Precision, Recall, F1-score, ROC-AUC.
*   **Validation:** Train/Test split. Cross-Validation (K-Fold).
*   **Saving:** Pickle, Joblib, ONNX (кроссплатформенный формат).

**Capstone Project: ML Microservice**
*Требования:*
1.  Обучить модель (например, предсказание стоимости авто).
2.  Сохранить веса в файл.
3.  Написать FastAPI сервис: принимает JSON с параметрами авто -> загружает модель -> отдает цену.
4.  Упаковать в Docker.

---

## 6. Deep Learning Foundations

**Цель:** Работа с "сырыми" данными (картинки, текст). Понимание архитектур PyTorch [web:66][web:71].

---

### 6.1 Neural Networks Core

От логистической регрессии к нейрону.

**Темы**
*   **Perceptron:** Взвешенная сумма + функция активации.
*   **Activation Functions:** Sigmoid (устарела), ReLU (стандарт), Softmax (для классификации).
*   **Loss Functions:** MSE (регрессия), Cross-Entropy (классификация).
*   **Optimization:** SGD, Adam (де-факто стандарт).
*   **Backpropagation:** Как обновляются веса (на пальцах, но с пониманием).

**Задачи**
*   [ ] Написать простую нейросеть на чистом Numpy (forward pass и backward pass).
*   [ ] Обучить "Hello World" в PyTorch: аппроксимация функции `y = 2x + 3`.

---

### 6.2 Computer Vision (CNN)

Как роботы видят.

**Темы**
*   **Convolution:** Свертка, ядро, страйд, паддинг. Как фильтры ищут грани и текстуры.
*   **Pooling:** MaxPool (уменьшение размерности).
*   **Architectures:** ResNet (skip connections — революция), YOLO (кратко про детекцию).
*   **Transfer Learning:** Взять предобученный ResNet, заморозить веса, переобучить последний слой под свои классы.

**Задачи**
*   [ ] **MNIST Classifier:** Написать CNN для распознавания рукописных цифр. Добиться точности >98%.
*   [ ] **Dog vs Cat:** Использовать Transfer Learning (ResNet18) для классификации кошек и собак.

---

### 6.3 NLP & Transformers (LLM Basics)

Как роботы читают.

**Темы**
*   **Embeddings:** Word2Vec. Слова как векторы.
*   **Attention Mechanism:** "Attention is all you need". Почему трансформеры победили RNN.
*   **Architecture:** Encoder (BERT), Decoder (GPT).
*   **HuggingFace:** Использование готовых токенизаторов и моделей.

**Задачи**
*   [ ] **Sentiment Analysis:** Взять предобученный BERT (через библиотеку `transformers`), дообучить (fine-tune) на датасете отзывов (позитивный/негативный).
*   [ ] Сделать инференс скрипт: вводишь текст -> получаешь эмоцию.

---

### 6.4 Final Boss: Full-Stack AI Application

Объединяем всё: Backend, Docker, DB, Neural Network.

**Проект: "Audio Note Taker"**
*Сценарий:* Пользователь загружает голосовое сообщение -> Система транскрибирует текст -> Выделяет ключевые теги.

**Стек:**
1.  **Backend:** FastAPI (асинхронная загрузка файлов).
2.  **AI Worker:** Celery + Redis (очередь задач, чтобы не блокировать API).
3.  **Models:** Whisper (OpenAI) для транскрипции.
4.  **Storage:** Postgres (метаданные), S3/MinIO (файлы).
5.  **Deploy:** Docker Compose.

**Чек-лист:**
- [ ] Асинхронная обработка (пользователь получает ID задачи и поллит статус).
- [ ] Docker-контейнеры для API, Worker, Redis, DB.
- [ ] Оптимизация: модель загружается в память один раз при старте воркера.

---

## Recommended Resources (DL)
*   **Andrej Karpathy (YouTube):** "Zero to Hero". Лучшее объяснение нейросетей в интернете.
*   **Fast.ai:** Курс "Practical Deep Learning for Coders". Сразу код, потом теория.
*   **PyTorch Blitz:** Официальный туториал на сайте PyTorch.org.
