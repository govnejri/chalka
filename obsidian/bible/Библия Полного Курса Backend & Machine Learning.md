

План развития от скриптов до Deep Learning архитектуры.
Акцент: практика, архитектура, code review, реальные практики.

## 0. Протокол работы

**Правила**
1. Git Flow обязателен. Ветки: `feature/name`, Merge Request, Code Review.
2. Никакого мерджа в `main` без аппрува второго участника.
3. Решаем задачи раздельно, обсуждаем решения вместе.

**Инструментарий**
- IDE: VS Code 
- OS: Linux / WSL / Windows - в начале
- Notes: Obsidian / Телега

---

## 1. Python Advanced & OOP

**Цель:** Перестать писать процедурные скрипты. Понять, как Python работает с памятью и объектами.

**Темы**
- Модель памяти: mutability, ссылки, garbage collection.
- ООП: наследование, полиморфизм, инкапсуляция, MRO.
- Магические методы: `__init__`, `__str__`, `__call__`, `__eq__`, `__enter__`.
- Итераторы и генераторы: `yield`, `iter`, `next`.
- Декораторы и замыкания.
- Typing: `mypy`, Type Hints.

**Задачи**
- [ ] Реализовать свой класс `Matrix` с поддержкой сложения и умножения через перегрузку операторов.
- [ ] Написать контекстный менеджер `Timer` для замера времени выполнения кода.
- [ ] Реализовать кэширующий декоратор `lru_cache` с нуля.
- [ ] Разобрать устройство `dict` и `set` (хеш-таблицы).

---

## 2. Algorithms & Data Structures

**Цель:** Мышление инженера. Оптимизация сложности O(n).

**Темы**
- Big O Notation.
- Массивы, связные списки, стеки, очереди.
- Хеш-таблицы.
- Деревья и графы (BFS, DFS).
- Сортировки и бинарный поиск.

**LeetCode Практика (Blind 75)**
*Решать ежедневно по 1-2 задачи.*
- [ ] Arrays & Hashing: Two Sum, Contains Duplicate.
- [ ] Two Pointers: Valid Palindrome.
- [ ] Sliding Window: Best Time to Buy and Sell Stock.
- [ ] Stack: Valid Parentheses.
- [ ] LinkedList: Reverse Linked List.

---

## 3. Systems Engineering

**Цель:** Понимание среды, в которой живет код.

**Linux & Bash**
- Права доступа (chmod, chown).
- I/O потоки, pipe `|`, grep, awk.
- Process management (ps, kill, top).

**Networks**
- Модель OSI (кратко).
- HTTP/HTTPS: заголовки, методы, статус-коды.
- JSON vs XML.
- REST архитектура.

**Databases**
- SQL: SELECT, JOIN, GROUP BY, INDEX.
- ACID транзакции.
- Проектирование схемы: Normalization, Foreign Keys.

**Задачи**
- [ ] Написать Bash-скрипт для автоматического бэкапа проекта.
- [ ] Спроектировать ER-диаграмму БД для клона Twitter.
- [ ] Написать сложные SQL запросы с JOIN и агрегацией.

---

## 4. Backend Production

**Цель:** Построение масштабируемых веб-сервисов.

**Темы**
- AsyncIO: event loop, coroutines, async/await.
- FastAPI: Pydantic, Dependency Injection.
- Docker: Dockerfile, docker-compose, networking.
- Testing: Pytest, mocks, fixtures.

**Capstone Project: Task Manager API**
*Требования:*
- [ ] Аутентификация JWT.
- [ ] CRUD операций с задачами.
- [ ] PostgreSQL в docker-compose.
- [ ] Миграции БД (Alembic).
- [ ] Покрытие тестами >80%.

---

## 5. ML Foundations

**Цель:** Математика и работа с данными.

**Темы**
- Linear Algebra: векторы, матрицы, операции.
- Statistics: распределения, bias/variance.
- Libraries: NumPy, Pandas, Matplotlib.
- Classic ML: Linear Regression, Logistic Regression, Decision Trees.

**Задачи**
- [ ] EDA (Exploratory Data Analysis) датасета Titanic.
- [ ] Реализовать Линейную Регрессию на чистом NumPy (без scikit-learn).
- [ ] Обучить модель классификации и оценить метрики (Accuracy, F1, ROC-AUC).

---

## 6. Deep Learning

**Цель:** Нейросети и работа с неструктурированными данными.

**Темы**
- PyTorch tensors & autograd.
- Архитектуры: MLP, CNN, RNN/LSTM.
- Backpropagation: как обучаются сети.
- Training Loop: loss functions, optimizers.

**Final Project: AI-Service**
*Интеграция всех этапов.*
1. Сервис на FastAPI.
2. Модель PyTorch внутри (классификация изображений или анализ текста).
3. Очередь задач (Redis/Celery) для тяжелых вычислений.
4. Деплой в Docker.

---


