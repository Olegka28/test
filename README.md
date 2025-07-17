# Crypto Signal Bot 🤖

Telegram бот для торговых сигналов на основе машинного обучения. Бот анализирует криптовалютные пары и предоставляет сигналы для торговли с уровнями Take Profit и Stop Loss.

## 🚀 Возможности

- **ML-анализ**: Использует XGBoost модели для предсказания движения цен
- **Множественные пары**: Поддержка SOL/USDT, BTC/USDT, ETH/USDT, HYPE/USDT
- **Разные таймфреймы**: 15m, 1h, 4h, 1d
- **Интерактивный интерфейс**: Удобные кнопки в Telegram
- **Технические индикаторы**: RSI, MACD, Bollinger Bands, Stochastic и др.
- **Паттерны свечей**: Анализ паттернов для улучшения точности

## 📁 Архитектура проекта

```
crypto-signal-bot/
├── src/
│   ├── config/              # Конфигурация
│   │   └── settings.py      # Настройки приложения
│   ├── data/                # Работа с данными
│   │   └── connectors/      # Коннекторы к биржам
│   │       └── bybit_connector.py
│   ├── features/            # Обработка признаков
│   │   ├── technical_indicators.py
│   │   ├── pattern_features.py
│   │   └── feature_processor.py
│   ├── ml/                  # Машинное обучение
│   │   ├── model_manager.py
│   │   └── train/
│   │       ├── train_model.py
│   │       └── data_builder.py
│   ├── services/            # Бизнес-логика
│   │   └── signal_service.py
│   ├── integrations/        # Интеграции
│   │   └── telegram_bot.py
│   ├── entrypoints/         # Точки входа
│   │   └── bot_runner.py
│   └── utils/               # Утилиты
│       └── logger.py
├── models/                  # Обученные модели
├── data/                    # Данные
├── logs/                    # Логи
├── main.py                  # Главный файл
└── requirements.txt         # Зависимости
```

## 🛠️ Установка

1. **Клонируйте репозиторий:**
```bash
git clone <repository-url>
cd crypto-signal-bot
```

2. **Создайте виртуальное окружение:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

3. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

4. **Настройте переменные окружения:**
Создайте файл `.env` в корне проекта:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
BYBIT_API_KEY=your_bybit_api_key_optional
BYBIT_SECRET_KEY=your_bybit_secret_key_optional
```

## 🤖 Настройка Telegram бота

1. Создайте бота через [@BotFather](https://t.me/botfather)
2. Получите токен бота
3. Добавьте токен в переменную окружения `TELEGRAM_BOT_TOKEN`

## 🚀 Запуск

### Запуск бота:
```bash
python main.py
```

### Альтернативный запуск:
```bash
python src/entrypoints/bot_runner.py
```

## 📊 Использование бота

### Команды:
- `/start` - Начать работу с ботом
- `/help` - Показать справку
- `/overview` - Обзор всех пар

### Процесс работы:
1. **Выберите пару** - нажмите на кнопку с нужной криптовалютой
2. **Получите обзор** - увидите сигналы по всем таймфреймам
3. **Выберите таймфрейм** - нажмите на конкретный таймфрейм для детального сигнала

### Сигналы:
- 🟢 **LONG** - Рекомендуется покупка
- 🔴 **SHORT** - Рекомендуется продажа  
- ⚪ **NONE** - Нет четкого сигнала

## 🎯 Обучение моделей

### Подготовка данных:
```python
from src.ml.train.data_builder import DataBuilder

# Создаем экземпляр
builder = DataBuilder()

# Загружаем исторические данные
data = builder.download_training_data(
    pairs=['SOL/USDT', 'BTC/USDT'],
    timeframes=['15m', '1h', '4h'],
    days=365
)

# Подготавливаем датасеты
datasets = builder.prepare_training_dataset(data)
```

### Обучение моделей:
```python
from src.ml.train.train_model import ModelTrainer

# Создаем тренер
trainer = ModelTrainer()

# Обучаем все модели
results = trainer.train_all_models(datasets)

# Оцениваем результаты
for key, result in results.items():
    report = trainer.evaluate_model(result)
    print(report)
```

## ⚙️ Конфигурация

Основные настройки в `src/config/settings.py`:

```python
# Поддерживаемые пары
SUPPORTED_PAIRS = ['SOL/USDT', 'BTC/USDT', 'ETH/USDT', 'HYPE/USDT']

# Поддерживаемые таймфреймы
SUPPORTED_TIMEFRAMES = ['15m', '1h', '4h', '1d']

# Параметры сигналов
SIGNAL_THRESHOLD = 0.6  # Порог уверенности
TP_RATIO = 0.02        # Take Profit (2%)
SL_RATIO = 0.01        # Stop Loss (1%)
```

## 📈 Технические индикаторы

Бот использует следующие индикаторы:
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands**
- **Stochastic Oscillator**
- **ATR** (Average True Range)
- **Moving Averages** (SMA, EMA)
- **Volume indicators** (VWAP, OBV)

## 🎨 Паттерны свечей

Анализируются паттерны:
- Hammer / Shooting Star
- Engulfing (бычий/медвежий)
- Doji
- Three White Soldiers / Three Black Crows
- Morning Star / Evening Star

## 🔧 Разработка

### Добавление новой пары:
1. Добавьте пару в `SUPPORTED_PAIRS` в `settings.py`
2. Добавьте эмодзи в `telegram_bot.py`
3. Обучите модель для новой пары

### Добавление нового индикатора:
1. Добавьте расчет в `technical_indicators.py`
2. Обновите `feature_processor.py`
3. Переобучите модели

## 📝 Логирование

Логи сохраняются в:
- `logs/bot.log` - основной лог
- Консоль - для отладки

Уровни логирования настраиваются в `src/utils/logger.py`

## ⚠️ Важные замечания

1. **Не финансовый совет** - бот предоставляет только аналитическую информацию
2. **Проводите собственный анализ** - всегда проверяйте сигналы
3. **Управление рисками** - используйте стоп-лоссы и не рискуйте больше, чем можете позволить
4. **Тестирование** - протестируйте на демо-счете перед реальной торговлей

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📄 Лицензия

MIT License

## 📞 Поддержка

Если у вас есть вопросы или проблемы:
1. Создайте Issue в репозитории
2. Проверьте логи в `logs/bot.log`
3. Убедитесь, что все зависимости установлены

---

**Удачной торговли! 🚀📈**

## Переобучение моделей

### Ручной запуск

Для полного переобучения всех моделей (удаляются старые и обучаются новые):

```bash
python scripts/retrain_models.py
```

### Автоматизация (cron)

Чтобы запускать переобучение автоматически раз в месяц (например, 1-го числа в 3:00 ночи), добавьте в crontab:

```
0 3 1 * * cd /path/to/crypto-signal-bot && /path/to/venv/bin/python scripts/retrain_models.py >> logs/retrain.log 2>&1
```

- Замените `/path/to/crypto-signal-bot` и `/path/to/venv/bin/python` на свои пути.
- Логи будут сохраняться в `logs/retrain.log`.

---
