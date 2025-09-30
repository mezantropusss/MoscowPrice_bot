# train_model.py
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 1. Загрузка и предварительная обработка данных
logging.info("Загрузка данных из data.csv")
try:
    df = pd.read_csv('data.csv')
    logging.info(f"Загружено {len(df)} записей")
except Exception as e:
    logging.error(f"Ошибка загрузки data.csv: {e}")
    exit(1)

# Функция очистки данных
def clean_data(df):
    # Сохраняем исходное количество строк
    original_count = len(df)
    
    # Удаление выбросов по цене
    q_low = df['Price'].quantile(0.01)
    q_hi = df['Price'].quantile(0.99)
    df = df[(df['Price'] > q_low) & (df['Price'] < q_hi)]
    
    # Фильтрация по площади
    df = df[(df['Area'] > 15) & (df['Area'] < 300)]
    
    # Фильтрация по этажу
    df = df[df['Floor'] > 0]
    
    # Фильтрация по минутам до метро
    df = df[(df['Minutes to metro'] > 0) & (df['Minutes to metro'] < 120)]
    
    # Удаление записей с пропущенными значениями
    df = df.dropna()
    
    # Логирование результатов очистки
    cleaned_count = len(df)
    logging.info(f"Удалено {original_count - cleaned_count} записей при очистке ({cleaned_count} осталось)")
    
    return df

# Применяем очистку данных
logging.info("Начало очистки данных")
df = clean_data(df)

# Удаление полных дубликатов
logging.info("Удаление дубликатов")
initial_count = len(df)
df.drop_duplicates(inplace=True)
logging.info(f"Удалено {initial_count - len(df)} дубликатов")

# 2. Кодирование категориальных признаков
# Регион: бинарное кодирование
logging.info("Кодирование Region")
try:
    df['Region'] = df['Region'].map({'Moscow': 1, 'Moscow Oblast': 0})
except KeyError:
    logging.error("Столбец 'Region' не найден в данных")
    exit(1)

# Станция метро: частотное кодирование
logging.info("Частотное кодирование метро")
try:
    # Нормализация названий станций: удаляем знаки препинания и пробелы
    df['Metro station'] = (
        df['Metro station']
        .str.lower()
        .str.strip()
        .str.replace('[^\w\s]', '', regex=True)
        .str.replace(' ', '')
    )
    metro_freq = df['Metro station'].value_counts(normalize=True).to_dict()
    df['Metro station'] = df['Metro station'].map(metro_freq)
    
    # Логируем количество уникальных станций
    logging.info(f"Обработано {len(metro_freq)} уникальных станций метро")
except KeyError:
    logging.error("Столбец 'Metro station' не найден в данных")
    exit(1)

# Тип жилья: бинарное кодирование
logging.info("Кодирование Apartment type")
try:
    df['Apartment type'] = df['Apartment type'].map({
        'Secondary': 0, 
        'New building': 1,
        'вторичка': 0,  # Дополнительные варианты
        'новостройка': 1
    })
    # Проверяем успешность преобразования
    if df['Apartment type'].isna().any():
        invalid_values = df.loc[df['Apartment type'].isna(), 'Apartment type'].unique()
        logging.error(f"Найдены недопустимые значения для типа жилья: {invalid_values}")
        exit(1)
except KeyError:
    logging.error("Столбец 'Apartment type' не найден в данных")
    exit(1)

# Ремонт: One-Hot Encoding
logging.info("One-Hot кодирование Renovation")
try:
    # Приводим значения к нижнему регистру и обрезаем пробелы
    df['Renovation'] = df['Renovation'].str.lower().str.strip()
    
    # Заменяем значения на стандартные
    renovation_map = {
        'без ремонта': 'No repair',
        'косметический': 'Cosmetic',
        'евроремонт': 'Euro',
        'дизайнерский': 'Euro',  # Считаем дизайнерский как евроремонт
        'требуется ремонт': 'No repair'
    }
    df['Renovation'] = df['Renovation'].map(renovation_map)
    
    # Заменяем отсутствующие значения на самый частый вариант
    if df['Renovation'].isna().any():
        most_common = df['Renovation'].mode()[0]
        df['Renovation'] = df['Renovation'].fillna(most_common)
        logging.warning(f"Заменено {df['Renovation'].isna().sum()} пропущенных значений ремонта на '{most_common}'")
    
    # Применяем One-Hot кодирование
    df = pd.get_dummies(df, columns=['Renovation'], prefix='reno')
    
    # Логируем созданные столбцы
    reno_cols = [col for col in df.columns if col.startswith('reno_')]
    logging.info(f"Созданы столбцы для ремонта: {', '.join(reno_cols)}")
except KeyError:
    logging.error("Столбец 'Renovation' не найден в данных")
    exit(1)

# 3. Обработка числовых признаков
logging.info("Удаление лишних столбцов")
try:
    columns_to_drop = ['Kitchen area', 'Living area', 'Number of floors']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns:
        df.drop(existing_columns, axis=1, inplace=True)
        logging.info(f"Удалены столбцы: {', '.join(existing_columns)}")
    else:
        logging.info("Нет столбцов для удаления")
except Exception as e:
    logging.warning(f"Не удалось удалить столбцы: {e}")

# Выделяем числовые признаки для масштабирования
numerical_features = ['Minutes to metro', 'Number of rooms', 'Area', 'Floor']
logging.info(f"Масштабирование признаков: {', '.join(numerical_features)}")

try:
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Проверяем результат
    logging.info("Масштабирование успешно завершено")
    for feature in numerical_features:
        logging.info(f"{feature}: mean={df[feature].mean():.2f}, std={df[feature].std():.2f}")
except KeyError as e:
    logging.error(f"Отсутствует числовой столбец: {e}")
    exit(1)
except Exception as e:
    logging.error(f"Ошибка масштабирования: {e}")
    exit(1)

# 4. Сохранение артефактов
logging.info("Сохранение scaler и metro_freq")
try:
    # Создаем папку artifacts, если её нет
    os.makedirs('artifacts', exist_ok=True)
    
    # Сохраняем scaler
    scaler_path = os.path.join('artifacts', 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Сохраняем частоты метро
    metro_freq_path = os.path.join('artifacts', 'metro_freq.csv')
    pd.Series(metro_freq).to_csv(metro_freq_path, header=False)
    
    logging.info(f"Артефакты успешно сохранены в папку artifacts/")
except PermissionError:
    logging.error("Ошибка доступа: закройте файлы scaler.pkl и metro_freq.csv")
    exit(1)
except Exception as e:
    logging.error(f"Ошибка сохранения артефактов: {e}")
    exit(1)

# 5. Подготовка обучающего набора
logging.info("Подготовка X и y для обучения модели")
try:
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Логируем размерность данных
    logging.info(f"Размерность признаков: {X.shape}, размерность целевой переменной: {y.shape}")
    
    # Проверяем на пропуски
    if X.isna().any().any():
        logging.error("Обнаружены пропуски в признаках после обработки")
        exit(1)
except KeyError:
    logging.error("Столбец 'Price' не найден в данных")
    exit(1)

# 6. Разбиение на train/test
logging.info("Разбиение данных на train и test (80/20)")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info(f"Обучающая выборка: {X_train.shape[0]} записей")
    logging.info(f"Тестовая выборка: {X_test.shape[0]} записей")
except Exception as e:
    logging.error(f"Ошибка разбиения данных: {e}")
    exit(1)

# 7. Обучение модели CatBoost
logging.info("Инициализация CatBoostRegressor")
try:
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function='MAE',
        eval_metric='MAE',
        verbose=100,
        random_seed=42,
        early_stopping_rounds=50
    )
    logging.info("Начало обучения модели")
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        plot=False
    )
    
    # Логируем результаты
    best_iteration = model.get_best_iteration()
    best_score = model.get_best_score()
    logging.info(f"Обучение завершено. Лучшая итерация: {best_iteration}")
    logging.info(f"Лучший результат MAE: {best_score['validation']['MAE']}")
except Exception as e:
    logging.error(f"Ошибка обучения модели: {e}")
    exit(1)

# 8. Сохранение обученной модели
logging.info("Сохранение модели catboost_model.pkl")
try:
    model_path = os.path.join('artifacts', 'catboost_model.pkl')
    joblib.dump(model, model_path)
    logging.info(f"Модель успешно сохранена по пути: {model_path}")
except PermissionError:
    logging.error("Ошибка доступа: закройте файл catboost_model.pkl")
    exit(1)
except Exception as e:
    logging.error(f"Ошибка сохранения модели: {e}")
    exit(1)

# 9. Вывод списка признаков
logging.info("Feature names used in model:")
feature_names = model.feature_names_
for i, feat in enumerate(feature_names, 1):
    logging.info(f"{i}. {feat}")

# 10. Оценка важности признаков
try:
    feature_importances = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    logging.info("Топ-10 важных признаков:")
    for i, row in importance_df.head(10).iterrows():
        logging.info(f"{row['Feature']}: {row['Importance']:.4f}")
except Exception as e:
    logging.warning(f"Не удалось получить важность признаков: {e}")

logging.info("Обучение модели успешно завершено!")