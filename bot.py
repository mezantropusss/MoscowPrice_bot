from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.filters.state import StateFilter
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import pandas as pd
import joblib
import logging
import re
import os
import string

# Настройки
API_TOKEN = "8086258501:AAHK59i3QHR0OGrx9wTAXLEd4Y4Z5qK7wDY"
UNCERTAINTY = 0.10  # 10% доверительный интервал
MAX_REASONABLE_PRICE = 300_000_000
MIN_REASONABLE_PRICE = 1_500_000
PRICE_ADJUSTMENT_FACTOR_MOSCOW = 1.2  # Для Москвы
PRICE_ADJUSTMENT_FACTOR_MO = 1.08      # Для МО

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Инициализация бота и FSM
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Шаги диалога
class Form(StatesGroup):
    region = State()
    apt_type = State()
    renovation = State()
    district = State()
    station = State()
    time = State()
    rooms = State()
    area = State()
    floor = State()

# Загрузка модели и артефактов
try:
    os.makedirs('artifacts', exist_ok=True)
    model = joblib.load('artifacts/catboost_model.pkl')
    scaler = joblib.load('artifacts/scaler.pkl')
    
    metro_freq_series = pd.read_csv('artifacts/metro_freq.csv', header=None, index_col=0).squeeze()
    metro_freq = {}
    for station, freq in metro_freq_series.items():
        key = station.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
        metro_freq[key] = freq
    
    logger.info("✅ Модель и артефакты успешно загружены")
    logger.info(f"📊 Загружено {len(metro_freq)} станций метро")
    
    avg_freq = sum(metro_freq.values()) / len(metro_freq) if metro_freq else 0
    logger.info(f"📈 Средняя частота станций: {avg_freq:.4f}")
    
except Exception as e:
    logger.error(f"❌ Ошибка загрузки: {e}")
    raise SystemExit("Не удалось загрузить модель. Переобучите модель и запустите бота снова")

# Справочники
district_map = {
    'ЦАО (центр)': ['Тверская', 'Пушкинская', 'Охотный Ряд', 'Чеховская', 'Кузнецкий мост', 'Лубянка'],
    'СВАО (северо-восток)': ['ВДНХ', 'Бабушкинская', 'Ботанический Сад', 'Алексеевская', 'Рижская', 'Медведково'],
    'ЮВАО (юго-восток)': ['Выхино', 'Люблино', 'Покрышкина', 'Кузьминки', 'Текстильщики', 'Рязанский проспект'],
    'ЗАО (запад)': ['Славянский бульвар', 'Кунцевская', 'Пионерская', 'Фили', 'Студенческая', 'Киевская'],
    'Химки': ['Планерная', 'Речной вокзал', 'Ховрино', 'Беломорская', 'Химки (город)'],
    'Красногорск': ['Мякинино', 'Пятницкое шоссе', 'Волоколамская', 'Тушинская'],
    'Мытищи': ['Медведково', 'Бульвар Дмитрия Донского', 'Бибирево', 'Алтуфьево']
}

reno_map = {
    'Без ремонта': 'No repair',
    'Косметический': 'Cosmetic',
    'Евроремонт': 'Euro'
}

back_btn = types.KeyboardButton(text='🔙 Назад')
reset_btn = types.KeyboardButton(text='🔄 Начать заново')

@dp.message(lambda m: m.text == reset_btn.text)
async def on_reset(message: types.Message, state: FSMContext):
    await state.clear()
    return await cmd_start(message, state)

@dp.message(Command(commands=['start', 'restart']))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    kb = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text='Москва'), types.KeyboardButton(text='МО')],
            [reset_btn]
        ],
        resize_keyboard=True
    )
    await message.answer('🏙️ <b>Оценка стоимости квартиры</b>\n\nВыберите регион:', 
                         reply_markup=kb, parse_mode='HTML')
    await state.set_state(Form.region)

@dp.message(StateFilter(Form.region))
async def process_region(message: types.Message, state: FSMContext):
    if message.text not in ['Москва', 'МО']:
        return await message.answer('Пожалуйста, нажмите одну из кнопок.')
    await state.update_data(region=1 if message.text == 'Москва' else 0)

    kb = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text='Новостройка'), types.KeyboardButton(text='Вторичка')],
            [back_btn, reset_btn]
        ],
        resize_keyboard=True
    )
    await message.answer('🏠 Тип жилья:', reply_markup=kb)
    await state.set_state(Form.apt_type)

@dp.message(StateFilter(Form.apt_type))
async def process_apt_type(message: types.Message, state: FSMContext):
    if message.text == back_btn.text:
        return await cmd_start(message, state)
    if message.text not in ['Новостройка', 'Вторичка']:
        return await message.answer('Нажмите одну из кнопок.')
    await state.update_data(apt_type=1 if message.text == 'Новостройка' else 0)

    kb = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text='Без ремонта'),
             types.KeyboardButton(text='Косметический'),
             types.KeyboardButton(text='Евроремонт')],
            [back_btn, reset_btn]
        ],
        resize_keyboard=True
    )
    await message.answer('🛠️ Тип ремонта:', reply_markup=kb)
    await state.set_state(Form.renovation)

@dp.message(StateFilter(Form.renovation))
async def process_renovation(message: types.Message, state: FSMContext):
    if message.text == back_btn.text:
        return await process_apt_type(message, state)
    if message.text not in reno_map:
        return await message.answer('Нажмите одну из кнопок.')
    await state.update_data(renovation=message.text)

    data = await state.get_data()
    opts = (
        ['ЦАО (центр)', 'СВАО (северо-восток)', 'ЮВАО (юго-восток)', 'ЗАО (запад)']
        if data['region'] == 1
        else ['Химки', 'Красногорск', 'Мытищи']
    )
    kb = types.ReplyKeyboardMarkup(
        keyboard=[[types.KeyboardButton(text=o)] for o in opts] + [[back_btn, reset_btn]],
        resize_keyboard=True
    )
    await message.answer('📍 Выберите район:', reply_markup=kb)
    await state.set_state(Form.district)

@dp.message(StateFilter(Form.district))
async def process_district(message: types.Message, state: FSMContext):
    if message.text == back_btn.text:
        return await process_renovation(message, state)
    if message.text not in district_map:
        return await message.answer('Пожалуйста, выберите район из списка.')
    
    await state.update_data(district=message.text)

    stations = district_map.get(message.text, [])
    # Сортируем станции по популярности (частоте встречаемости)
    top_stations = sorted(
        stations, 
        key=lambda s: metro_freq.get(
            s.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', ''), 
            0
        ), 
        reverse=True
    )[:6]
    
    # Формируем кнопки (по 2 в ряд)
    rows = []
    for i in range(0, len(top_stations), 2):
        row = top_stations[i:i+2]
        rows.append([types.KeyboardButton(text=name) for name in row])
    
    rows.append([types.KeyboardButton(text='✏️ Ввести вручную')])
    rows.append([back_btn, reset_btn])

    kb = types.ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)
    await message.answer('🚇 Выберите станцию метро:', reply_markup=kb)
    await state.set_state(Form.station)

@dp.message(StateFilter(Form.station))
async def process_station(message: types.Message, state: FSMContext):
    txt = message.text
    if txt == back_btn.text:
        return await process_district(message, state)
    if txt == reset_btn.text:
        return await on_reset(message, state)
    
    # Если пользователь выбрал ручной ввод
    if txt == '✏️ Ввести вручную':
        kb = types.ReplyKeyboardMarkup(keyboard=[[back_btn, reset_btn]], resize_keyboard=True)
        await message.answer('✏️ Введите название станции метро:', reply_markup=kb)
        return  # Остаемся в том же состоянии
    
    # Нормализация названия станции (как при обучении)
    station_norm = txt.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
    station_freq = metro_freq.get(station_norm, 0)
    
    # Если станция не найдена, используем среднюю частоту
    if station_freq == 0:
        avg_freq = sum(metro_freq.values()) / len(metro_freq)
        station_freq = avg_freq
        logger.warning(f"Станция '{txt}' не найдена, используется средняя частота: {avg_freq:.4f}")
    
    await state.update_data(
        station=txt,
        station_freq=station_freq
    )
    
    kb = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text='1-5 мин'), types.KeyboardButton(text='6-10 мин')],
            [types.KeyboardButton(text='11-15 мин'), types.KeyboardButton(text='15+ мин')],
            [back_btn, reset_btn]
        ],
        resize_keyboard=True
    )
    await message.answer('⏱️ Минут до метро:', reply_markup=kb)
    await state.set_state(Form.time)

@dp.message(StateFilter(Form.time))
async def process_time(message: types.Message, state: FSMContext):
    if message.text == back_btn.text:
        return await process_station(message, state)
    t = message.text
    
    # Сохраняем оригинальный текст для отображения
    time_display = t.replace(' мин', '')  # Убираем "мин" для отображения
    
    try:
        if '-' in t:
            parts = t.split('-')
            lo = int(parts[0])
            hi = int(parts[1].replace('+', '').replace(' мин', ''))
            minutes = (lo + hi) / 2
        elif t.endswith('+ мин') or t.endswith('+'):
            minutes = int(t.replace('+ мин', '').replace(' мин', '').replace('+', '')) + 10
        else:
            minutes = float(t.replace(',', '.').replace(' мин', ''))
    except (ValueError, IndexError):
        return await message.answer('Неверный формат. Используйте кнопки.')
            
    await state.update_data(minutes=minutes, time_display=time_display)

    kb = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text='1 комната'), types.KeyboardButton(text='2 комнаты')],
            [types.KeyboardButton(text='3 комнаты'), types.KeyboardButton(text='4+ комнаты')],
            [types.KeyboardButton(text='✏️ Ввести вручную')],
            [back_btn, reset_btn]
        ],
        resize_keyboard=True
    )
    await message.answer('🚪 Количество комнат:', reply_markup=kb)
    await state.set_state(Form.rooms)

@dp.message(StateFilter(Form.rooms))
async def process_rooms(message: types.Message, state: FSMContext):
    if message.text == back_btn.text:
        return await process_time(message, state)
    txt = message.text
    if txt == '✏️ Ввести вручную':
        kb = types.ReplyKeyboardMarkup(keyboard=[[back_btn, reset_btn]], resize_keyboard=True)
        return await message.answer('✏️ Введите число комнат:', reply_markup=kb)
    
    try:
        if txt.isdigit():
            rooms = int(txt)
        elif txt == '4+ комнаты':
            rooms = 4
        else:
            # Извлекаем первое число из текста
            match = re.search(r'\d+', txt)
            if not match:
                raise ValueError
            rooms = int(match.group())
            
        if rooms < 1 or rooms > 10:
            raise ValueError
    except (ValueError, AttributeError):
        return await message.answer('Неверный формат. Введите число от 1 до 10.')
    
    await state.update_data(rooms=rooms)

    kb = types.ReplyKeyboardMarkup(keyboard=[[back_btn, reset_btn]], resize_keyboard=True)
    await message.answer('📏 Введите площадь квартиры в м²:', reply_markup=kb)
    await state.set_state(Form.area)

@dp.message(StateFilter(Form.area))
async def process_area(message: types.Message, state: FSMContext):
    if message.text == back_btn.text:
        return await process_rooms(message, state)
    try:
        # Заменяем запятые на точки для корректного преобразования
        area = float(message.text.replace(',', '.'))
        if area < 10 or area > 500:
            raise ValueError
    except ValueError:
        return await message.answer('Неверный формат. Введите число от 10 до 500 (например, 45.5).')
    await state.update_data(area=area)

    kb = types.ReplyKeyboardMarkup(keyboard=[[back_btn, reset_btn]], resize_keyboard=True)
    await message.answer('🏢 Введите этаж квартиры:', reply_markup=kb)
    await state.set_state(Form.floor)

@dp.message(StateFilter(Form.floor))
async def process_floor(message: types.Message, state: FSMContext):
    if message.text == back_btn.text:
        return await process_area(message, state)
    try:
        floor = int(message.text)
        if floor < 0 or floor > 100:
            raise ValueError
    except ValueError:
        return await message.answer('Этаж должен быть целым числом от 0 до 100.')

    data = await state.get_data()

    # Сводка для пользователя
    summary = (
        f"📋 <b>Проверьте введенные данные:</b>\n"
        f"• Регион: <b>{'Москва' if data['region'] == 1 else 'Московская область'}</b>\n"
        f"• Тип жилья: <b>{'Новостройка' if data['apt_type'] == 1 else 'Вторичка'}</b>\n"
        f"• Ремонт: <b>{data['renovation']}</b>\n"
        f"• Район: <b>{data['district']}</b>\n"
        f"• Станция: <b>{data.get('station', 'не указана')}</b>\n"
        f"• Минут до метро: <b>{data.get('time_display', data.get('minutes', 'не указано'))}</b>\n"
        f"• Комнат: <b>{data['rooms']}</b>\n"
        f"• Площадь: <b>{data['area']} м²</b>\n"
        f"• Этаж: <b>{floor}</b>"
    )
    await message.answer(summary, parse_mode='HTML')

    # Готовим DataFrame для модели
    try:
        required_keys = ['region', 'apt_type', 'station_freq', 'minutes', 'rooms', 'area']
        for key in required_keys:
            if key not in data:
                logger.error(f"Отсутствует ключ в данных: {key}")
                raise ValueError(f"Отсутствует ключ: {key}")
        
        df = pd.DataFrame([{
            'Region': data['region'],
            'Apartment type': data['apt_type'],
            'Metro station': data['station_freq'],
            'Minutes to metro': data['minutes'],
            'Number of rooms': data['rooms'],
            'Area': data['area'],
            'Floor': floor
        }])

        # One-hot для ремонта
        selected = reno_map.get(data['renovation'], 'No repair')
        for feat in model.feature_names_:
            if feat.startswith('reno_'):
                df[feat] = 1 if feat == f'reno_{selected}' else 0

        # Добавляем отсутствующие признаки
        for feat in model.feature_names_:
            if feat not in df.columns:
                df[feat] = 0
        df = df[model.feature_names_]

        # Масштабирование
        numerical = ['Minutes to metro', 'Number of rooms', 'Area', 'Floor']
        df_scaled = df.copy()
        df_scaled[numerical] = scaler.transform(df_scaled[numerical])
        X = df_scaled.values

        # Предсказание
        pred_rub = model.predict(X)[0]
        logger.info(f"Исходное предсказание модели: {pred_rub:.2f} рублей")
        
        # КОРРЕКТИРОВКА ЦЕН С УЧЕТОМ РЕГИОНА
        region_factor = PRICE_ADJUSTMENT_FACTOR_MOSCOW if data['region'] == 1 else PRICE_ADJUSTMENT_FACTOR_MO
        pred_rub *= region_factor
        logger.info(f"Скорректированное предсказание ({region_factor}x): {pred_rub:.2f} рублей")
        
        # Проверка на нереалистичную цену
        if pred_rub > MAX_REASONABLE_PRICE or pred_rub < MIN_REASONABLE_PRICE:
            logger.warning(f"Нереалистичное предсказание: {pred_rub:.2f} руб.")
            response = "⚠️ <b>Не удалось оценить объект</b>\n\nПроверьте введенные данные или попробуйте другие параметры."
        else:
            # Рассчитываем границы доверительного интервала
            low_rub = max(MIN_REASONABLE_PRICE, pred_rub * (1 - UNCERTAINTY))
            hi_rub = min(MAX_REASONABLE_PRICE, pred_rub * (1 + UNCERTAINTY))
            
            # Форматируем вывод
            low_millions = low_rub / 1_000_000
            hi_millions = hi_rub / 1_000_000
            
            # Определяем рекомендацию по цене
            if low_millions < 10:
                price_range = f"{low_millions:.2f}-{hi_millions:.2f} млн ₽"
            else:
                price_range = f"{low_millions:.1f}-{hi_millions:.1f} млн ₽"
            
            # Создаем информативное сообщение
            response = (
                f"💰 <b>Оценка стоимости квартиры</b>\n\n"
                f"• <b>Диапазон цены:</b> {price_range}\n"
                f"• <b>Средняя цена:</b> {pred_rub / 1_000_000:.2f} млн ₽\n\n"
                f"<i>Для точной оценки рекомендуется консультация с риелтором.</i>"
            )
                
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}", exc_info=True)
        response = '❌ <b>Ошибка оценки</b>\n\nПопробуйте другие параметры или повторите позже.'

    # Ответ пользователю
    kb = types.ReplyKeyboardMarkup(keyboard=[[reset_btn]], resize_keyboard=True)
    await message.answer(response, reply_markup=kb, parse_mode='HTML')
    await state.clear()

# Глобальный обработчик ошибок
@dp.errors()
async def errors_handler(update: types.Update, exception: Exception):
    logger.error(f"Глобальная ошибка: {exception}", exc_info=True)
    try:
        await update.message.answer("⚠️ <b>Произошла ошибка</b>\n\nПожалуйста, начните заново командой /start", 
                                   parse_mode='HTML')
    except:
        pass
    return True

if __name__ == '__main__':
    import asyncio
    async def main():
        try:
            await bot.delete_webhook(drop_pending_updates=True)
            logger.info("🚀 Бот успешно запущен!")
            logger.info(f"🔧 Коэффициент для Москвы: {PRICE_ADJUSTMENT_FACTOR_MOSCOW}")
            logger.info(f"🔧 Коэффициент для МО: {PRICE_ADJUSTMENT_FACTOR_MO}")
        except Exception as e:
            logger.error(f"Ошибка при запуске бота: {e}")
        await dp.start_polling(bot, skip_updates=True)
    asyncio.run(main())