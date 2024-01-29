import pandas as pd
from tqdm.notebook import tqdm
from Levenshtein import distance

SOURCE_COLUMN_NAME = 'INPUT:source'
TARGET_COLUMN_NAME = 'OUTPUT:output'

def filter_ru_adapt(path, min_sim = 0.5, min_lev = 20, max_elong_rate = 0.5):
    """
    Фильтрация текстов из части корпуса RuAdapt
    :param path: Путь к файлу с корпусом
    :param min_sim: Минимальная близость сложного и упрощеннного предложения (близость указана в корпусе)
    :param min_lev: Минимальное расстояние между сложным и упрощеннным предложениями (пары (почти) одинаковых предложений нам не нужны)
    :param max_elong_rate: Максимальная доля удлинения упрощеннного предложения.
    В среднем упрощеннные предложения короче сложных.
    Но если они длиннее на 50% это странно и, кажется, что-то пошло не так при выравнивании корпуса.
    :return: Отфильтрованная часть корпуса
    """
    df = pd.read_csv(path)
    df = df[(df.cos_sim >= min_sim) & (df.cos_sim < 1)]
    df['lev'] = df.apply(lambda x: distance(x.source, x.target), axis=1)
    df = df[df.lev >= min_lev]
    elong = df.apply(lambda x: (len(x.target) - len(x.source))/ len(x.target), axis=1)
    df = df[elong <= max_elong_rate]
    return pd.DataFrame({SOURCE_COLUMN_NAME:df.source.str.strip('–—').str.strip(), TARGET_COLUMN_NAME:df.target.str.strip('–—').str.strip()})

def prepare_data_for_eval(data):
    """
    Подготовка данных к формату для подсчета корпусных метрик
    :param data: Датафрейм с параллельными предложениями
    :return: orig: сложные предложения
             refs: упрощеннные предложения (возможно, по нескольку на одно сложное)
    """
    refs = []
    orig = data[SOURCE_COLUMN_NAME].unique()
    for sent in tqdm(orig):
        refs.append(data.where(data[SOURCE_COLUMN_NAME] == sent)[TARGET_COLUMN_NAME].dropna())
    return list(orig), refs

def info(paths, datasets):
    """
    Вывод информации про размер частей датасета
    :param paths: Пути к файлам (для идентификации)
    :param datasets: Сами датасеты
    """
    for path, dataset in zip(paths, datasets):
        dataset_name = path.split('/')[-1]
        print(f'Name: {dataset_name}  Size: {dataset.shape[0]}')