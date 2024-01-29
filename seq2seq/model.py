import os
import torch
from transformers import AutoTokenizer
from utils import MODEL_CONFIG

def get_model(model_id, from_checkpoints = True, checkpoints_path = None, device = None):
    """
    Получение модели по названию
    :param model_id: Название модели (см. seq2seq.utils MODEL_CONFIG)
    :param from_checkpoints: Загружать ли модель из сохраненных чекпоинтов
    :param checkpoints_path: Путь к чекпоинтам
    :param device: Устройство, на котором модель обучается (cpu или cuda)
    :return: Модель
    """
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if checkpoints_path and os.path.exists(checkpoints_path) and from_checkpoints:
        if not is_file_empty(checkpoints_path):
            print(f'Loading model from checkpoints {checkpoints_path}')
            return torch.load(checkpoints_path, map_location = device)
    print(f'Loading pretrained model {MODEL_CONFIG[model_id]["pretrained_model_name"]}')
    checkpoints_path = MODEL_CONFIG[model_id]['pretrained_model_name']
    return MODEL_CONFIG[model_id]['model_class'].from_pretrained(checkpoints_path).to(device)

def get_tokenizer(model_id):
    """
    Получение токенизатора по названию модели
    :param model_id: Название модели (см. seq2seq.utils MODEL_CONFIG)
    :return:
    """
    return AutoTokenizer.from_pretrained(MODEL_CONFIG[model_id]['pretrained_model_name'])

def is_file_empty(file_path):
    return os.stat(file_path).st_size == 0