import os
import gc
import torch
import logging
import importlib
import numpy as np
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, BartForConditionalGeneration

BATCH_SIZE = 32
RANDOM_STATE = 42

MODEL_CONFIG = {
    'rut5-base':{'pretrained_model_name':'sberbank-ai/ruT5-base', 'model_class':T5ForConditionalGeneration},
    'mt5-small':{'pretrained_model_name':'google/mt5-small', 'model_class':MT5ForConditionalGeneration},
    'paraphraser': {'pretrained_model_name':'cointegrated/rut5-base-paraphraser', 'model_class':T5ForConditionalGeneration},
    'bart': {'pretrained_model_name':'sn4kebyt3/ru-bart-large', 'model_class':BartForConditionalGeneration},
    'flan-t5':{'pretrained_model_name':'Yasbok/Flan-t5-fine-tune-PEFT-Lora', 'model_class':T5ForConditionalGeneration},
    'rut5-absum':{'pretrained_model_name':'cointegrated/rut5-base-absum', 'model_class':T5ForConditionalGeneration},
    'rut5-multitask':{'pretrained_model_name':'cointegrated/rut5-base-multitask', 'model_class':T5ForConditionalGeneration}
}

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def sent_length_info(sents, tokenizer):
    sent_lengths = [len(tokenizer(sent)[0]) for sent in sents]
    print(f'Average text length: {np.mean(sent_lengths)}')
    print(f'Median text length: {np.median(sent_lengths)}')
    print(f'Max text length: {max(sent_lengths)}')

def add_prefix(text):
    return f'simplify | {text}'

def add_prefixes(texts):
    return [add_prefix(text) for text in texts]




def set_logging():
    """
    Настройка логирования
    :return:
    """
    file_path='train.logins'
    if os.path.exists(file_path):
        logger = logging.getLogger('results')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fhl = logging.FileHandler(file_path)
        logger.addHandler(fhl)
    else:
        with open(file_path, 'w') as file:
            logger = logging.getLogger('results')
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            fhl = logging.FileHandler(file_path)
            logger.addHandler(fhl)                
            pass
    return logger


def reload(module_path):
    """
    Перезагрузка модуля
    :param module_path: Путь к модулю, разделенный точками (например, 'experiments.utils')
    """
    module = importlib.import_module(module_path)
    importlib.reload(module)
    print(f'{module} reloaded successfully')


def stringify_dict(d):
    return ' '.join(f'{k}:{v}' for k, v in d.items())
