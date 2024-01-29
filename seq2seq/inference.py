import torch
from utils import BATCH_SIZE
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

def batch_inference(texts, model, tokenizer, max_new_tokens = 50, batch_size = BATCH_SIZE, **kwargs):
    """
    Инференс модели на нескольких текстах
    :param texts: Список текстов
    :param model: Модель
    :param tokenizer: Токенизатор
    :param max_new_tokens: Параметр max_new_tokens метода generate
    :param batch_size: Размер батча
    :return: Список упрощеннных текстов
    """
    generated_texts = []
    loader = DataLoader(texts, batch_size = batch_size)
    for batch in tqdm(loader):
        input_ids, attention_mask = tokenizer(batch, return_tensors = 'pt', padding=True).values()
        with torch.no_grad():
            outputs = model.generate(input_ids = input_ids.to(model.device), attention_mask = attention_mask.to(model.device), max_new_tokens = max_new_tokens, **kwargs)
        generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens = True, **kwargs)
        generated_texts.extend(generated_batch)
    return generated_texts

def example(source, taget, model, tokenizer):
    """
    Пример упрощения текста моделью
    :param source: Сложный текст
    :param taget: Упрощенный текст из датасета
    :param model: Модель
    :param tokenizer: Токенизатор
    :return: Текст, упрощенный моделью
    """
    print(f'SOURCE: {source}\nTARGET: {taget}')
    input_ids, attention_mask = tokenizer(source, return_tensors = 'pt').values()
    with torch.no_grad():
        output = model.generate(input_ids = input_ids.to(model.device), attention_mask = attention_mask.to(model.device), 
                                max_new_tokens = input_ids.size(1)*2, min_length=0)
    return tokenizer.decode(output.squeeze(0), skip_special_tokens = True)