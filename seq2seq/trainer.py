import os
import time
import random
from tqdm.notebook import tqdm, trange
import torch
from torch.optim import Adam


from utils import cleanup
from inference import batch_inference, example
from evaluation import compute_corpus_metrics
from utils import stringify_dict

def train(model, tokenizer, train_loader, val_loader, model_id = 'rut5-base', dataset_name = 'senteval+ruadapt', max_epochs=30, cleanup_step=1, window=50, lr=3e-5, save_checkpoints=True,
          checkpoints_path=None, logger = None, orig = None, refs = None, save_epoch = 5, report_epoch = 5):
    """
    Обучение модели
    :param model:
    :param tokenizer:
    :param train_loader:
    :param val_loader:
    :param model_id: Название модели (см. seq2seq.utils MODEL_CONFIG) (для идентификации экспериментов и чекпоинтов)
    :param dataset_name: Название датасета (для идентификации экспериментов и чекпоинтов)
    :param max_epochs: Количество эпох обучения
    :param cleanup_step: Количество шагов, после которого выполняется отчистка памяти
    :param window: Размер окна скользящего среднего для подсчета лосса
    :param lr: Шаг обучения
    :param save_checkpoints: Сохранять ли чекпоинты модели
    :param checkpoints_path: Путь к чекпоинтам модели
    :param logger:
    :param orig: Сложные тексты (для подсчета метрик)
    :param refs: Упрощенные тексты из датасета (для подсчета метрик)
    :param save_epoch: Количество эпох, после которых сохраняются чекпоинты модели
    :param report_epoch: Количество эпох, после которого считаются корпусные метрики (bleu, sari, etc)
    """
    if save_checkpoints and not checkpoints_path:
        print('Path to checkpoints not provided')
    cleanup()
    optimizer = Adam(params=[p for p in model.parameters() if p.requires_grad], lr=lr)
    ewm_loss = 0
    model.train()
    for epoch in trange(1, max_epochs+1):
        tq = tqdm(train_loader)
        for step, batch in enumerate(tq):
            try:
                batch['labels'][batch['labels'] == 0] = -100
                loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
                loss.backward()
            except Exception as e:
                print(f'Error on step {step}: {e}')
                loss = None
                cleanup()
                continue
            optimizer.step()
            optimizer.zero_grad()

            if step % cleanup_step == 0:
                cleanup()

            w = 1 / min(step + 1, window)
            ewm_loss = ewm_loss * (1 - w) + loss.item() * w
            tq.set_description(f'Train loss: {ewm_loss:4.4f}')

        model.eval()
        eval_loss = evaluate_model(model, val_loader)
        model.train()
        print(f'Epoch {epoch}, step {step}: train loss: {ewm_loss:4.4f}  val loss: {eval_loss:4.4f}')
        if orig:
            idx = random.randint(0, len(orig))
            print(f'PREDICTION: {example(orig[idx], list(refs[idx])[0], model = model, tokenizer = tokenizer)}')

        if epoch % save_epoch == 0 and save_checkpoints and checkpoints_path:
            torch.save(model, os.path.join(checkpoints_path))
            print(f'Model saved to {checkpoints_path}')
        if epoch % report_epoch == 0 and orig and refs:
            print('Computing sari and bleu')
            simplification_func = lambda texts: batch_inference(texts, model = model, tokenizer = tokenizer)
            metrics, quality_estimation = compute_corpus_metrics(orig = orig, refs = refs, simplification_func = simplification_func, compute_quality_estimation = False)
            if logger:
                logger.info(f' model:{model_id} dataset: {dataset_name} {stringify_dict(metrics)} time:{time.asctime()}')
            else:
                print(stringify_dict(metrics))

def evaluate_model(model, loader):
    loss_accum = 0
    num_samples = 0
    for batch in loader:
        with torch.no_grad():
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            loss_accum += len(batch) * loss.item()
            num_samples += len(batch)
    avg_loss = loss_accum / num_samples
    return avg_loss