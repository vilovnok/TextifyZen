import torch
torch.set_default_dtype(torch.float32)

# from experiments.data import SOURCE_COLUMN_NAME, TARGET_COLUMN_NAME
from data import SOURCE_COLUMN_NAME, TARGET_COLUMN_NAME


class Dataset(torch.utils.data.Dataset):
    """
    Датасет
    """
    def __init__(self, data, tokenizer, max_length=50, train=True):
        """
        :param data: Датафрейм: со сложными текстами и их упрощеннными версиями
        :param tokenizer: Токенизатор
        :param max_length: Максимальная длина текста в bpe-токенах. Остальные токены обрезаются
        :param train: Тренировочный датасет (True) или тестовый (False)
        """
        self.df = data
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.train = train

    def __getitem__(self, idx):
        input = self.df.iloc[idx][SOURCE_COLUMN_NAME]
        input_ids, attention_mask = self.tokenize(input)
        item = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if self.train:
            output = self.df.iloc[idx][TARGET_COLUMN_NAME]
            labels, decoder_attention_mask = self.tokenize(output)
            item.update({'labels': labels, 'decoder_attention_mask': decoder_attention_mask})
        return item

    def tokenize(self, text):
        inputs = self.tokenizer(
            text,
            return_attention_mask=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

    def __len__(self):
        return self.df.shape[0]