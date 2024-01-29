import os
from pathlib import Path
import json
import nltk
nltk.download('punkt')
from transformers import T5ForConditionalGeneration, AutoTokenizer



class Simplifier:
    def __init__(self):
        """
        :param model_name: Полное название модели с huggingface
        :param model_id: Короткое название модели (см. seq2seq.utils MODEL_CONFIG)
        """
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, "r") as file:
            self.config = json.load(file)
        # загружаем параметры генерации
        self.default_generation_config_path = Path(__file__).parent / "generation_config.json"
        with open(self.default_generation_config_path, "r") as file:
            self.default_generation_config = json.load(file)
            
        print(f"GenConfig {self.default_generation_config}")    
        print(f"Confug {self.config}")
            
        self.DEVICE = self.config['device']
        self.model_name = self.config['model_name']
        self.model_id = self.config['model_id']
        self.checkpoints_path = os.path.join(Path(__file__).parent, f'{self.model_id}.pt')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.max_length = lambda x: len(x.split()) * self.default_generation_config['max_length_factor']

    def tokenize(self, text):
        """
        Токенизация текста
        :param text: Текст
        :return: Токенизированный текст и маски внимания
        """
        inputs = self.tokenizer(
            text,
            return_attention_mask=True,
            return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

    def simplify_sent(self, sent):
        """
        Упрощение одного предложения
        :param sent: Предложение
        :return: Упрощеннное предложение
        """
        input_ids, attention_mask = self.tokenize(sent)
        output = self.model.generate(input_ids=input_ids.unsqueeze(0).to(self.DEVICE),
                                     attention_mask=attention_mask.unsqueeze(0).to(self.DEVICE),
                                     do_sample = self.default_generation_config['do_sample'],
                                     repetition_penalty = self.default_generation_config['repetition_penalty'],
                                     max_length = self.max_length(sent))
        return self.tokenizer.decode(output.squeeze(0), skip_special_tokens=True)

    def simplify(self, text):
        """
        Упрощение текста
        :param text: Текст
        :return: Упрощеннный текст
        """
        
        sents = nltk.sent_tokenize(text)
        simplified_sents = [self.simplify_sent(sent) for sent in sents]
        return ' '.join(simplified_sents)

if __name__ == '__main__':
    simplifier = Simplifier()
    text = '14 декабря 1944 года рабочий посёлок Ички был переименован в рабочий посёлок Советский, после чего поселковый совет стал называться Советским.'
    text = 'Rocksteady Studios задумала создать Arkham City ещё во время разработки Arkham Asylum, оставив намёки на сиквел в игре и начав полноценное создание в феврале 2009 года. «Аркхэм-Сити» в пять раз превышает площадь лечебницы Аркхэм, а его дизайн был изменён, чтобы Бэтмен мог во время планирования пикировать. На маркетинговую кампанию было потрачено более года и десяти миллионов долларов, а выпуск игры сопровождался двумя музыкальными альбомами — один содержал партитуру от Рона Фиша и Ника Арундела, а другой — одиннадцать оригинальных песен от различных популярных исполнителей. Большинство персонажей озвучены актёрами, принимавшие участие в Arkham Asylum и анимационных проектах от DC.'
    print(simplifier.simplify(text))