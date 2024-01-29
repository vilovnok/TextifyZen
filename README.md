# TextifyZen(🤬➡😊Упрощение текста)

Сервис для автоматического упрощения текстов на русском языке. 

**Корпус**: [RuSimpleSentEval](https://github.com/dialogue-evaluation/RuSimpleSentEval) + [RuAdapt](https://github.com/Digital-Pushkin-Lab/RuAdapt).


**Модель:** t5 (несколько вариантов моделей). 
Код для обучения и инференса в папке `seq2seq`, эксперименты - там же, в ноутбуке `seq2seq_simplification.ipynb`.

**Метрики**: [SARI](https://aclanthology.org/Q16-1029.pdf), [BLEU](http://aclanthology.lst.uni-saarland.de/P02-1040/), [FKGL](https://www.semanticscholar.org/paper/Derivation-of-New-Readability-Formulas-%28Automated-Kincaid-Fishburne/26d5981f7da4b508961aea01d53cd60e2202ff2d) (модифицированная для русского языка). 
Чекпоинты метрик для нейросети хранятся в файле с логами (`seq2seq/train.logs`).
Библиотека, которая может помочь с подсчетом метрик [easse](https://github.com/feralvam/easse.git).

**Обертка**: cервис обернут в **телеграм-бота**, который делает запросы к API модели на huggingface ([r1char9/ruT5-base-pls](https://api-inference.huggingface.co/models/r1char9/ruT5-base-pls)). 
Код в папке `bot`.
Также в разработке расширение для Google Chrome. Подробности в папке `extension`.  
API сервиса упакован в docker-контенейнер. Подробности в `app`.


