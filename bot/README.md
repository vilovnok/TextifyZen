# Simplification API
Чтобы запустить API в docker-контейнере:
1. Клонируйте репозиторий: 
```commandline
git clone https://github.com/vilovnok/TextifyZen.git
```
2. Перейдите в директорию bot: 
```commandline
cd bot
```
3. Выполните команды: 
``` commandline
docker build -t название_образа . 
docker run test
```
4. После того, как в консоли появилось сообщение
```commandline
INFO:root:Model loaded successfully!
```
Можно обращаться к телеграм боту

Ps: Вы должны будете настроить под себя [бота](https://youtu.be/ayUBlf9pvn0?si=-xdyJJHcTxQEMTZB)