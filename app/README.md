# Simplification API
Чтобы запустить API в docker-контейнере:
1. Клонируйте репозиторий: 
```commandline
git clone https://github.com/vilovnok/TextifyZen.git
```
2. Перейдите в директорию app: 
```commandline
cd app
```
3. Выполните команды: 
``` commandline
docker compose -f docker-compose.yaml up 
```
4. После того, как в консоли появилось сообщение
```commandline
Uvicorn running on http://localhost:8000/
```
Можно обращаться к API (см. пример запроса в `test_app.py`).
Настройки параметров генерации можно редактировать в файле `generation_config.json`.