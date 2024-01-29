DC = docker compose 
APP_FILE = docker-compose.yaml

.PHONY: app 
app:
	${DC} -f ${APP_FILE} up -d 

.PHONY: drop
drop:
	${DC} -f ${APP_FILE} down 
	
.PHONY: logs
logs: 
	${DC} -f ${APP_FILE} logs -f 