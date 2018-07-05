# krypton-geo-tagger
Search for Montevideo locations in text


### Martín Steglich . msteglichc@gmail.com
### Raúl Speroni . raulsperoni@gmail.com



## Configuración y Deploy

### BD
* Mongo (Requerido)
```
docker-compose  up -d mongo
```
* Mongo Express
```
docker-compose  up -d mongo-express
```
* Mongo Restore (Requerido por única vez)

```
docker-compose  up -d restore
```
* Mongo Backup

```
docker-compose  up -d backup
```

### Tagger
* Tagger Server

```
docker-compose up -d tagger
```

