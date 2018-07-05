FROM mongo
MAINTAINER Raul Speroni (raulsperoni@gmail.com)

RUN mkdir /opt/krypton-db


COPY dump/ /opt/krypton-db/dump/

ADD mongo_backup.sh /opt/krypton-db/
RUN chmod +x /opt/krypton-db/mongo_backup.sh
ADD mongo_restore.sh /opt/krypton-db
RUN chmod +x /opt/krypton-db/mongo_restore.sh

WORKDIR /opt/krypton-db