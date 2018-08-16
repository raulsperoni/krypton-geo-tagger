#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask,jsonify,request
import time,os,threading,sys,traceback
import requests
import geoEngine
import logging
from pprint import pprint, pformat



def doTheHardWork(data,id):
    with app.app_context():
        try:
            coordinates = data.get('coordinates',None)
            callback = data.get('callback',None)
            text = data.get('text',None)
            error = False
            logger.info('Working: ')
            if callback and text:
                start = time.time()
                solutions = geoEngine.process(data['text'],coordinates)
                end = time.time()
                requests.post(callback, json={"id":id,"solutions":solutions,"time":str(end-start),"error":False})
                logger.info('Done.'+str(len(solutions))+'geoTags. Demora:'+str(end-start))
        except Exception as e:
            logger.error(e)
            traceback.print_exc(file=sys.stdout)


app = Flask(__name__)

@app.route('/api/print', methods=['POST'])
def log():
    data = request.get_json(silent=True)
    logger.info('CALLBACK '+pformat(data))
    return jsonify({}),200

@app.route('/api/sync/find', methods=['POST'])
def findSync():
    data = request.get_json(silent=True)
    start = time.time()
    solutions = geoEngine.process(data['text'], data.get('coordinates',None))
    end = time.time()
    logger.info('Done.' + str(len(solutions)) + 'geoTags. Demora:' + str(end - start))
    return jsonify({"id": id, "solutions": solutions, "time": str(end - start), "error": False}),200

@app.route('/api/find/<id>', methods=['POST'])
def find(id):
    data = request.get_json(silent=True)
    thread = threading.Thread(target=doTheHardWork, args = (data,id))
    thread.daemon = True
    thread.start()
    return 'OK'

if __name__ == '__main__':
    mongostring = os.getenv('MONGO_GEO_STRING',None)
    geoEngine = geoEngine.GeoEngine(mongostring)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    app.run(debug=True,host='0.0.0.0')
