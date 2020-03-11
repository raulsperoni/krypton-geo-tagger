#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time

import geoSearch
from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields


app = Flask(__name__)
api = Api(app, version='3.0', title='Krypton Geo Tagger',
    description='Servicio para obtener ubicaciones geográficas a partir de texto',
)

ns = api.namespace('geoTagger', description='Geo Tagger Operations')



input = api.model('Input', {
    'text': fields.String(required=True, description='Texto a analizar')
})

output = api.model('Output', {
    'name': fields.String(required=True, description='Nombre de la Ubicación'),
    'geometry': fields.Arbitrary(required=True, description='Punto representativo de la Ubicación'),
    'error': fields.Boolean(required=True, description='Indicador de error'),
    'time_ms': fields.String(required=True, description='Tiempo en ms'),
})

@ns.route('/api/find')
class GeoTagger(Resource):
    '''Encuentra una ubicación a partir del texto y devuelve un punto representativo'''
    @ns.doc('find_location')
    @ns.expect(input)
    def post(self):
        try:
            data = request.get_json(silent=True)
            text = data.get('text', None)
            best_point = None
            name_point = None
            start = time.time()
            if text:
                matches = geoSearch.complete_search(data['text'])
                if len(matches[:1]) == 1:
                    name_point, match = matches[0]
                    logger.info(name_point,match)
                    best_point = geoSearch.get_intersection_point(match)
                    logger.info(best_point)
            end = time.time()
            return jsonify({"name": name_point, "geometry": best_point, "error": False, "time_ms": '{:.3f}'.format((end - start) * 1000.0)}, 200)
        except Exception as e:
            logger.error(e)
            return jsonify({"geometry": None, "error": True}, 200)


if __name__ == '__main__':
    geoSearch = geoSearch.GeoSearch()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    app.run(debug=True, host='0.0.0.0')
