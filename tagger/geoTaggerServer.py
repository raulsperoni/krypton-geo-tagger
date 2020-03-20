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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
geoSearch = geoSearch.GeoSearch()


@ns.route('/api/find')
class GeoTagger(Resource):
    '''Encuentra una ubicación a partir del texto y devuelve un punto representativo'''

    @ns.doc('find_location')
    @ns.expect(input)
    def post(self):
        try:
            data = request.get_json(silent=True)
            text = data.get('text', None)
            start = time.time()
            name_point, score_point, best_point, match_field = geoSearch.get_result(text)
            logger.info('{} -> {:.3f}'.format(name_point, score_point))
            end = time.time()
            return jsonify(
                name=name_point,
                geometry=best_point,
                field=match_field,
                error=False,
                score='{:.3f}'.format(score_point),
                time_ms='{:.3f}'.format((end - start) * 1000.0))
        except Exception as e:
            logger.error(e)
            return jsonify(geometry=None, error=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
