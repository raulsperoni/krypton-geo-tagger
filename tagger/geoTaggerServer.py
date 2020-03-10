#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time

import geoSearch
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/api/find', methods=['POST'])
def find():
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
        return jsonify(
            {"name": name_point, "geometry": best_point, "error": False, "time_ms": '{:.3f}'.format((end - start) * 1000.0)}), 200
    except Exception as e:
        logger.error(e)
        return jsonify({"geometry": None, "error": True}), 200


if __name__ == '__main__':
    geoSearch = geoSearch.GeoSearch()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    app.run(debug=True, host='0.0.0.0')
