# coding: utf-8
import math
import time
import configparser
from difflib import SequenceMatcher
import json
from shapely.geometry import mapping, shape

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as cm

from elasticsearch import Elasticsearch


class GeoSearch(object):
    """
    Clase para hacer enrich con ubicaciones a partir de texto
    """

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('conf/montevideo.conf')

        self.index = config['MONTEVIDEO']['elasticsearch_index']
        self.geo_search_type = config['MONTEVIDEO']['geo_search_type']
        self.must_not_terms = config['MONTEVIDEO']['must_not_terms']
        self.boost_negative_types = config['MONTEVIDEO']['boost_negative_types']
        self.boost = float(config['MONTEVIDEO']['boost'])
        self.negative_boost = float(config['MONTEVIDEO']['negative_boost'])
        self.result_size = int(config['MONTEVIDEO']['result_size'])
        self.es = Elasticsearch([{'host': config['MONTEVIDEO']['elasticsearch_host'], 'port': int(config['MONTEVIDEO']['elasticsearch_port'])}])

    def boosting_match_bool_search(self, text, size=500, boost=2, negative_boost=0.5):
        """
        Elasticsearch query. I'm trying to boost results that doesn't come from negative_types or must_not terms.
        """
        return self.es.search(index=self.index, body=
        {
            "from": 0, "size": size,
            "query": {
                "boosting": {
                    "positive": {
                        "bool": {
                            "must": {
                                "multi_match": {
                                    "query": text,
                                    "analyzer": "calle_analyzer",
                                    "fields": ["nombre", "aliases"],
                                    "type": "best_fields"

                                }
                            },
                            "must_not": {
                                "multi_match": {
                                    "query": self.must_not_terms,
                                    "analyzer": "calle_analyzer",
                                    "fields": ["nombre", "aliases"],
                                    "type": "best_fields"
                                }
                            },
                            "boost": boost
                        }
                    },
                    "negative": {
                        "match": {
                            "type": self.boost_negative_types
                        }
                    },
                    "negative_boost": negative_boost
                }
            },
            "highlight": {
                "fields": {
                    "nombre": {},
                    "aliases": {}
                }
            }
        })['hits']['hits']

    def search_geo_vias(self, id, size=10):
        """
        I'm trying to get size documents intersecting with document id.
        """
        return self.es.search(index=self.index, body=
        {
            "from": 0, "size": size,
            "query": {
                "bool": {
                    "must": {
                        "match": {
                            "type": self.geo_search_type
                        }
                    },
                    "filter": {
                        "geo_shape": {
                            "geometry": {
                                "indexed_shape": {
                                    "index": self.index,
                                    "id": id,
                                    "path": "geometry"
                                }
                            }
                        }
                    }
                }
            }
        })['hits']['hits']

    @staticmethod
    def strip_name(name):
        return name.replace(' ', '-').strip().lower()

    @staticmethod
    def get_match_name(result_obj_1, result_obj_2):
        return result_obj_1['s_name'] + '|' + result_obj_2['s_name']

    @staticmethod
    def is_name_matched(match_dict, result_object):
        matched_key = next((key for key in match_dict.keys() if result_object['s_name'] in key), None)
        return matched_key != None

    @staticmethod
    def calculate_score(res_obj, minus_highlights=[]):
        """
        I'm weighting how many terms matched (highlights) on each obj,
        maybe this should be optional because i'm changing elastic bias.
        Also, i'm not considering common highlights.
        """
        highlights = set(res_obj['highlights']) - set(minus_highlights)
        how_many_highlights = max(len(highlights), 1)
        avg_highlights_lenght = max(sum([len(hg) for hg in highlights]) / how_many_highlights, 3)
        elastic_score = res_obj['score']
        return how_many_highlights * math.log(avg_highlights_lenght) * elastic_score

    def make_match(self, match_type, res_obj, geo_obj=None):
        """
        Make match object to pass around
        """
        if geo_obj:
            common_highlights = set(res_obj['highlights']).intersection(geo_obj['highlights'])
            return {'score': self.calculate_score(res_obj) * self.calculate_score(geo_obj, common_highlights),
                    'objects': [res_obj, geo_obj],
                    'type': match_type}
        else:
            return {'score': self.calculate_score(res_obj), 'objects': [res_obj], 'type': match_type}

    @staticmethod
    def exists_common_significant_string(text1, text2, limit=5):
        match = SequenceMatcher(None, text1, text2).find_longest_match(0, len(text1), 0, len(text2))
        return match.size > limit

    @staticmethod
    def have_same_highlights(obj1, obj2):
        intersection = set(obj1['highlights']).intersection(obj2['highlights'])
        return len(list(intersection)) == len(obj1['highlights']) or len(list(intersection)) == len(obj2['highlights'])

    def try_to_match_lines(self, match_dict, line_results, all_results, favor_more_matches=True):
        if not favor_more_matches:
            results = line_results.values()
        else:
            results = sorted(line_results.values(), key=lambda i: len(i['highlights']), reverse=True)
        for res_obj in results:
            if self.is_name_matched(match_dict, res_obj):
                # If there is alredy a match that involves this name continue
                continue
            for geo_result in self.search_geo_vias(res_obj['id']):
                # I only care for those objects intersecting current object AND where part of the original results.
                geo_obj = all_results.get(geo_result['_id'], None)
                if geo_obj and res_obj['s_name'] != geo_obj['s_name'] and not self.have_same_highlights(res_obj,
                                                                                                        geo_obj) and not self.exists_common_significant_string(
                    res_obj['s_name'], geo_obj['s_name']):
                    # I only need to care for different objects, matching names and not ids.
                    # In case of streets more than one block can intersect with the next one, same name diff id.
                    # I discard obj with same highlights or common significant substrings
                    match_name = self.get_match_name(res_obj, geo_obj)
                    match_obj = self.make_match('LINES', res_obj, geo_obj)
                    match_dict[match_name] = match_obj

    def try_to_match_points(self, match_dict, point_results):
        for key, res_obj in point_results.items():
            if self.is_name_matched(match_dict, res_obj):
                # If there is alredy a match that involves this name continue
                continue
            match_obj = self.make_match('POINT', res_obj)
            match_dict[res_obj['s_name']] = match_obj

    def try_to_match_polygons(self, match_dict, polygon_results):
        for key, res_obj in polygon_results.items():
            if self.is_name_matched(match_dict, res_obj):
                # If there is alredy a match that involves this name continue
                continue
            match_obj = self.make_match('POLYGON', res_obj)
            match_dict[res_obj['s_name']] = match_obj

    def get_result_object(self, result):
        result_geo_type = result['_source']['geometry']['type']
        result_id = result['_id']
        result_score = result['_score']
        result_geometry = result['_source']['geometry']
        result_name = 'NO_NAME'
        if result['_source'].get('nombre', None):
            result_name = result['_source']['nombre'].encode('ascii', 'ignore').decode('ascii')
        elif result['_source'].get('aliases', None):
            result_name = result['_source']['aliases'].encode('ascii', 'ignore').decode('ascii')
        result_striped_name = self.strip_name(result_name)
        result_highlights = []
        highlights = result.get('highlight', None)
        if highlights:
            for a in highlights.values():
                for b in a:
                    for c in b.split(' '):
                        if '<em>' in c:
                            hg = c.replace('<em>', '').replace('</em>', '')
                            for hg_one in hg.split(','):
                                if not hg_one in result_highlights:
                                    result_highlights.append(hg_one)
        return {'id': result_id, 'geo_type': result_geo_type, 'name': result_name, 's_name': result_striped_name,
                'score': result_score, 'geometry': result_geometry, 'highlights': result_highlights}

    def complete_search(self, text):
        match_dict = {}

        line_results = {}
        point_results = {}
        polygon_results = {}
        all_results = {}

        results = self.boosting_match_bool_search(text, self.result_size, self.boost, self.negative_boost)
        for result in results:
            result_object = self.get_result_object(result)
            if result_object['geo_type'] == 'LineString':
                line_results[result_object['id']] = result_object
            elif result_object['geo_type'] == 'Point':
                point_results[result_object['id']] = result_object
            elif result_object['geo_type'] == 'Polygon':
                polygon_results[result_object['id']] = result_object
            else:
                print('Do i have other?')

        all_results.update(line_results)
        all_results.update(point_results)
        all_results.update(polygon_results)

        self.try_to_match_lines(match_dict, line_results, all_results)
        self.try_to_match_points(match_dict, point_results)
        self.try_to_match_polygons(match_dict, polygon_results)

        return sorted(match_dict.items(), key=lambda i: i[1]['score'], reverse=True)

    def test_kit(self, csv_file_name, line_score_limit=300, point_score_limit=100):
        test_set = pd.read_csv(csv_file_name, index_col=0)
        mejorMatch = []
        scoreMejorMatch = []
        matchType = []
        times = []
        for index, row in test_set.iterrows():
            time1 = time.time()
            results, matches = self.complete_search(row['texto'])
            time2 = time.time()
            times.append(float('{:.3f}'.format((time2 - time1) * 1000.0)))
            if any(matches):
                mejorMatch.append(matches[0][0])
                scoreMejorMatch.append(float(matches[0][1]['score']))
                matchType.append(matches[0][1]['type'])
            else:
                mejorMatch.append('')
                scoreMejorMatch.append(0)
                matchType.append('')
        test_set['mejorMatch'] = mejorMatch
        test_set['scoreMejorMatch'] = scoreMejorMatch
        test_set['matchType'] = matchType
        test_set['time_ms'] = times
        test_set['encontreUbicacionCalle'] = np.where(
            (test_set['scoreMejorMatch'] > line_score_limit) & (test_set['matchType'] == 'LINES'), True, False)
        test_set['encontreUbicacion'] = np.where(test_set['encontreUbicacionCalle'] | (
                (test_set['scoreMejorMatch'] > point_score_limit) & (
                (test_set['matchType'] == 'POINT') | (test_set['matchType'] == 'POLYGON'))), True, False)
        tn, fp, fn, tp = cm(test_set['tieneUbicacion'], test_set['encontreUbicacion']).ravel()
        file_name = 'LINE_LIM:{}_POINT_LIM:{}_TN:{}_FP:{}_FN:{}_TP:{}-{}'.format(line_score_limit, point_score_limit,
                                                                                 tn,
                                                                                 fp, fn, tp, csv_file_name)
        test_set.to_csv(file_name)
        return test_set

    @staticmethod
    def get_intersection_point(match):
        if match and len(match['objects']) == 1:
            shape_1 = shape(match['objects'][0]['geometry'])
            return mapping(shape_1.representative_point())
        elif match and len(match['objects']) > 1:
            shape_1 = shape(match['objects'][0]['geometry'])
            shape_2 = shape(match['objects'][1]['geometry'])
            return mapping(shape_1.intersection(shape_2).representative_point())
        else:
            return None

    def get_result(self, text, line_score_limit=300, point_score_limit=100):
        if text:
            matches = self.complete_search(text)
            if len(matches[:1]) == 1:
                name_point, match = matches[0]
                score_point = match['score']
                if ((match['type'] == 'LINES') and (score_point > line_score_limit)) or \
                        ((match['type'] in ['POINT','POLYGON']) and (score_point > point_score_limit)):
                    best_point = self.get_intersection_point(match)
                    return name_point, score_point, best_point
        return None, 0, None


