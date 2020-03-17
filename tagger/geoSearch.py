# -*- coding: utf-8 -*-
import configparser
import re
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from shapely.geometry import mapping, shape
from sklearn.metrics import confusion_matrix as cm
from unidecode import unidecode


class GeoSearch(object):
    """
    Clase para hacer enrich con ubicaciones a partir de texto
    """

    def __init__(self, elastic_host, elastic_port):
        config = configparser.ConfigParser()
        config.read('conf/montevideo.conf')

        self.index = config['MONTEVIDEO']['elasticsearch_index']
        self.geo_search_type = config['MONTEVIDEO']['geo_search_type']
        self.must_not_terms = config['MONTEVIDEO']['must_not_terms']
        self.boost_negative_types = config['MONTEVIDEO']['boost_negative_types']
        self.boost = float(config['MONTEVIDEO']['boost'])
        self.negative_boost = float(config['MONTEVIDEO']['negative_boost'])
        self.result_size = int(config['MONTEVIDEO']['result_size'])
        if (elastic_host and elastic_port):
            self.es = Elasticsearch([{'host': elastic_host,
                                      'port': elastic_port}])
        else:
            self.es = Elasticsearch([{'host': config['MONTEVIDEO']['elasticsearch_host'],
                                      'port': int(config['MONTEVIDEO']['elasticsearch_port'])}])

    def boosting_match_bool_search(self, text, field, size=500, boost=2, negative_boost=0.5):
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
                                "term": {"type": field}
                            },
                            "should": [
                                {
                                    "multi_match": {
                                        "query": text,
                                        "fields": ["nombre*", "aliases*"],
                                        "type": "best_fields",
                                        "tie_breaker": 0.2,
                                        "cutoff_frequency": 0.001
                                    }
                                }

                            ],
                            "boost": boost
                        }
                    },
                    "negative": {
                        "multi_match": {
                            "query": "",
                            "fields": ["nombre", "nombre_1", "aliases"],
                            "type": "best_fields",
                            "tie_breaker": 0.2
                        }
                    },
                    "negative_boost": negative_boost
                }
            },
            "highlight": {
                "fields": {
                    "*": {}
                },
                "number_of_fragments": 10
            }
        })['hits']['hits']

    def multi_match_by_field(self, text, field, size=50):
        """
        Elasticsearch query. I'm trying to boost results that doesn't come from negative_types or must_not terms.
        """
        return self.es.search(index='montevideo', body=
        {
            "from": 0, "size": size,
            "query": {
                "bool": {
                    "must": {
                        "term": {"type": field}
                    },
                    "should": [
                        {
                            "match": {
                                "nombre": {
                                    "query": text,
                                    "cutoff_frequency": 0.1
                                }
                            }
                        },
                        {
                            "match": {
                                "nombre.variant_2": {
                                    "query": text,
                                    "cutoff_frequency": 0.1
                                }
                            }
                        }, {
                            "match": {
                                "nombre_1": {
                                    "query": text,
                                    "cutoff_frequency": 0.1
                                }
                            }
                        },
                        {
                            "match": {
                                "nombre_1.variant_2": {
                                    "query": text,
                                    "cutoff_frequency": 0.1
                                }
                            }
                        }

                    ]
                }
            },
            "highlight": {
                "fields": {
                    "*": {}
                },
                "number_of_fragments": 10
            }
        })['hits']['hits']

    def multi_match(self, text, size=5):
        """
        Elasticsearch query. I'm trying to boost results that doesn't come from negative_types or must_not terms.
        """
        return self.es.search(index='montevideo', body=
        {
            "from": 0, "size": size,
            "query": {

                "multi_match": {
                    "query": text,
                    "fields": ["nombre_1*", "nombre_2.*", "aliases*"],
                }

            },
            "highlight": {
                "fields": {
                    "*": {}
                },
                "number_of_fragments": 10
            }
        })['hits']['hits']

    def multi_cruces_calles(self, text, size=50):
        """
        Elasticsearch query. I'm trying to boost results that doesn't come from negative_types or must_not terms.
        """
        return self.es.search(index='montevideo', body=
        {
            "from": 0, "size": size,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": text,
                            "fields": [
                                "text_first_street^2",
                                # "text_first_street.raw",
                                "text_first_street.variant_1^3",
                                "text_first_street.variant_2^4",
                                "text_second_street^2",
                                # "text_second_street.raw",
                                "text_second_street.variant_1^3",
                                "text_second_street.variant_2^4"
                            ],
                            "type": "cross_fields",
                            "minimum_should_match": "2",
                            "cutoff_frequency": 0.1
                        }
                    },
                    "filter": {
                        "term": {"type": 'cruces_vias'}
                    }
                }
            },
            "highlight": {
                "fields": {
                    "*": {}
                },
                "number_of_fragments": 10
            }
        })['hits']['hits']

    def multi_bool_cruces_calles(self, text, size=50):
        """
        Elasticsearch query. I'm trying to boost results that doesn't come from negative_types or must_not terms.
        """
        return self.es.search(index='montevideo', body=
        {
            "from": 0, "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text_first_street": {
                                    "query": text,
                                 #   "minimum_should_match": "30%"
                                   # "analyzer": "search_calle_analyzer"
                                }
                            },

                        },
                        {
                            "match": {
                                "text_second_street": {
                                    "query": text,
                                  #  "minimum_should_match": "30%"
                                   # "analyzer": "search_calle_analyzer"
                                }
                            },

                        }
                    ],
                    "should": {
                        "multi_match": {
                            "query": text,
                            "fields": [
                                "text_first_street.variant_1^2",
                                "text_first_street.variant_2^3",
                                "text_second_street.variant_1^2",
                                "text_second_street.variant_2^3"
                            ],
                            "type": "cross_fields",
                            #"minimum_should_match": "2",
                        }
                   #     "match_phrase": {
                   #         "text_first_street": {
                   #             "query": text,
                   #             "slop": 50
                   #         }
                   #     }
                    }
                }
            },
            "highlight": {
                "fields": {
                    "*": {}
                },
                "number_of_fragments": 10
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
    def strip_name(name, delimiter='-'):
        if name:
            name = unidecode(name)
            name = name.strip().lower()
            name = re.sub(r'[^0-9a-zA-Z]+', delimiter, name)
            return name
        else:
            return ''

    @staticmethod
    def is_name_matched(match_dict, result_object):
        matched_key = next((key for key in match_dict.keys() if
                            (result_object.get('s_nombre', None) and result_object['s_nombre'] in key) or
                            (result_object.get('s_nombre_1', None) and result_object['s_nombre_1'] in key) or
                            (result_object.get('s_aliases', None) and result_object['s_aliases'] in key)), None)
        return matched_key != None

    def get_min_substring_index(self, text, string):
        text = self.strip_name(text)
        string = self.strip_name(string)
        idx = 250
        if string in text:
            return text.index(string)
        else:
            for sub in string.split('-'):
                if sub in text:
                    idx = min(idx, text.index(sub))
        return idx

    def get_max_substring_index(self, text, string):
        text = self.strip_name(text)
        string = self.strip_name(string)
        idx = 0
        if string in text:
            return text.index(string)
        else:
            for sub in string.split('-'):
                if sub in text:
                    idx = max(idx, text.index(sub))
        return idx

    def calculate_score(self, res_obj, field_name, text):
        """
        I'm weighting how many terms matched (highlights) on each obj,
        maybe this should be optional because i'm changing elastic bias.
        Also, i'm not considering common highlights.

        ver por separado hg de nom1 nom2 y aliases
        """

        hg_main_dict = res_obj['highlights'].get(field_name, None)
        how_many_highlights = 0
        best_len_of_highlight = 0
        min_index_of_hg_in_text = 0
        max_index_of_hg_in_text = 250
        field_with_best_highlights = None
        best_relative_importance_sum = 0
        elastic_score = res_obj['score']

        if hg_main_dict:
            # print(hg_main_dict)
            for field, highlights_dict in hg_main_dict.items():
                sum_for_field = 0
                for highlight, hg_relative_importance in highlights_dict.items():
                    sum_for_field += hg_relative_importance
                best_relative_importance_sum = max(best_relative_importance_sum, sum_for_field)
                if best_relative_importance_sum == sum_for_field:
                    field_with_best_highlights = field
                    # print(field,highlight,hg_relative_importance)

            if hg_main_dict.get(field_with_best_highlights, None):
                # print(text)
                # print(hg_main_dict[field_with_best_highlights])
                how_many_highlights = max(len(hg_main_dict[field_with_best_highlights].values()), 1)
                best_len_of_highlight = max([len(hg) for hg in hg_main_dict[field_with_best_highlights].keys()])
                min_index_of_hg_in_text = min(
                    [self.get_min_substring_index(text, hg) for hg in hg_main_dict[field_with_best_highlights].keys()])
                max_index_of_hg_in_text = max(
                    [self.get_max_substring_index(text, hg) for hg in hg_main_dict[field_with_best_highlights].keys()])

        # print('how_many_highlights',how_many_highlights)
        # print('elastic_score',elastic_score)
        # print('best_len_of_highlight',best_len_of_highlight)
        # print('best_relative_importance_sum',best_relative_importance_sum)
        # print('min_index_of_hg_in_text', min_index_of_hg_in_text)
        # print('max_index_of_hg_in_text',max_index_of_hg_in_text)
        return (min_index_of_hg_in_text, max_index_of_hg_in_text, best_len_of_highlight,
                how_many_highlights * elastic_score * best_len_of_highlight ** 2 * best_relative_importance_sum)

    def make_match(self, match_type, res_obj, field, boost_field, text):
        """
        Make match object to pass around
        """
        score = 0
        distance = 250
        match_name = res_obj['s_nombre']
        if res_obj.get('nombre', None) and res_obj.get('nombre_1', None):
            # Cruce de calles
            match_name = res_obj['s_nombre'] + '#' + res_obj['s_nombre_1']

            min_idx, max_idx, len_name, name_score = self.calculate_score(res_obj, 'nombre', text)
            min_idx_1, max_idx_1, len_name_1, name_1_score = self.calculate_score(res_obj, 'nombre_1', text)

            distance = self.distance_of_strings_in_string(min_idx, max_idx, len_name, min_idx_1, max_idx_1, len_name_1)

            if name_score > 0 and name_1_score > 0:
                score = (name_score + name_1_score) * boost_field * (1 / float(distance))
            else:
                score = 0
        elif res_obj.get('nombre', None) and res_obj.get('aliases', None):
            match_name = res_obj['s_nombre'] + '&' + res_obj['s_aliases']
            min_idx, max_idx, len_name, name_score = self.calculate_score(res_obj, 'nombre', text)
            min_idx_a, max_idx_a, len_aliases, aliases_score = self.calculate_score(res_obj, 'aliases', text)
            score = (0.5 * name_score + 0.3 * aliases_score) * boost_field
        else:
            match_name = res_obj['s_nombre']
            min_idx, max_idx, len_name, name_score = self.calculate_score(res_obj, 'nombre', text)
            score = name_score * boost_field

        return {'score': score, 'name': match_name, 'objects': [res_obj], 'type': match_type,
                'field': field}

    @staticmethod
    def distance_of_strings_in_string(min_index_text_1, max_index_text1, len_text_1, min_index_text_2, max_index_text_2,
                                      len_text_2):
        if (min_index_text_1 == min_index_text_2) or (max_index_text1 == max_index_text_2):
            return 250
        return max(min(abs(min_index_text_2 - max_index_text1 + len_text_1),
                       abs(min_index_text_1 - max_index_text_2 + len_text_2)), 1)

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

    def get_result_object(self, elastic_result):
        result = {
            'id': elastic_result['_id'],
            'geo_type': elastic_result['_source']['geometry']['type'],
            'type': elastic_result['_source']['type'],
            'score': elastic_result['_score'],
            'geometry': elastic_result['_source']['geometry']
        }

        text_fields = ['text', 'text_aliases', 'text_first_street', 'text_second_street']
        text_fields_lens = {}

        for field in text_fields:
            text = self.strip_name(elastic_result['_source'].get(field, ''), ' ')
            text_fields_lens[field] = len(text)
            result[field] = text

        result_highlights = {}
        if elastic_result.get('highlight', None):
            for key, hg_values in elastic_result['highlight'].items():
                field = key.split('.')[0]
                variant = 'normal'
                if len(key.split('.')) > 1:
                    variant = key.split('.')[1]
                if field in text_fields:
                    if not result_highlights.get(field, None):
                        result_highlights[field] = {}
                    for hg in re.findall("<em>(.*?)<\/em>", ' '.join(hg_values)):
                        hg_relative_importance = len(hg) / float(text_fields_lens[field])
                        result_highlights[field][hg] = (hg_relative_importance, variant)

        result['highlights'] = result_highlights
        return result

    @staticmethod
    def print_object(res_object):
        if res_object.get('type') == 'cruces_vias':
            hgs = ''
            for main_key, hg_main_field in res_object['highlights'].items():
                hgs += '--> {: >20}\n'.format(main_key)
                for key, pair in hg_main_field.items():
                    hg, value = pair
                    hgs += '{: >20} {: >10} => {:.3f}\n'.format(value, key, hg)
            return "{} # {} ## {} \n {}".format(res_object['text_first_street'], res_object['text_second_street'],
                                                res_object['score'], hgs)
        else:
            return "$$$$$$"

    def complete_search(self, text, result_size=500):
        match_dict = {}

        search_fields = [("cruces_vias", 100), ("lugares_interes", 1), ("espacios_libres", 1), ("geonames", 1),
                         ("limites_barrios", 1)]

        for search_field, boost_field in search_fields:
            results = self.boosting_match_bool_search(text, search_field, result_size)
            for result in results:
                try:
                    result_object = self.get_result_object(result)

                    if self.is_name_matched(match_dict, result_object):
                        # If there is alredy a match that involves this name continue
                        continue
                    match_obj = self.make_match(result_object['geo_type'], result_object, search_field, boost_field,
                                                self.strip_name(text))
                    match_dict[match_obj['name']] = match_obj
                except Exception as e:
                    print(e)

        return sorted(match_dict.items(), key=lambda i: i[1]['score'], reverse=True)

    def test_kit(self, csv_file_name, line_score_limit=300, point_score_limit=100):
        test_set = pd.read_csv(csv_file_name, index_col=0)
        mejorMatch = []
        scoreMejorMatch = []
        matchType = []
        times = []
        for index, row in test_set.iterrows():
            time1 = time.time()
            matches = self.complete_search(row['texto'], 10)
            time2 = time.time()
            times.append(float('{:.3f}'.format((time2 - time1) * 1000.0)))
            if any(matches):
                mejorMatch.append(matches[0][0])
                scoreMejorMatch.append(float(matches[0][1]['score']))
                matchType.append(matches[0][1]['field'])
            else:
                mejorMatch.append('')
                scoreMejorMatch.append(0)
                matchType.append('')
        test_set['mejorMatch'] = mejorMatch
        test_set['scoreMejorMatch'] = scoreMejorMatch
        test_set['matchType'] = matchType
        test_set['time_ms'] = times
        test_set['encontreUbicacionCalle'] = np.where(
            (test_set['scoreMejorMatch'] > line_score_limit) & (test_set['matchType'] == 'cruces_vias'), True, False)
        test_set['encontreUbicacion'] = np.where(test_set['encontreUbicacionCalle'] | (
                (test_set['scoreMejorMatch'] > point_score_limit) & (test_set['matchType'] != 'cruces_vias')), True,
                                                 False)
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
            matches = self.complete_search(text, self.result_size)
            if len(matches) >= 1:
                name_point, match = matches[0]
                score_point = match['score']
                if ((match['type'] == 'LINES') and (score_point > line_score_limit)) or \
                        ((match['type'] in ['POINT', 'POLYGON']) and (score_point > point_score_limit)):
                    best_point = self.get_intersection_point(match)
                    return name_point, score_point, best_point
        return None, 0, None
