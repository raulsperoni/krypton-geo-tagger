# -*- coding: utf-8 -*-
import configparser
import logging
import re
import time

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from shapely.geometry import mapping, shape
from sklearn.metrics import confusion_matrix as cm
from unidecode import unidecode

logging.basicConfig()
logger = logging.getLogger('GeoSearch')


class GeoSearch(object):
    """
    Clase para hacer enrich con ubicaciones a partir de texto
    """

    def __init__(self, elastic_host, elastic_port):
        config = configparser.ConfigParser()
        config.read('conf/montevideo.conf')
        logger.setLevel(logging.DEBUG)
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

    def search_adversary_fields(self, text, field, size=50):
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
                                }
                            },

                        },
                        {
                            "match": {
                                "text_second_street": {
                                    "query": text,
                                }
                            },

                        }
                    ],
                    "filter": {
                        "term": {"type": field}
                    },
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
                        }
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

    def search_complementary_fields(self, text, field, size=50):
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
                            "fields": ["text", "text_aliases"],
                            "type": "best_fields",
                            "cutoff_frequency": 1,
                            "fuzziness": "1",
                        }
                    },
                    "filter": {
                        "term": {"type": field}
                    },
                    "should": {
                        "multi_match": {
                            "query": text,
                            "fields": [
#                                "text.variant_1^2",
                                "text.variant_2^3",
#                                "text_aliases.variant_1^2",
                                "text_aliases.variant_2^3"
                            ],
                            "type": "best_fields",
                            "cutoff_frequency": 1,
                            "fuzziness": "1",
                        }
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

    def search_single_field(self, text, field, size=50):
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
                            "fields": ["text"],
                            "cutoff_frequency": 1,
                            "fuzziness": "1",
                        }
                    },
                    "filter": {
                        "term": {"type": field}
                    },
                    "should": {
                        "multi_match": {
                            "query": text,
                            "fields": [
#                                "text.variant_1^2",
                                "text.variant_2^3",
                            ],
                            "type": "best_fields",
                            "cutoff_frequency": 1,
                            "fuzziness": "1",
                        }
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

        field_highlights = res_obj['highlights'].get(field_name, None)
        elastic_score = res_obj['score']

        if field_highlights:
            best_highlights = {}
            for highlight, variant_tuple in field_highlights.items():
                hg_relative_importance, variant = variant_tuple
                stored_relative_importance = best_highlights.get(highlight, 0)
                if hg_relative_importance > stored_relative_importance:
                    best_highlights[highlight] = hg_relative_importance

            unique_highlights_count = len(best_highlights.keys())
            relative_importance = max(val for val in best_highlights.values())
            len_longest_highlight = max([len(key) for key in best_highlights.keys()])
            min_position_in_text = min([self.get_min_substring_index(text, hg) for hg in best_highlights.keys()])
            max_position_in_text = max([self.get_min_substring_index(text, hg) for hg in best_highlights.keys()])

            logger.debug(
                'FIELD:{}\nmin_idx = {}\nmax_idx = {}\nlen_longest = {}\nhighlights_count = {}\nimportance = {}'.format(
                    field_name,
                    min_position_in_text,
                    max_position_in_text,
                    len_longest_highlight,
                    unique_highlights_count,
                    relative_importance))

            return (
                best_highlights,
                min_position_in_text,
                max_position_in_text,
                len_longest_highlight,
                unique_highlights_count * elastic_score * len_longest_highlight ** 2 * relative_importance)

        return None, 0, 0, 0, 0

    def exists_match_complementary_fields(self, match_dict, result_object):

        first = self.strip_name(result_object.get('text', None))
        secnd = self.strip_name(result_object.get('text_aliases', None))

        for existing_match_name in match_dict.keys():
            if '&' in existing_match_name:
                parts = existing_match_name.split('&')
                if ((first == parts[0]) and (secnd == parts[1])) or ((first == parts[1]) and (secnd == parts[0])):
                    return True
        return False

    def exists_match_single_field(self, match_dict, result_object):

        first = self.strip_name(result_object.get('text', None))

        for existing_match_name in match_dict.keys():
            if first == existing_match_name:
                return True
        return False

    def exists_match_adversary_fields(self, match_dict, result_object):

        first = self.strip_name(result_object.get('text_first_street', None))
        secnd = self.strip_name(result_object.get('text_second_street', None))

        for existing_match_name in match_dict.keys():
            parts = existing_match_name.split('#')
            if ((first == parts[0]) and (secnd == parts[1])) or ((first == parts[1]) and (secnd == parts[0])):
                return True

        return False

    def make_match_adversary_fields(self, res_obj, field, boost_field, text):
        """
        Make match object to pass around
        """
        result = {
            'score': 0,
            'name': None,
            'objects': [res_obj],
            'field': field,
            'type': res_obj['geo_type']
        }

        first_field = 'text_first_street'
        secnd_field = 'text_second_street'

        if res_obj.get(first_field, None) and res_obj.get(secnd_field, None):
            result['name'] = self.strip_name(res_obj[first_field], '-') + '#' + self.strip_name(res_obj[secnd_field],
                                                                                                '-')

            first_hgs, first_min_idx, first_max_idx, first_len, first_score = self.calculate_score(res_obj, first_field,
                                                                                                   text)
            secnd_hgs, secnd_min_idx, secnd_max_idx, secnd_len, secnd_score = self.calculate_score(res_obj, secnd_field,
                                                                                                   text)

            exclusive_highlights_first = set(first_hgs.keys()).difference(set(secnd_hgs.keys()))
            exclusive_highlights_secnd = set(secnd_hgs.keys()).difference(set(first_hgs.keys()))
            len_exclusive_first = len(exclusive_highlights_first)
            len_exclusive_secnd = len(exclusive_highlights_secnd)
            logger.debug('len_exclusive_first = {}\nlen_exclusive_secnd = {}'.format(
                len_exclusive_first,
                len_exclusive_secnd
            ))
            if (len_exclusive_first == 0) or (len_exclusive_secnd == 0):
                return result

            distance = self.distance_of_strings_in_string(
                first_min_idx, first_max_idx, first_len, secnd_min_idx, secnd_max_idx, secnd_len)

            if first_score > 0 and secnd_score > 0:
                result['score'] = (first_score + secnd_score) * boost_field * (1 / float(distance)) * (
                        len_exclusive_first + len_exclusive_secnd)

            logger.debug('TOTAL = {}\nFirst Score = {}\nSecond Score = {}\nBoost Field = {}\nDistance = {}'.format(
                result['score'], first_score, secnd_score, boost_field, distance
            ))

        return result

    def make_match_complmentary_fields(self, res_obj, field, boost_field, text):
        """
        Make match object to pass around
        """
        result = {
            'score': 0,
            'name': None,
            'objects': [res_obj],
            'field': field,
            'type': res_obj['geo_type']
        }

        first_field = 'text'
        secnd_field = 'text_aliases'

        if res_obj.get(first_field, None) and res_obj.get(secnd_field, None):
            result['name'] = self.strip_name(res_obj[first_field], '-') + '&' + self.strip_name(res_obj[secnd_field],
                                                                                                '-')

            first_hgs, first_min_idx, first_max_idx, first_len, first_score = self.calculate_score(res_obj, first_field,
                                                                                                   text)
            secnd_hgs, secnd_min_idx, secnd_max_idx, secnd_len, secnd_score = self.calculate_score(res_obj, secnd_field,
                                                                                                   text)

            if first_score > 0 or secnd_score > 0:
                result['score'] = max(first_score, secnd_score) * boost_field

            logger.debug('TOTAL = {}\nFirst Score = {}\nSecond Score = {}\nBoost Field = {}'.format(
                result['score'], first_score, secnd_score, boost_field
            ))

        return result

    def make_match_single_field(self, res_obj, field, boost_field, text):
        """
        Make match object to pass around
        """
        result = {
            'score': 0,
            'name': None,
            'objects': [res_obj],
            'field': field,
            'type': res_obj['geo_type']
        }

        first_field = 'text'

        if res_obj.get(first_field, None):
            result['name'] = self.strip_name(res_obj[first_field], '-')

            first_hgs, first_min_idx, first_max_idx, first_len, first_score = self.calculate_score(res_obj, first_field,
                                                                                                   text)

            if first_score > 0:
                result['score'] = first_score * boost_field

            logger.debug('TOTAL = {}\nFirst Score = {}\nBoost Field = {}'.format(
                result['score'], first_score, boost_field
            ))

        return result

    @staticmethod
    def distance_of_strings_in_string(min_index_text_1, max_index_text1, len_text_1, min_index_text_2, max_index_text_2,
                                      len_text_2):
        if (min_index_text_1 == min_index_text_2) or (max_index_text1 == max_index_text_2):
            return 250
        return max(min(abs(min_index_text_2 - max_index_text1 + len_text_1),
                       abs(min_index_text_1 - max_index_text_2 + len_text_2)), 1)

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

    def print_object(self, res_object):
        hgs = ''
        for main_key, hg_main_field in res_object['highlights'].items():
            hgs += '--> {: >20}\n'.format(main_key)
            for key, pair in hg_main_field.items():
                hg, value = pair
                hgs += '{: >20} {: >10} => {:.3f}\n'.format(value, self.strip_name(key, ' '), hg)
        if res_object.get('type') == 'cruces_vias':
            return "{} # {} ## {} \n {}".format(res_object['text_first_street'], res_object['text_second_street'],
                                                res_object['score'], hgs)
        elif res_object.get('type') in ['geonames', 'lugares_interes']:
            return "{} & {} ## {} \n {}".format(res_object['text'], res_object['text_aliases'],
                                                res_object['score'], hgs)
        else:
            return '$$$'

    @staticmethod
    def log_match(match):
        logger.info('✔️ MATCH: {: >15}{: >20}{: >50}'.format(match['score'], match['field'], match['name']))

    def complete_search(self, text, result_size=500):
        #text = self.strip_name(text, ' ')
        match_dict = {}

        datasets_with_adversary_fields = [("cruces_vias", 100)]
        datasets_with_complementary_fields = [("lugares_interes", 1), ("geonames", 1)]
        datasets_with_single_field = [("vias", 1), ("limites_barrios", 10)]

        for search_field, boost_field in datasets_with_adversary_fields:
            try:
                results = self.search_adversary_fields(text, search_field, result_size)
                for result in results:
                    result_object = self.get_result_object(result)
                    if not self.exists_match_adversary_fields(match_dict, result_object):
                        logger.debug(self.print_object(result_object))
                        match_obj = self.make_match_adversary_fields(result_object, search_field, boost_field, text)
                        if match_obj['score'] > 0:
                            match_dict[match_obj['name']] = match_obj
            except Exception as e:
                logger.error(e, exc_info=True)

        for search_field, boost_field in datasets_with_complementary_fields:
            try:
                results = self.search_complementary_fields(text, search_field, result_size)
                for result in results:
                    result_object = self.get_result_object(result)
                    if not self.exists_match_complementary_fields(match_dict, result_object):
                        logger.debug(self.print_object(result_object))
                        match_obj = self.make_match_complmentary_fields(result_object, search_field, boost_field, text)
                        if match_obj['score'] > 0:
                            match_dict[match_obj['name']] = match_obj
            except Exception as e:
                logger.error(e, exc_info=True)

        for search_field, boost_field in datasets_with_single_field:
            try:
                results = self.search_single_field(text, search_field, result_size)
                for result in results:
                    result_object = self.get_result_object(result)
                    if not self.exists_match_single_field(match_dict, result_object):
                        logger.debug(self.print_object(result_object))
                        match_obj = self.make_match_single_field(result_object, search_field, boost_field, text)
                        if match_obj['score'] > 0:
                            match_dict[match_obj['name']] = match_obj
            except Exception as e:
                logger.error(e, exc_info=True)

        return sorted(match_dict.items(), key=lambda i: i[1]['score'], reverse=True)

    def test_kit(self, csv_file_name, line_score_limit=300, point_score_limit=100):
        test_set = pd.read_csv(csv_file_name, index_col=0, encoding="utf-8")
        mejorMatch = []
        scoreMejorMatch = []
        matchType = []
        times = []
        textos = []
        for index, row in test_set.iterrows():
            time1 = time.time()
            texto = self.strip_name(row['texto'],' ')
            textos.append(texto)
            matches = self.complete_search(texto, 10)
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
        test_set['texto'] = textos
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
