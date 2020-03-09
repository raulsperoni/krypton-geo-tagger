# coding: utf-8

# In[12]:


from datetime import datetime
from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200}])







def boosting_match_bool_search(text, size=250, boost=2 - 0, negative_boost=0.5,
                               must_not="la el la las los calle psje 1 mas"):
    return es.search(index="montevideo", body=
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
                                "query": "la el la las los calle psje 1 mas",
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
                        "type": "geonames_uy_montevideo limites_barrios v_mdg_espacios_libres"
                    }
                },
                "negative_boost": negative_boost
            }
        },
        "highlight": {
            "fields": {
                "nombre": {},
            }
        }
    })['hits']['hits']


# In[257]:


def search_geo_vias(id, size=10):
    return es.search(index="montevideo", body=
    {
        "from": 0, "size": size,
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "type": "v_mdg_vias"
                    }
                },
                "filter": {
                    "geo_shape": {
                        "geometry": {
                            "indexed_shape": {
                                "index": "montevideo",
                                "id": id,
                                "path": "geometry"
                            }
                        }
                    }
                }
            }
        }
    })['hits']['hits']


# In[258]:


def stripName(name):
    return name.replace(' ', '-').strip().lower()


# In[259]:


def getMatchName(result_obj_1, result_obj_2):
    return result_obj_1['s_name'] + '|' + result_obj_2['s_name']


# In[260]:


def matchedName(match_dict, result_object):
    matched_key = next((key for key in match_dict.keys() if result_object['s_name'] in key), None)
    return matched_key != None


# In[261]:


def getMatchObj(res_obj, geo_obj=None):
    if geo_obj:
        match_score = res_obj['score'] + geo_obj['score']
        match_objects = [res_obj, geo_obj]
        return {'score': match_score, 'objects': match_objects}
    else:
        return {'score': res_obj['score'], 'objects': [res_obj]}

    # In[262]:


from difflib import SequenceMatcher


def commonSignificantSubstringExists(text1, text2, limit=5):
    match = SequenceMatcher(None, text1, text2).find_longest_match(0, len(text1), 0, len(text2))
    # print(text1[match.a: match.a + match.size])  # -> apple pie
    # print(text2[match.b: match.b + match.size])  # -> apple pie
    return match.size > limit


# In[263]:


def sameHighlights(obj1, obj2):
    intersection = set(obj1['highlights']).intersection(obj2['highlights'])
    return len(list(intersection)) == len(obj1['highlights'])


# In[264]:


def tryToMatchLines(match_dict, line_results, all_results, favorMoreMatches=False):
    results = []
    if not favorMoreMatches:
        results = line_results.values()
    else:
        results = sorted(line_results.values(), key=lambda i: len(i['highlights']), reverse=True)
    for res_obj in results:
        if matchedName(match_dict, res_obj):
            # If there is alredy a match that involves this name continue
            continue
        for geo_result in search_geo_vias(res_obj['id']):
            # I only care for those objects intersecting current object AND where part of the original results.
            geo_obj = all_results.get(geo_result['_id'], None)
            if geo_obj and res_obj['s_name'] != geo_obj['s_name'] and not sameHighlights(res_obj,
                                                                                         geo_obj) and not commonSignificantSubstringExists(
                    res_obj['s_name'], geo_obj['s_name']):
                # I only need to care for different objects, matching names and not ids.
                # In case of streets more than one block can intersect with the next one, same name diff id.
                # I discard obj with same highlights or common significant substrings
                match_name = getMatchName(res_obj, geo_obj)
                match_obj = getMatchObj(res_obj, geo_obj)
                match_dict[match_name] = match_obj

            # In[265]:


def tryToMatchPoints(match_dict, point_results, all_results):
    for key, res_obj in point_results.items():
        if matchedName(match_dict, res_obj):
            # If there is alredy a match that involves this name continue
            continue
        match_obj = getMatchObj(res_obj)
        match_dict[res_obj['s_name']] = match_obj


# In[266]:


def tryToMatchPolygons(match_dict, polygon_results, all_results):
    for key, res_obj in polygon_results.items():
        if matchedName(match_dict, res_obj):
            # If there is alredy a match that involves this name continue
            continue
        match_obj = getMatchObj(res_obj)
        match_dict[res_obj['s_name']] = match_obj


# In[267]:


def getResultObject(result):
    result_geo_type = result['_source']['geometry']['type']
    result_id = result['_id']
    result_score = result['_score']
    result_geometry = result['_source']['geometry']
    result_name = 'NO_NAME'
    if result['_source'].get('nombre', None):
        result_name = result['_source']['nombre'].encode('ascii', 'ignore').decode('ascii')
    elif result['_source'].get('aliases', None):
        result_name = result['_source']['aliases'].encode('ascii', 'ignore').decode('ascii')
    result_striped_name = stripName(result_name)
    result_highlights = []
    highlights = result.get('highlight', None)
    if highlights:
        for a in highlights.values():
            for b in a:
                for c in b.split(' '):
                    if '<em>' in c:
                        result_highlights.append(c.replace('<em>', '').replace('</em>', ''))
    return {'id': result_id, 'geo_type': result_geo_type, 'name': result_name, 's_name': result_striped_name,
            'score': result_score, 'geometry': result_geometry, 'highlights': result_highlights}


# In[210]:


def complete_search(text, searchResults=300, favorMoreMatches=False):
    match_dict = {}

    line_results = {}
    point_results = {}
    polygon_results = {}
    all_results = {}

    results = boosting_match_bool_search(text, searchResults)
    for result in results:
        result_object = getResultObject(result)
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

    tryToMatchLines(match_dict, line_results, all_results, favorMoreMatches)
    tryToMatchPoints(match_dict, point_results, all_results)
    tryToMatchPolygons(match_dict, polygon_results, all_results)

    return results, sorted(match_dict.items(), key=lambda i: i[1]['score'], reverse=True)


# In[236]:


import pandas as pd

test_set = pd.read_csv('test_complete - test_complete.csv', index_col=0)

# In[242]:


mejorMatch = []
scoreMejorMatch = []
encontreUbicacionCalle = []
for index, row in test_set.iterrows():
    results, matches = complete_search(row['texto'], 500, True)
    if any(matches):
        mejorMatch.append(matches[0][0])
        scoreMejorMatch.append(matches[0][1]['score'])
        row['encontreUbicacion'] = matches[0][1]['score'] > 12
        encontreUbicacionCalle.append("|" in matches[0][0])
    else:
        mejorMatch.append('')
        scoreMejorMatch.append('')
        encontreUbicacionCalle.append(False)
        row['encontreUbicacion'] = False
test_set['mejorMatch'] = mejorMatch
test_set['scoreMejorMatch'] = scoreMejorMatch
test_set['encontreUbicacionCalle'] = encontreUbicacionCalle

# In[245]:


test_set['encontreUbicacionCalle'] = encontreUbicacionCalle

# In[246]:


test_set

# In[247]:


test_set.to_csv('test_complete.csv')

# In[252]:


test_set[(test_set['tieneUbicacionCalle'] == True) & (test_set['encontreUbicacionCalle'] == False)].shape

# In[253]:


from sklearn.metrics import confusion_matrix as cm

cm(test_set['tieneUbicacion'], test_set['encontreUbicacion'])

# In[254]:


cm(test_set['tieneUbicacionCalle'], test_set['encontreUbicacionCalle'])

# In[207]:


text = test_set['texto'][78]

print(analyzedText(text))
results, matches = complete_search(text, 500, True)
for match in matches[:4]:
    print('✔️ MATCH: {}\t/\t{}'.format(match[1]['score'], match[0]))
print('\n')

# In[147]:


for res in boosting_match_bool_search(text, 450):
    print(res['_source']['nombre'], res['_score'], res.get('highlight'), res['_source']['type'])

