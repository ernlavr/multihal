import json
from pprint import pprint
import time
from urllib import parse, request
from src.network.api.entity import KGEntity
import pdb

def name_to_wiki_id(name):
    pass

def wiki_id_to_name(wiki_id):
    pass

class WikidataAPI():
    def __init__(self, lookup_url="https://www.wikidata.org/w/api.php"):
        self.service_url = lookup_url
        
        
    def getJSONRequest(self, params, attempts=3):
        try:
            #urllib has been split up in Python 3. 
            #The urllib.urlencode() function is now urllib.parse.urlencode(), 
            #and the urllib.urlopen() function is now urllib.request.urlopen().
            #url = service_url + '?' + urllib.urlencode(params)
            url = self.service_url + '?' + parse.urlencode(params)
            
            
            req = request.Request(url)
            #Customize headers. For example dbpedia lookup returns xml by default
            req.add_header('Accept', 'application/json')
        
            response = json.loads(request.urlopen(req).read())
            return response
        
        except:
            print("Lookup '%s' failed. Attempts: %s" % (url, str(attempts)))
            time.sleep(60) #to avoid limit of calls, sleep 60s
            attempts-=1
            if attempts>0:
                return self.getJSONRequest(params, attempts)
            else:
                return None
        
        
    def getURL(self):
        return "https://www.wikidata.org/w/api.php"
    
    def __createParams(self, query, limit, type='item'):
        return {
            'action': 'wbsearchentities',
            'format' : 'json',
            'search': query,
            'type': type,
            'limit': limit,
            'language' : 'en'
        }
    
    def getKGName(self):
        return 'Wikidata'
    
    '''
    Returns list of ordered entities according to relevance: wikidata
    '''
    def __extractKGEntities(self, json, filter=''):
        entities = list()

        for element in json['search']:
            #empty list of type from wikidata lookup
            types = set()
            
            description=''
            if 'description' in element:
                description = element['description']
            
            kg_entity = KGEntity(
                element['concepturi'],
                element['label'],
                description,
                types,
                self.getKGName()
                )
            #We filter according to givem URI
            if filter=='' or element['concepturi']==filter:
                entities.append(kg_entity)  
        return entities
    
    
    
    def getKGEntities(self, query, limit, type='item', filter=''):        
        json = self.getJSONRequest(self.__createParams(query, limit, type), 3)     
        if json==None:
            print("None results for", query)
            return list()
        return self.__extractKGEntities(json, filter) #Optionally filter by URI