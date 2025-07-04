import socket
import json
import time
from pprint import pprint
import logging
import src.network.api.endpoints as endpoints

class NetworkBridge:
    def __init__(self, host="127.0.0.1", port=12347):
        self.host = host
        self.port = port
        self.wd_endpoint = endpoints.WikidataEndpoint()
        

    def forward_to_container(self, query):
        """Forward the incoming traffic to the container at localhost:1234/api/endpoint/sparql"""
        # Create a TCP/IP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect("localhost", 1234)  # Connect to the container
            client_socket.sendall(query.encode("utf-8"))
            print(f"Sent: {query}")

            # Receive response from the container
            response = client_socket.recv(1024)
            print(f"Received")

    def send_message(self, host="127.0.0.1", port=12347, message="Hello, Server!", use_only_wd=False):
        """ Send a message to the server endpoint and return the response """
        
        # first try to query wikidata public endpoint
        wd_response = self.query_wd_endpoint(message)
        if wd_response is not None:
            return wd_response
        
        if use_only_wd:
            return wd_response
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((host, port))  # Connect to the server
            client_socket.sendall(message.encode("utf-8"))  # Send the message
            # print(f"Sent: {message}")

            # Receive response from the server
            response = b""
            while True :
                part = client_socket.recv(1024)
                
                response += part
                # check if last part is <|EOS|>
                if response.endswith(b"<|EOS|>"):
                    response = response[:-7]
                    break
            
            response = response.decode("utf-8")
            if response.startswith("Error"):
                logging.error(f"Error: {response}")
                return None
            
            # parse the response as json
            response = json.loads(response)
            return response

    def query_wd_endpoint(self, query):
        query = query.replace("<|EOS|>", "") # EOS is a custom token for us, WD doesnt need it
        results = self.wd_endpoint.getQueryResults(query, 3)
        if results is None:
            logging.error("Error: No results from the endpoint")
            return None
        
        return results

if __name__ == "__main__":
    prefixes = r"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wds: <http://www.wikidata.org/entity/statement/>
        PREFIX wdv: <http://www.wikidata.org/value/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""
    query = r"""
       
        SELECT ?p ?o ?pLabel ?oLabel WHERE {
        # Main claims about Natalie Portman
            wd:Q37876 ?p ?o .
        FILTER CONTAINS(str(?p),"wikidata.org/prop/direct/")
        FILTER CONTAINS(str(?o), 'en')


        # Link the property to its label
        # ?property wikibase:claim ?p .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }

        """

    query = r"""
        SELECT ?p ?o ?pLabel ?oLabel WHERE {
            wd:Q34086 ?p ?o .
        FILTER CONTAINS(str(?p),"wikidata.org/prop/direct/")
        FILTER CONTAINS(str(?o),"en")

        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
    """

    query = r"""
        SELECT ?entity ?label
        WHERE {
        ?entity skos:altLabel|rdfs:label ?label .
        FILTER(LANG(?label) = "en") # Restricting to English labels
        FILTER(LCASE(?label) = "redskins") # Case-insensitive match
        }
        LIMIT 15
    """

    # clean up query
    query = prefixes + query + "<|EOS|>"
    print(query.replace("\n", " ").strip())

    nb = NetworkBridge()
    response = nb.send_message(message=query)
    pprint(response)
