from abc import ABC
from dataclasses import dataclass
import polars as pl
import re
import src.utils.config as config

@dataclass
class JudgeEvalResult:
    id: str                 # id of the datapoint
    source_dataset: str     # source dataset
    task: str               # task of the datapoint
    domain: str             # question domain from dataset
    input: str
    output: str
    responses: str          # wikidata paths
    trip_labels: str        # wikidata path labels
    judged_by: str          # model name
    judged_label: str       # yes, no, unsure
    judged_score: float     # 0-1


class JudgeBaseClass(ABC):

    def get_results_df(self):
        # create a dataframe based on JudgeEvalResult
        schema = {col:datatype for col, datatype in JudgeEvalResult.__annotations__.items()}
        output = pl.DataFrame(schema=schema)
        return output
    
    def add_result(self, result: JudgeEvalResult, df: pl.DataFrame) -> pl.DataFrame:
        result_df = pl.from_dict(result.__dict__)
        return pl.concat([df, result_df])
    
    def filter_circular_trips(self, data: pl.DataFrame):
        trip_labels = []
        trip_codes = []
        _RE_COMBINE_WHITESPACE = re.compile(r"\s+") # any number of whitespace to 1 whitespace

        # filter out circular paths
        for label, trip in zip(data['trip_labels'].split(config.LIST_SEP), data['responses'].split(config.LIST_SEP)):
            if len(trip) == 0: 
                continue
            
            trip_split = _RE_COMBINE_WHITESPACE.sub(" ", trip).strip().split()
            if trip_split[0] != trip_split[-1] and label not in trip_labels: 
                trip_labels.append(label)
                trip_codes.append(" ".join(trip_split))
        
        return trip_codes, trip_labels
        
    def get_prompt_triple_relevance(self, question, answer, triple):
        """ Returns the prompt for the LLM to evaluate the relevance of the triple to the question and answer """
        messages = [
            {"role": "system", "content": r"""Rate the given Wikidata Knowledge Graph triple on how informative and relevant it is with respect to the given answer and question.
            Give me your output in YAML format with of a given score of relevance from 0-1.
            The triples can have multiple hops where the object and subject alternates with predicates seperating them.
            Each triple is contained within parenthesis and whole triples are seperated by a semicolon.
            
            Here is an expected format of the input:
            Question: What is the capital of France?
            Answer: Paris
            Triples: (France, capital, Paris);
            
            Here is an expected format of the output:
            ```yml
            Triple: France, capital, Paris
            Score: 0.9
            ```
            
            """},
            {"role": "user", "content": f"Question: {question}; \nAnswer: {answer}; \nTriple: ({triple})"},
        ]
        return messages
    
    def get_prompt_top_triples(self, question, answer, triples):
        """ Returns the prompt for the LLM to evaluate the relevance of the triple to the question and answer """
        triples = [f"({triple})" for triple in triples]
        triples = "; ".join(triples)
        messages = [
            {"role": "system", "content": r"""From the given Wikidata Knowledge Graphs triples, you need to select the top 5 most relevant triples that are informative and relevant with respectto the given answer and question.
            If there are more than 5 triples, then simply return them.
             
            Give me your output in YAML format with of a given score of relevance from 0-1.
            The triples can have multiple hops where the object and subject alternates with predicates seperating them.
            Each triple is contained within parenthesis and whole triples are seperated by a semicolon.
            
            Here is an expected format of the input:
            Question: What is the capital of France?
            Answer: Paris
            Triples: (France, capital, Paris); (Microsoft, founder, Bill Gates)
            
            Here is an expected format of the output:
            ```yml
            Triple: France, capital, Paris
            Score: 0.9
            Triple: Microsoft, founder, Bill Gates
            Score: 0.05
            ```
            
            """},
            {"role": "user", "content": f"Question: {question}; \nAnswer: {answer}; \nTriple: ({triples})"},
        ]
        return messages
    
    def split_unprocessed_ents(self, data: pl.DataFrame):
        unprocessed = data.filter(~data['responses'].is_in(['N/A', "", "<NO_PATHS_FOUND>"]))
        processed = data.filter(data['responses'].is_in(['N/A', "", "<NO_PATHS_FOUND>"]))
        return unprocessed, processed