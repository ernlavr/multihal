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
    judged_score: pl.Int32     # 0-1


class JudgeBaseClass(ABC):

    def get_results_df(self):
        # create a dataframe based on JudgeEvalResult
        schema = {col:datatype for col, datatype in JudgeEvalResult.__annotations__.items()}
        output = pl.DataFrame(schema=schema)
        return output
    
    def add_result(self, result: dict, df: pl.DataFrame) -> pl.DataFrame:
        result_df = pl.from_dict(result, schema=df.schema)
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
        # weird indenting is for formatting
        messages = [
            {"role": "system", "content": f"""Score the given Wikidata Knowledge Graph path on how informative and relevant it is with respect to the given answer and question. The path can have multiple hops where the entities are connected predicates seperating them. 

Give me your output in YAML format with a given score in Likert scale from 1 to 5.
1 - Very poor. Completley unrelated path.
2 - Poor. Syntactic overlap may exist between the path and question/answer but semantics are different.
3 - Normal. Syntactic overlap exists touching upon some semantics. Could be usable as a starting point for information support, but not directly related to the question without knowing the answer.
4 - Good. Good semantic overlap which allows the question to be implicitly answered with the path.
5 - Excellent. Directly addresses the question.

Here is an expected format of the input:
Question: What is the capital of France?
Answer: Paris
Path: Napoleon residence Paris capital of France

Your output needs to be only the score, no explanation or justification is needed. Example:
Score: 5"""},
            {"role": "user", "content": f"Question: {question}; \nAnswer: {answer}; \nPath: {triple}"},
        ]
        return messages
    
    def get_prompt_top_triples(self, question, answer, triples, num_triples=10):
        """ Returns the prompt for the LLM to evaluate the relevance of the triple to the question and answer """
        triples = "\n".join(triples)
        messages = [
            {"role": "system", "content": f"""From the given Wikidata Knowledge Graph paths, you need to select the Top {num_triples} most relevant paths that are informative and relevant with respect to answering the given question.
The paths can have multiple hops where the entities and predicates alternate. Each path is seperated by a new line and the within the path the entities and predicates are seperated by whitespace. Your output needs to be exact matches to the paths given in the input.

The number of paths can vary but here is an example of the input:
Question: What is the capital of France?
Answer: Paris
Paths: France capital Paris
Microsoft founder Bill Gates
Napoleon residence Paris capital of France

Here is an expected format of the output:
```yml
Path: France capital Paris
Path: Napoleon residence Paris capital of France
```"""
            },
            {"role": "user", "content": f"Question: {question}; \nAnswer: {answer}; \nPaths: {triples}"},
        ]    
        return messages
    
    def split_unprocessed_ents(self, data: pl.DataFrame):
        unprocessed = data.filter(~data['responses'].is_in(['N/A', "", "<NO_PATHS_FOUND>"]))
        processed = data.filter(data['responses'].is_in(['N/A', "", "<NO_PATHS_FOUND>"]))
        return unprocessed, processed