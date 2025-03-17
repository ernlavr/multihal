from abc import ABC
from dataclasses import dataclass
import polars as pl

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
        