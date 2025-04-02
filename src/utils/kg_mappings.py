import polars as pl
import src.utils.helpers as hlp

def tag_question_type(datapoint: dict) -> str:
    output = datapoint['output']
    date_parsed = hlp.parse_flexible_date(output)
    if date_parsed is not None and len(output) < 20:
        tag = "time"
    else:
        tag = "other"
    return tag
