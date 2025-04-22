from semscore import *
import numpy as np
import polars as pl
from tqdm import tqdm
import scipy.stats as stats

def get_similarities(model, x, y):
    embs1 = model.get_embeddings(x) # expected ans
    embs2 = model.get_embeddings(y)
    return model.get_similarities(embs1, embs2)[0]

# model_name = "sentence-transformers/all-mpnet-base-v2"
model_roberta = EmbeddingModelWrapper(model_path="sentence-transformers/roberta-base-nli-mean-tokens", bs=None)
model_mpnet = EmbeddingModelWrapper(model_path="sentence-transformers/all-mpnet-base-v2", bs=None)

# Load data
data_name = "llm_eval_google-gemini-2.0-flash-001_rag.json"
# data_name = "llm_eval_google-gemini-2.0-flash-001_grag.json"
# data_name = "llm_eval_google-gemini-2.0-flash-001_grag_gem_judge.json"
data = pl.read_json(data_name)
output = pl.DataFrame(schema=data.schema)

# add score column
output = output.with_columns(
    pl.Series(name="semscore_mpnet", values=[], dtype=pl.Float32),
    pl.Series(name="semscore_roberta", values=[], dtype=pl.Float32)
)

for i in tqdm(data.iter_rows(named=True), total=data.shape[0]):
    i['semscore_mpnet'] = get_similarities(model_mpnet, i['output'], i['model_response'])
    i['semscore_roberta'] = get_similarities(model_roberta, i['output'], i['model_response'])
    output = pl.concat([output, pl.DataFrame(i, schema=output.schema)])

# print correlation
corr = stats.pearsonr(output['semscore_mpnet'].to_list(), output['semscore_roberta'].to_list())
print(corr)
output.write_json(f"semscore_{data_name}")