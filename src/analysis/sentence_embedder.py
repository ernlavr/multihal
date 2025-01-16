import sentence_transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import logger as stLogger
import polars as pl
import src.utils.decorators as dec
import logging

class SentenceEmbeddings():
    def __init__(self, data, args):
        stLogger.setLevel(logging.WARNING) # Suppress logging from SentenceTransformers, TODO find a better solution
        self.args = args
        self.embedder = self.get_embedder()

    def get_embedder(self):
        return SentenceTransformer(self.args.sentence_embedder)
    
    def gen_embeddings(self, data: pl.DataFrame):
        if self.args.debug_mode:
            logging.info("Generating embeddings")

        data = data.with_columns(
            embeddings=pl.col('input').map_elements(lambda x: self.embedder.encode(x).tolist())
        )

        if self.args.debug_mode:
            logging.info("Got embeddings")
            data.write_json('data/data_with_embeddings.json')
        return data