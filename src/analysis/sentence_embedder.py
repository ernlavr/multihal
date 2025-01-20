import sentence_transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import logger as stLogger
import src.utils.decorators as dec
import polars as pl
import src.utils.decorators as dec
import logging
import time

class SentenceEmbeddings():
    def __init__(self, data, args):
        stLogger.setLevel(logging.WARNING) # Suppress logging from SentenceTransformers, TODO find a better solution
        self.args = args
        self.embedder = self.get_embedder()

    def get_embedder(self):
        return SentenceTransformer(self.args.sentence_embedder)
    
    @dec.cache_decorator("embeddings")
    def gen_embeddings(self, data: pl.DataFrame):
        if self.args.debug_mode:
            logging.info("Generating embeddings")

        start_time = time.time()
        data = data.with_columns(
            embeddings=pl.col('input').map_elements(lambda x: self.embedder.encode(x).tolist())
        )
        end_time = time.time()
        logging.info(f"Embeddings generated in: {end_time - start_time:.2f} seconds; for dataset of size: {data.shape[0]}")

        if self.args.debug_mode:
            logging.info("Got embeddings")
            data.write_json('output/data/data_with_embeddings.json')
        return data