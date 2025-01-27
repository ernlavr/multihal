import pandas as pd
import logging
import os
import time
import numpy as np
from datetime import datetime
import sys


class KgLogger():
    def __init__(self, create_log=False, continue_from=None):
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        self.num_queries = 0
        if create_log:
            self.setupLogging()

    def setupLogging(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        os.makedirs('logs', exist_ok=True)
        filename = f'logs/{self.timestamp}.log'
        logging.basicConfig(filename=filename, 
                            format='%(asctime)s %(message)s',
                            level=logging.INFO,
                            )
        logging.info("Starting logging")

    def setupContinueLogging(self):
        filename = f'logs/{self.continue_from}.log'
        parsed_time = datetime.strptime(self.continue_from, '%Y%m%d-%H%M%S')
        self.timestamp = parsed_time.strftime("%Y%m%d-%H%M%S")
        logging.basicConfig(filename=filename, 
                            format='%(asctime)s %(message)s',
                            level=logging.INFO)
        logging.info("<---                                              --->")
        logging.info(f"<--- Continuing from previous log file {filename} --->")
        logging.info("Starting logging")

    def getResultDataframe(self, hops):
        tmp_cols = ['q_ID', 'utterance', 'answer', 'entity']
        tmp_cols.extend([f'hop_{i}' for i in range(1, hops+1)])
        self.df = pd.DataFrame(columns=tmp_cols)
        return self.df
    
    def concatResults(self, df, datapoint, matches):
        qid = datapoint['questionid']
        utterance = datapoint['utterance']
        answer = ' <SEP> '.join(datapoint['answers_str']) if datapoint['answers_str'] else datapoint['answers']
        entity = ' <SEP> '.join(datapoint['answers']) if datapoint['answers'] else datapoint['answers']
        
        # hops
        hops = []
        for hop in matches:
            h = []
            for n in hop:
                _hop = n[0]
                val = n[1]
                vs = val['s']
                vp = val['p']
                vo = val['o']
                hop = "-".join((_hop, vs, vp, vo))
                if hop not in h:
                    h.append(hop)   

            if len(h) > 1: h = " | ".join(h)
            hops.append(h)
        
        tmp = [qid, utterance, answer, entity]
        tmp.extend(hops)
        
        n_cols = len(df.columns)
        if n_cols != len(tmp):
            # add None values
            number_to_add = n_cols - len(tmp)
            tmp.extend([np.nan for i in range(number_to_add)])
        
        return pd.concat([df, pd.DataFrame([tmp], columns=self.df.columns)])

    def logHopResults(self, datapoint, entities, next_level_entities, records, subject_label, hop):
        logging.info(f"Subj: {subject_label}; E: {len(entities)}; NLE: {len(next_level_entities)}")
        for r in records:
            hop_result = r[1]

            s, sLabel = hop_result['s'], hop_result['sLabel']
            p, pLabel = hop_result['p'], hop_result['pLabel']
            o, oLabel = hop_result['o'], hop_result['oLabel']

            logging.info(f"Hop {hop}: <{sLabel} ({s}) - {pLabel} ({p}) - {oLabel} ({o})>")
        if len(records) == 0: # loop is empty
            logging.info(f"No results for Hop {hop}")
    
    def saveResults(self, df):
        os.makedirs('results', exist_ok=True)

        if self.continue_from:
            df.to_csv(f'results/{self.continue_from}.csv', index=False)
        else:
            df.to_csv(f'results/{self.timestamp}.csv', index=False)

    def logTimeDetails(self):
        if self.num_queries == 0:
            self.query_time = time.time()
        
        # Log time details
        self.num_queries += 1

        # average time per query, time so far
        time_so_far = time.time() - self.start_time
        avg_time = time_so_far / self.num_queries

        logging.info(f"Elapsed: {time_so_far:.2f}s; Avg/Question: {avg_time:.2f}s; Questions: {self.num_queries}")
        
        # reset query time
        self.query_time = time.time()
        