import evaluate
import polars as pl
import src.network.LlmApi as llmApi
import src.utils.prompts as prompts
import numpy as np
import logging
from tqdm import tqdm
from src.evaluation.semscore.semscore import EmbeddingModelWrapper

class KnowledgeInjectionEval():
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.semantic_similarity = EmbeddingModelWrapper(model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", bs=None)
    
    def _get_task_prompt(self, task):
        if task == 'grag':
            return prompts.get_GRAG_prompt(self.args.tgt_lang)
        elif task == 'rag':
            return prompts.get_RAG_prompt
        elif task == 'qa':
            return prompts.get_QA_prompt(self.args.tgt_lang)
        
    def get_score(self, ground_truth, prediction):
        # return self.semantic_similarity.get_embeddings_e5(ground_truth, prediction)
        gt_emb = self.semantic_similarity.get_embeddings(ground_truth)
        pred_emb = self.semantic_similarity.get_embeddings(prediction)
        return self.semantic_similarity.get_similarities(gt_emb, pred_emb)
    
    def _get_eval_dataset(self, data):
        """ Currently the score dataset and full dataset is seperate. We're missing context
            and some other useful information so merge it back in. This is a bit hacky and should
            be fixed at the core.
        """
        score_data = pl.read_json(self.args.load_score_dataset)
        filtered_data = data.filter(pl.col("id").is_in(score_data['id']))
        
        # remove overlapping columns
        cols_to_drop = score_data.columns
        cols_to_drop.remove('id')
        
        filtered_data = filtered_data.drop(cols_to_drop, strict=False)
        
        # add missing coluns from filtered_data into score_data based on ID
        score_data = score_data.join(filtered_data, on='id', maintain_order='left')
        return score_data
        
    def _merge_and_save_results(self, row, data, task):
        data = pl.concat([data, pl.DataFrame(row, schema=data.schema)])
        data.write_json(f"{self.args.data_dir}/llm_eval_{self.model_name.replace('/', '-')}_{self.args.tgt_lang}_{task}.json")
        return data
        
    
    def run_eval(self, data: pl.DataFrame, task="grag"):
        # fetch the data, extend it with another row for model prediction and score
        data = data.filter(pl.col("judged_by") != "<NOT_JUDGED>")
        data = data.filter(pl.col("judged_score") >= 4)
        unprocessed_dp = None
        if 'model_response' not in data.columns:
            unprocessed_dp = data.with_columns(
                model_response=pl.Series(["N/A" for _ in range(len(data))]),
                responding_model=pl.Series([self.model_name for _ in range(len(data))]),
                sem_score=pl.Series([np.inf for _ in range(len(data))])
            )
        else:
            unprocessed_dp = data.filter(pl.col("model_response") == 'N/A')
        
        prompt_func = self._get_task_prompt(task)
        output = pl.DataFrame(schema=unprocessed_dp.schema)
        
        if task == 'rag':
            unprocessed_dp = unprocessed_dp.filter((
                    pl.col("context").is_not_null() &
                    ~pl.col("context").str.contains('https://'),
                    pl.col("context").str.len_chars() > 20,
                )).unique('id')
            
            
        
        for idx, row in enumerate(tqdm(unprocessed_dp.iter_rows(named=True), total=len(unprocessed_dp))):
            question = row['input']
            context = None
            answer = row['output']
            if task == 'rag':
                context = row['context']
            if task == 'grag':
                context = row['trip_labels'].replace("_", " ")
            if task != 'qa' and (context is None or context == 'N/A'):
                continue
            
            prompt = prompt_func(context, question)
            if idx % 100 == 0:
                logging.info(prompt)
            api_response = llmApi.post_api_request(self.model_name, prompt, self.args, temp=1, max_tokens=1024)
            if api_response is None:
                logging.error(f"Failed to get API response for row {row['id']}")
                row['model_response'] = "<API_ERROR>"
                row['sem_score'] = np.inf
                output = self._merge_and_save_results(row, output, task)
                continue
                            
            model_response = api_response['choices'][0]['message']['content']
            if model_response.startswith("Answer: "):
                model_response = model_response[len("Answer: "):]
                
            row['model_response'] = model_response
            row['sem_score'] = self.get_score(answer, model_response)
            # add row to output
            output = self._merge_and_save_results(row, output, task)
            
            _datapoint = pl.from_dict(row, strict=False)
            data = data.update(_datapoint, on="id")
            data.write_json(f"{self.args.data_dir}/llm_eval_{self.model_name.replace('/', '-')}_full_{self.args.tgt_lang}_{task}.json")
            
        return data