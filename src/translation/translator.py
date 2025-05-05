import polars as pl
import transformers
import torch
import datasets
import os
import src.utils.helpers as hlp
import string


class Translator():
    def __init__(self, model_name, args):
        self.args = args
        self.pipeline = self.load_model_and_tokenizer(model_name)
        
    def load_model_and_tokenizer(self, model_name):
        # Load the model and tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8
        
        return transformers.pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            src_lang='eng_Latn',
            tgt_lang=self.args.tgt_lang,
            device=device,
            max_length=1024,
            batch_size=self.batch_size,
            
            # beam decoding
            num_beams=5,
            length_penalty=1.1,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        
    def get_non_translatable_mask(self, texts: list[str]) -> bool:
        mask = []
        for i in texts:
            if i.translate(str.maketrans('', '', string.punctuation)).strip().isdigit():
                mask.append(True)
            else:
                mask.append(False)
                
        return mask
        
    
    def _translate(self, text: str) -> str:
        return text
    
    def translate_batch(self, batch, translation_pipeline, cols):
        for col in cols:            
            # get IDs of context which is non-null
            data = batch[col]
            if col == 'trip_labels':
                print(data[0])
                data = [i.replace(" ", "; ") for i in data]
                data = [i.replace("_", " ") for i in data]
                print(data[0])
            
            non_null_data_id = [i for i, x in enumerate(data) if x is not None]
            mask = self.get_non_translatable_mask(data)
            replacements = [data[i] if mask[i] else None for i in range(len(data))]
            
            # select IDs of context which are non-null
            non_null_data_text = [data[i] for i in non_null_data_id]
            data_transl = translation_pipeline(non_null_data_text)
            data_transl_loc = {k:v['translation_text'] for k, v in zip(non_null_data_id, data_transl)}
            
            translations = [data[i] if i not in data_transl_loc else data_transl_loc[i] for i in range(len(data))]
            
            # replace masked values
            translations = [replacements[idx] if replacements[idx] is not None else i for idx, i in enumerate(translations)]
            
            batch[col] = translations
        return batch
        

    
    def translate_df(self, df: pl.DataFrame, cols: list, save_name="") -> pl.DataFrame:
        # prepare dataframe for inference
        df = df.with_columns(
                translation=pl.lit('N/A'),
            )
        
        dataset = datasets.Dataset.from_dict(df.to_dict(as_series=False))
        translated_dataset = dataset.map(lambda x: self.translate_batch(x, self.pipeline, cols), batched=True, batch_size=self.batch_size)
        
        # convert back to polars dataframe
        translated = pl.from_dict(translated_dataset.to_dict())
        save_path = os.path.join(self.args.data_dir, f"multihal_{save_name}_{self.args.tgt_lang}.json")
        translated.write_json(save_path)
        return translated