import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')

    def generate_batch_sized_chunks(self, dataset, batch_size):
        """Yield successive batch_size chunks from the dataset."""
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer):
        article_batches = list(
            self.generate_batch_sized_chunks(
                dataset[self.config.input_column], self.config.batch_size
            )
        )
        target_batches = list(
            self.generate_batch_sized_chunks(
                dataset[self.config.target_column], self.config.batch_size
            )
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)
        ):
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )

            summaries = model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                length_penalty=0.8,
                num_beams=self.config.num_beams,
                max_length=self.config.max_generate_length,
            )

            decoded_summaries = [
                tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for s in summaries
            ]

            metric.add_batch(
                predictions=decoded_summaries,
                references=target_batch,
            )

        return metric.compute()

    def evaluate(self):
        tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_path))
        model.to(self.device)
        model.eval()

        test_ds = load_from_disk(str(self.config.ingested_test_dir))
        rouge_metric = evaluate.load('rouge')

        print('Running evaluation on test set...')
        score = self.calculate_metric_on_test_ds(
            test_ds, rouge_metric, model, tokenizer
        )

        rouge_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        rouge_dict  = {rn: round(score[rn], 4) for rn in rouge_names}

        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(str(self.config.root_dir / self.config.metric_file_name), index=False)

        print('\n=== ROUGE Scores ===')
        print(df.to_string())
        print(f'\nMetrics saved to: {self.config.root_dir / self.config.metric_file_name}')
        return df