import os
from src.textsummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk

from src.textsummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def tokenize_batch(self, batch):
        input_encodings = self.tokenizer(
            batch[self.config.input_column],
            max_length=self.config.max_input_length,
            truncation=True,
            padding="max_length",
        )

        # Use text_target for newer tokenozers (replaces as target_tokenizer)
        target_encodings = self.tokenizer(
            text_target=batch[self.config.target_column],
            max_length=self.config.max_target_length,
            truncation=True,
            padding="max_length",
        )

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
        }

    def transform(self):
        train_ds = load_from_disk(str(self.config.ingested_train_dir))
        test_ds = load_from_disk(str(self.config.ingested_test_dir))

        print(f"Train size before: {len(train_ds)}")
        print(f"Test size before: {len(test_ds)}")

        train_ds = train_ds.map(self.tokenize_batch, batched=True)
        test_ds = test_ds.map(self.tokenize_batch, batched=True)

        train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        train_ds.save_to_disk(str(self.config.root_dir / "transformed_train"))
        test_ds.save_to_disk(str(self.config.root_dir / "transformed_test"))

        print('Transformation complete. Splits savved to disk:', self.config.root_dir)
        return train_ds, test_ds
    
