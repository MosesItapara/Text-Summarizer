from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk
import torch
from src.textsummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name).to(self.device)

        seq2seq_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

        train_ds = load_from_disk(str(self.config.transformed_train_dir))
        test_ds = load_from_disk(str(self.config.transformed_test_dir))

        training_args = TrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            test_dataset=test_ds,
            eval_dataset=test_ds,
            data_collator=seq2seq_collator,
            tokenizer=tokenizer,
        )

        trainer.train()
        print(' Training complete')