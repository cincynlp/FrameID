import os
import argparse
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import PeftModel

MAX_CHOICES = 43
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"


class FrameIdentificationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = list(data)
        self.tokenizer = tokenizer
        self.encoded_data = [self._encode_example(ex) for ex in self.data]

    def _encode_example(self, ex):
        input_text = ex["question"]
        label_number = int(ex["correct_answer"]) 


        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length= self.tokenizer.model_max_length,
            padding=False,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_number - 1, dtype=torch.long),  # labels 0–42
        }

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


class SingleSampleCollator:
    def __call__(self, features):
        return features[0]


class LlamaFrameChoiceModel(torch.nn.Module):
    def __init__(self, base_model_name, lora_path):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=None
        )
        self.llama = PeftModel.from_pretrained(base_model, lora_path)
        self.llama = self.llama.eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.choice_token_ids = [self.tokenizer.encode(str(i+1), add_special_tokens=False)[0] for i in range(MAX_CHOICES)]
        

    def forward(self, input_ids, attention_mask, label=None ):
        input_ids = input_ids.to(self.llama.device)
        attention_mask = attention_mask.to(self.llama.device)

        outputs = self.llama(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits

        last_index = attention_mask.sum() - 1
        logits_at_next_token = logits[0, last_index, :]
        
        
        choice_logits = logits_at_next_token[self.choice_token_ids]
        padded_logits = choice_logits.unsqueeze(0)


        loss = None
        if label is not None:
            label = label.to(logits.device) 
            loss = torch.nn.CrossEntropyLoss()(padded_logits, label.unsqueeze(0))

        return (loss, padded_logits) if loss is not None else padded_logits


def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # print(logits, labels)
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Frame Identification with LoRA adapter (letter-choice logits).")
    p.add_argument("--dataset", required=True, help="Path to JSONL file (e.g., /path/to/test.jsonl)")
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter folder")
    return p.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token


    new_data = load_dataset("json", data_files={"test": args.dataset})["test"]
    new_dataset = FrameIdentificationDataset(new_data, tokenizer)

    model = LlamaFrameChoiceModel(BASE_MODEL_NAME, args.adapter)

    training_args = TrainingArguments(
                output_dir=f"./tmp",
                per_device_eval_batch_size=1,
                dataloader_drop_last=False,
                fp16=True,
                report_to="none"
            )

    trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                eval_dataset=new_dataset,
                data_collator=SingleSampleCollator(),
                compute_metrics=compute_metrics
            )

    eval_result = trainer.evaluate()
    acc = eval_result['eval_accuracy'] * 100

    print(f"\n Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
