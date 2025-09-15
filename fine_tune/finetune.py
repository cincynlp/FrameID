# QA Fine-tuning Script for llama on Frame Identification
import torch
import os
import argparse
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from torch.utils.data import Dataset
import random
import numpy as np

class FrameIdentificationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = list(data)
        self.tokenizer = tokenizer
        self.encoded_data = [self._encode_example(ex, i) for i, ex in enumerate(self.data)]

    def _encode_example(self, ex, idx):
        input_text = ex["question"]
        label_letter = ex["answer"] 
        num_choices = ex['num_choices']
        max_len = self.tokenizer.model_max_length
        
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(ord(label_letter) - ord("A"), dtype=torch.long),
            "num_choices" : torch.tensor(int(num_choices), dtype=torch.long)
        }

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


class SingleSampleCollator:
    def __call__(self, features):
        return features[0]  # each batch is just one sample

class LlamaFrameChoiceModel(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        for param in base_model.parameters():
            param.requires_grad = False

        self.llama = get_peft_model(base_model, LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        ))

        for name, param in self.llama.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        self.tokenizer = tokenizer
        
    def save_pretrained(self, save_directory, **kwargs):
            self.llama.save_pretrained(save_directory, **kwargs)

    def forward(self, input_ids, attention_mask, num_choices, labels=None):
        device = self.llama.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.llama(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits  # [1, seq_len, vocab_size]
        

        last_index = int((attention_mask.sum() - 1).item())
        logits_at_next_token = logits[0, last_index, :]
        
        
        self.choice_token_ids = []
        for i in range(int(num_choices.item())):
            label_str = " " + chr(ord("A") + i)
            token_ids = self.tokenizer.encode(label_str, add_special_tokens=False)
            
            if len(token_ids) != 1:
                raise ValueError(
                    f"Choice '{label_str}' is not a single token: got {token_ids} ({self.tokenizer.convert_ids_to_tokens(token_ids)})"
                )
            
            self.choice_token_ids.append(token_ids[0])
        
        choice_ids = torch.tensor(self.choice_token_ids,  device=device, dtype=torch.long)
        choice_logits = logits_at_next_token.index_select(0, choice_ids)
        
        out = choice_logits.unsqueeze(0)

        loss = None
        if labels is not None:
            # labels = torch.tensor([labels], dtype=torch.long, device=logits.device)
            labels = labels.view(1).to(out.device).long() 
            loss = nn.CrossEntropyLoss()(out, labels)
        return (loss,out) if loss is not None else out
    

def set_seed(seed: int = 130):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch (single GPU).
    Note: with fp16 training, results are reproducible in practice but not bit-exact.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # cuDNN settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed set] {seed}")
    

# Training Setup
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)

    print("--- Evaluation Predictions vs Labels ---")
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        pred_letter = chr(ord("A") + int(pred))
        label_letter = chr(ord("A") + int(label))
        status = "Correct" if pred == label else "Incorrect"
        print(f"Sample {i+1:02d}: Predicted = {pred_letter} | True = {label_letter} {status}")

    return {"accuracy": (predictions == labels).mean()}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune llama for Frame Identification")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing train.jsonl, val.jsonl, and test.jsonl")
    parser.add_argument("--output-dir", type=str, required=True, help="Output Directory to store the trained model.")
    parser.add_argument("--seed", type=int, default=130, help="Random seed for reproducibility")

    args = parser.parse_args()
    set_seed(args.seed)
    
    
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path   = os.path.join(args.data_dir, "dev.jsonl")
    test_path  = os.path.join(args.data_dir, "test.jsonl")
    
    # load and wrap datasets
    train_ds = load_dataset("json", data_files={"train": train_path})["train"]
    val_ds   = load_dataset("json", data_files={"val":   val_path})["val"]
    test_ds  = load_dataset("json", data_files={"test":  test_path})["test"]


    train_dataset = FrameIdentificationDataset(train_ds, tokenizer)
    val_dataset = FrameIdentificationDataset(val_ds, tokenizer)
    test_dataset = FrameIdentificationDataset(test_ds, tokenizer)
    
    model = LlamaFrameChoiceModel(model_name, tokenizer).to("cuda")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
        fp16=True,
        seed= args.seed
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=SingleSampleCollator(),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    
    model.llama.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Evaluate on validation set
    val_results = trainer.evaluate()
    print(f"Validation Accuracy: {val_results['eval_accuracy'] * 100:.2f}%")

    # Evaluate on test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy'] * 100:.2f}%")

if __name__ == "__main__":
    main()
