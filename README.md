## Code For the paper : "Do LLMs Encode Frame Semantics? Evidence from Frame Identification". 

## Environment Setup

- **Python:** 3.12.8  
- **PyTorch:** 2.5.1+cu121  
- **Transformers:** 4.49.0  

## Data Formatting

```bash
cd data
python convert_csv_to_json.py --folder fn1.7/data-csv
````

This will generate:

```
json-output/
```

---

## Preparing Data for Fine-Tuning

```bash
cd fine_tune
python data_for_finetuning.py --folder ../data/fn1.7/json_output --outdir finetune_data_1_7
```

---

## Fine-Tuning

```bash
python finetune.py --data-dir finetune_data_1_7 --output-dir model_1_7
```

The trained model and adapter will be saved in:

```
model_1_7/
```

---

## Evaluation

1. Copy the adapter files to a new folder:

```bash
cd evaluate
mkdir lora_model_1_7
cp ../fine_tune/model_1_7/adapter_config.json lora_model_1_7/
cp ../fine_tune/model_1_7/adapter_model.safetensors lora_model_1_7/
```

2. Run evaluation:

```bash
python evaluate.py --dataset ../fine_tune/finetune_data_1_7/test.jsonl --adapter lora_model_1_7
```
This will output evaluation results.

## Contact

For questions or issues, please:
- Contact: [chundrja@mail.uc.edu](mailto:chundrja@mail.uc.edu)
