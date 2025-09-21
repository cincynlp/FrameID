
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

import re
import random
import numpy as np


def set_seed(seed):
    print(f'Setting seed {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seeds = [42, 123, 2025]

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").eval()




def format_unified_prompt(entry):
    option_lines = "\n".join(f"{opt} - Lexical Unit Definition: {lu_def}" for opt, lu_def in zip(entry["frame_name"],entry['lu_definitions']))
    if prompt_type == 'zero_shot':
            prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in FrameNet semantics.

Your task is to identify the most appropriate FrameNet frame that best captures the meaning of a given target word in context.

You will be given:
- A sentence containing the target word.
- Target Word
- A list of frames and their corresponding lexical unit definitions.

Your output must be a **single JSON object** in this exact format:
{{"frame_Name": "Intentionally_act"}}

Where:
- "frame_Name" is the exact name of the selected FrameNet frame.

<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Sentence: {entry["question"]}

Target Word: {entry["target_word"]}

Frames:
{option_lines}

Which of the given frames best represents the meaning of the target word "{entry['target_word']}" in the sentence above ?

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>'''
        
    else:

        prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in FrameNet semantics.

Your task is to identify the most appropriate FrameNet frame that best captures the meaning of a given target word in context.

You will be given:
- A sentence containing the target word.
- Target Word
- A list of frames and their corresponding lexical unit definitions.

Your output must be a **single JSON object** in this exact format:
{{"frame_Name": "Intentionally_act"}}

Where:
- "frame_Name" is the exact name of the selected FrameNet frame.

<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Sentence: united states helps uzbekistan secure dangerous nuclear materials : energy agency announces completion of secret uranium transfer back to russia

Target Word: secure

Frames:
Getting - Lexical Unit definition : secure.v - succeed in obtaining.
Being_at_risk - Lexical Unit definition : secure.v - protect against threats.
Protecting -  Lexical Unit definition : secure.v - .
Attaching - Lexical Unit definition : secure.v - make secure; fix or fasten securely.

Which of the given frames best represents the meaning of the target word "secure" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{"frame_Name": "Protecting"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Sentence: these shipments of highly enriched uranium spent fuel are part of nnsa 's efforts to make sure this type of material does n't fall into the wrong hands , ' nnsa administrator linton f. brooks said in an april 20 press release .

Target Word:  uranium

Frames:
Substance - Lexical Unit definition : uranium.n - the chemical element of atomic number 92, a grey dense radioactive metal used as a fuel in nuclear reactors.

Which of the given frames best represents the meaning of the target word "uranium" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{"frame_Name": "Substance"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Sentence: there 's a lot of smart sensors that take away the ( guesswork ) of the soldiers , ' said bruce baldwin , the u.s. army chemical school 's lead trainer for new systems. '' ( the olympian online , 20apr06 , christian hill ) ( link )".

Target Word: say

Frames:
Communication - Lexical Unit definition : say.v - communicate a message nonverbally.
Text_creation - Lexical Unit definition : say.v - recite or create a speech, formula, or other meaningful text.
Statement - Lexical Unit definition : say.v - utter words so as to convey information, an opinion, an instruction, etc.
Spelling_and_pronouncing - Lexical Unit definition : say.v - pronounce.


Which of the given frames best represents the meaning of the target word "say" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{"frame_Name": "Statement"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Sentence: the transfer - part of the global threat reduction initiative ( gtri ) , a nonproliferation effort that secures high-risk nuclear and radiological materials and equipment around the world - resulted from a cooperative effort by the united states , uzbekistan , kazakhstan and russia , as well as coordination through the international atomic energy agency ( iaea ) . '' ( news blaze , 21apr06 ) ( link )",


Target Word:  international

Frames:
Foreign_or_domestic_country - Lexical Unit definition : international.a - of multiple nations, especially excluding the current one.
Political_locales - Lexical Unit definition : international.a - pertaining to multiple nations.


Which of the given frames best represents the meaning of the target word  "international" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{"frame_Name": "Political_locales"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Sentence: india and us to advance cooperation to deal with terrorism.

Target Word: deal

Frames:
Quantified_mass - Lexical Unit definition : deal.v - .
Speak_on_topic - Lexical Unit definition : deal.v - .
Be_in_agreement_on_action - Lexical Unit definition : deal.v - 
Resolve_problem - Lexical Unit definition : deal.v - take action to resolve a problem..
Make_agreement_on_action - Lexical Unit definition : deal.v -.


Which of the given frames best represents the meaning of the target word "deal" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{"frame_Name": "Resolve_problem"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Sentence: {entry["question"]}

Target Word: {entry["target_word"]}

Frames:
{option_lines}

Which of the given frames best represents the meaning of the target word "{entry['target_word']}" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
'''

    return prompt, entry["frame_name"][entry["correct_answer"]]



dataset_files = {
    "FN1.7": "../data/fn1.7/json_output/test.json"
}

for dataset_name, path in dataset_files.items():
    with open(path) as f:
        dataset = json.load(f)
        print(f"\nEvaluating on {dataset_name}.......................")
        type = ['zero_shot', 'few_shot']
        for ty in type:
            for seed in seeds:
                print(f"\nRunning with seed {seed} ----------------------------")
                set_seed(seed)
                count = 0
                missed = 0
                total = 0
                prompt_type = ty
                for entry in dataset:
                    prompt, gold_frame = format_unified_prompt(entry)
                    # print(prompt)
                    print(gold_frame)
                    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)


                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        
                    response = tokenizer.decode(output[0], skip_special_tokens=True)
                    total += 1
                    # print("Model Output:\n", (response))
                    
                    if prompt_type == 'zero_shot':
                        pattern = r'\bassistant\b\s*\{\s*"frame_Name"\s*:\s*"([^"]+)"\s*\}'
                        match = re.search(pattern, response)
                    else:
                        parts = re.split(r'\bassistant\b', response)
                        last_part = parts[-1].strip()
                        print(last_part)
                        # \bassistant\b\s*\
                        pattern = r'{\s*"frame_Name"\s*:\s*"([^"]+)"\s*}'
                        match = re.search(pattern, last_part)
                        
                    if match:
                        frame = match.group(1)
                        print(f"Predicted : {frame}  ;  Gold label : {gold_frame}" )
                        if frame == gold_frame:
                            count += 1
                            print('Correct : ',count)
                    else:
                        missed +=1
                        print(response)
                        print(f"Total Format Missed  : {missed}" )
                        print('Missed : ', missed)
                        
                accuracy = ( count / total ) * 100

                print("Total Accuracy : ", accuracy)
                print("Total Missed : ", missed)
                print("Total Dataset Size : ", total)      