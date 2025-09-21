import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
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
    option_lines = "\n".join(f"{opt}" for opt in (entry["frame_definitions"]))
    if prompt_type == 'zero_shot':
            prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in FrameNet semantics.

Your task is to identify the most appropriate FrameNet frame that best captures the meaning of a given target word in context.

You will be given:
- A sentence containing the target word.
- Target Word
- A list of frames and its corresponding definitions.

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
- A list of frames and its corresponding definitions.

Your output must be a **single JSON object** in this exact format:
{{"frame_Name": "Intentionally_act"}}

Where:
- "frame_Name" is the exact name of the selected FrameNet frame.

<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Sentence: united states helps uzbekistan secure dangerous nuclear materials : energy agency announces completion of secret uranium transfer back to russia

Target Word:  secure

Frames:
Getting - A Recipient starts off without the Theme in their possession, and then comes to possess it.  Although the Source from which the Theme came is logically necessary, the Recipient and its changing relationship to the Theme is profiled.  'I got two whistles from John.',
Protecting - Some Protection prevents a Danger from harming an Asset.  'The rubber on her sneakers protected her from the mud.'  'The coral reefs shielded the coastline from the worst of the waves.'  'A growing number of sustainable farmers are preserving agricultural variety and protecting biodiversity by raising ?heritage? or ?heirloom? varieties.'  'The man responsible for this entire ordeal cowardly shielded himself from justice with the little boy's struggling body.',
Taking_captive - An Agent begins preventing a Captive from moving despite a tendency or desire on the part of the Captive to do so.  'Authorities recently captured several suspects who were hiding in a remote mountain area.'  'High value target has been quietly and safely secured.',
Being_at_risk - An Asset is in a state where it is exposed to or otherwise liable to be affected by a Harmful_event, which may be metonymically evoked by reference to a Dangerous_entity. Words expressing relative safety (i.e., lack of risk) are also in this frame.  'If you are a farmer, you are at risk for hearing loss caused by noise exposure in your work.'  'No child is secure from the temptation to do what \"my friends\" are doing.'  'Our nation is making a disastrous mistake thinking that we can have security from people - There is only security with people. '  'You don't have security against impersonation unless you have security against eavesdropping.',
Attaching - The Attaching frame covers two situations - a scene in which somebody causes one thing to be physically connected to something else; or a scene in which somebody causes two things to be connected to each other.  In the first, the frame includes an Agent who attaches an Item to a Goal by manipulating a Connector, creating an asymmetric relationship between the Item and the Goal.  In the second, the Agent attaches two Items to each other, where each serves as a Goal for the other, creating a symmetric relationship between the two Items. In both cases, the  Connector remains to bind the two entities (either Item and Goal, or two Items), without creating a new entity.  'The robber tied Harry to the chair.'  'Make sure you bind his wrists as tightly as possible.'

Which of the given frames best represents the meaning of the target word "secure" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{"frame_Name": "Protecting"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Sentence: these shipments of highly enriched uranium spent fuel are part of nnsa 's efforts to make sure this type of material does n't fall into the wrong hands , ' nnsa administrator linton f. brooks said in an april 20 press release .

Target Word:  uranium

Frames:
Substance - This frame concerns internally undifferentiated Substances.   'As their gut bacteria ferment cellulose , they generate methane gas , which escapes from the rear ends of the insects.'  ' The pipes are leaking noxious gas.'  'Some CH-47 \"Bombers\" were equipped to roll-out riot control gas or napalm onto Viet Cong bunker complexes from the rear cargo ramp.'

Which of the given frames best represents the meaning of the target word "uranium" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{"frame_Name": "Substance"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Sentence: there 's a lot of smart sensors that take away the ( guesswork ) of the soldiers , ' said bruce baldwin , the u.s. army chemical school 's lead trainer for new systems. '' ( the olympian online , 20apr06 , christian hill ) ( link )".

Target Word: say

Frames:
Communication - A Communicator conveys a Message to an Addressee; the Topic and Medium of the communication also may be expressed.  This frame includes no specification of the method of communication (speech, writing, gesture, etc.).  This frame and the frames that inherit the general Communication frame can add elaboration to the Medium in a variety of ways (in French, on the radio program, in a letter) or to the Manner of communication (babble, rant, shout, whisper). There are also frames related to Communication that either do not inherit all of the FEs of this frame or do not inherit them in a straightforward manner (such as Conversation, in which Communicator and Addressee alternate roles, and are often expressed by a single, plural NP).  'Let's hope it did n't have time to communicate any of its findings to its overlords.'  NB -  For many words in this domain (those which do not inherently ascribe cognitive/social processes to a Communicator, position a statement within a conversation, or specify Medium in detail), there is a metaphorical usage which maps Communication onto an alternate POV of Becoming_aware, with corresponding maps for frames related to Communication.  This metaphor is most felicitous with Means or Medium subjects, which are mapped onto the Evidence FE of Becoming_aware.  In many cases this metaphor maps Communication onto the Evidence frame, but not always.  Some words from the Communication domain are listed in the Evidence frame since they are deemed to be lexicalized.  'A flick of his gaze toward the doorway told me what he really wanted. (METAPHOR)' 'This painting really speaks to me. INI (MET) ' 'Putting his arm around her protectively achieved nothing but announcing to their captors their vulnerability. (MET)' 'It says a lot that he did n't come back. INI (MET) ',
Text_creation - An Author creates a Text, either written, such as a letter, or spoken, such as a speech,  that contains meaningful linguistic tokens, and may have a particular Addressee in mind.  The Text may include information about its topic, although the latter is not an FE in this frame.  'I penned a letter concerning racism  to Congress.' 'The brothers said not two words to each other.' 'Jot any notes you need below the line in red pen only.',
Statement - This frame contains verbs and nouns that communicate the act of a Speaker to address a Message to some Addressee using language. Instead of (or in addition to) a Speaker, a Medium may also be mentioned. Likewise, a Topic may be stated instead of a Message. A number of the words can be used performatively, such as declare and insist.  'I now declare you members of this Society.'   'She said that I would be late .'  'The announcement about raffle winners came across the loudspeaker.',
Spelling_and_pronouncing - A Speaker realizes a Sign, usually a linguistic one, in some Formal_realization. The Formal_realization may be described as correct or incorrect in a Judgment expression, or it may be partially indicated as a Part_of_form.  Manner may be used as a more general indication of the Formal_realization  'A person doesn't always pronounce a word the way they say  it in context'  'I can't spell. INI INI'  'Stephanopolous spells his name with a \"ph\".'  'On page 87465, you spelled the name of my dog incorrectly as \"Checquers\".'  'In German, the word for 'peace' is written \"Friede\" CNI'



Which of the given frames best represents the meaning of the target word "say" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{"frame_Name": "Statement"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Sentence: the transfer - part of the global threat reduction initiative ( gtri ) , a nonproliferation effort that secures high-risk nuclear and radiological materials and equipment around the world - resulted from a cooperative effort by the united states , uzbekistan , kazakhstan and russia , as well as coordination through the international atomic energy agency ( iaea ) . '' ( news blaze , 21apr06 ) ( link )",


Target Word:  international

Frames:
Foreign_or_domestic_country - A Current_country is identified as matching or not matching the instance of the Discourse_salient_country.  'Foreign sales account for only 20% of IBM's income .',
Political_locales - This frame covers words that name Locations as defined politically, or administratively.  'Berkeley is a wonderful city.'


Which of the given frames best represents the meaning of the target word  "international" in the sentence above?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{"frame_Name": "Political_locales"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Sentence: india and us to advance cooperation to deal with terrorism.

Target Word: deal

Frames:
Quantified_mass - This frame contains transparent nouns (and some adjectives) denoting quantities of a Mass or of Individuals. As opposed to aggregate words such as group, set, these Quantity do not have a status as Wholes on their own.   'He found them in the deluge of papers on his desk.'  Initially, at least, we have annotated both Quantity of Masses and of Individuals in this frame, though we may split the frame along these lines later on.,
Speak_on_topic - A Speaker addresses an Audience on a particular Topic.  The Audience is generally passive, although for many types of address (including academic talks and press conferences), a discussion or question-answer period is virtually always required.  'McNamara addressed congress on the state of affairs of the Department of Defense.',
Be_in_agreement_on_action - Two (or more) people (the Parties, also encodable as Party_1 and Party_2) have an agreement. Both sides are construed as having a commitment to an Obligation--the relationship is understood to be symmetrical or reciprocal. Instead of a specific mention of the Obligation, a Topic expression may be used to indicate the domain covered.  'The US and Russia have an agreement to share early warning data on missile launches.',
Resolve_problem - An Agent resolves an outstanding Problem by finding its solution, explanation, answer, etc.  Alternatively, some non-agentive Cause may resolve the Problem.  The Means, Degree, Time and Place may also be expressed.  'As it turns out, putting all of my eggs in one basket completely solved all of my problems.' 'Students finally worked through all of the difficult problems by working together.',
Make_agreement_on_action - Two (or more) people (the Parties, also encodable as Party_1 and Party_2) negotiate an agreement. Both sides are construed as making a commitment to assume an Obligation--the process is understood to be symmetrical or reciprocal. Instead of a specific mention of the Obligation, a Topic expression may be used to indicate the domain covered.  'So far , we have agreedto cede 100km of coastline to Kiribati . DNI '  'At common law, the parties could make an oral agreement to submit a dispute to arbitration and such oral agreement would be valid.'

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
                print('-----------------------------------------------------------------------------------------------------------------------')