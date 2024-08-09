import os
import random

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login
import regex as re
import argparse
login(token=os.environ['READ_HF_TOKEN'])

device = "cuda"  # the device to load the model onto

model_path = 'MarcoMurgia97/Mistral-7B-FT'
parser = argparse.ArgumentParser(description="parameters for experiments")
parser.add_argument('--k', type=int, required=True, help="K for precision")
parser.add_argument('--sentence', type=str, required=True, help="sentence with or without context")
parser.add_argument('--entity', type=str, required=True, help="type of entity required")
parser.add_argument('--path_test_set', type=str, required=True, help="path_test_set")
parser.add_argument('--path_result', type=str, required=True, help="path_result")
args = parser.parse_args()

if args.entity.startswith('titles'):
    correct = 'titles_correct_refe'
elif args.entity.startswith('abstract_'):
    correct = 'abstract_correct_refe'
else:
    correct = 'abstract&title_correct'


tuples = [
    (args.k, args.sentence, correct, args.entity, 'COT')
]
print(tuples)
'''
tuples = [
    (3, 'sentence_no_context', 'titles_correct_refe', 'titles_same_paper', 'COT')
]'''


def load_dataset(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
    except FileNotFoundError:
        print('File does not exist')
        return {}
    except Exception as e:
        print(e)
        return {}
    return dataset


def format_prompt_COT(cits, sentence, t):
    ids = [i for i in range(1, len(cits) + 1)]
    cits_formatted = ["REFERENCE" + str(id) + ': ' + el + "\n\n" for id, el in
                      zip(ids, cits)]

    if 'abs' in t[3]:
        placeholder1 = 'abstracts'
        placeholder2 = 'abstract'
    else:
        placeholder1 = 'titles'
        placeholder2 = 'title'

    prompt = f'''Given a sentence with a placeholder '[cit]' representing a reference to a scholarly paper,
    please evaluate the relevance of several provided citation {placeholder1} and return the {placeholder2} that best fits the
    placeholder. \n\nThe possible references are listed in the following\n\n'''
    prompt += "".join(cits_formatted)
    prompt += f'''Return the reference that best fits the [cit] in the following sentence\n\n'''
    prompt += sentence
    prompt += f'''\n\nPerform the following tasks:\n 1) State the correct reference for the sentence. 
    Consider the topic, key concepts, and overall meaning of the sentence when making your decision.'''

    prompt = prompt.replace('  ', ' ')
    return prompt


def load_model_tokenizer(path_m):
    tok = AutoTokenizer.from_pretrained(path_m)

    '''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    '''
    mod = AutoPeftModelForCausalLM.from_pretrained(
        path_m,  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=True,
    )

    mod.config.use_cache = True

    return mod, tok


def parser(correct, response):
    if correct.lower() in response.lower():
        return True
    return False


def remove_ref(list_of_en, response, type_of_prompt):
    if type_of_prompt == 'COT':
        ref = [s for s in response.split() if 'REFERENCE' in s]
        if len(ref) > 0:
            for r in ref:
                nums = re.findall("\d+", r)
                if len(nums) > 0:
                    num = int(nums[0])
                    if num - 1 < len(list_of_en):
                        refe = list_of_en[num - 1]
                        list_of_en.remove(refe)
                    else:
                        refe = list_of_en[len(list_of_en) - 1]
                        list_of_en.remove(refe)
                    return list_of_en, refe
        else:
            refe = list_of_en[len(list_of_en) - 1]
            list_of_en.remove(refe)
            return list_of_en, refe
    else:
        for c in list_of_en:
            if c.lower() in response.lower():
                list_of_en.remove(c)
                break
        return list_of_en


def citation_suggestor(dataset, model, tokenizer, tu):
    contSentences = 0
    contCorrect = 0
    answers = []
    type_of_prompt = tu[4]
    prompt = ''
    k = tu[0]
    for it in dataset:
        response = ''
        cits = it[tu[3]].copy()
        sentence = it[tu[1]]
        correct = it[tu[2]][0]
        for i in range(0, k):
            prompt = format_prompt_COT(cits, sentence, tu)
            tmpResp = gen_answer(model, tokenizer, prompt)
            cits, ref = remove_ref(cits, tmpResp, type_of_prompt)
            response += f'''<[|{i + 1}|]> {tmpResp}: {ref}\n'''
            if parser(correct, response):
                contCorrect = contCorrect + 1
                break

        el = {'prompt': prompt, 'response': response, 'correct': correct,
              'sentence': sentence, 'list_of_en': it[tu[3]]}
        answers.append(el)
        print(response)
        print('-------')
        print(correct)
        print('-------')

        print(contSentences)
        contSentences = contSentences + 1
        print(contCorrect / contSentences, contCorrect)

    return contCorrect / contSentences, answers


def gen_answer(model, tokenizer, prompt):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=160, do_sample=False)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text'].split("[/INST]", 2)[-1]


if __name__ == '__main__':
    model, tokenizer = load_model_tokenizer(model_path)
    path = args.path_test_set
    dataset = load_dataset(path)
    for tu in tuples:
        acc, list_of_resp = citation_suggestor(dataset, model, tokenizer, tu)
        path_result = args.path_result
        with open(
                path_result,
                'w',
                encoding='utf-8') as f:
            json.dump(list_of_resp, f, indent=2, ensure_ascii=False)
        acc = round(acc, 4) * 100
        print(acc)
