from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os, json, re
import argparse

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


def format_prompt_COT(cits, sentence, t, k):
    ids = [i for i in range(1, 11)]
    cits_formatted = ["REFERENCE" + str(id) + ': ' + el + "\n\n" for id, el in
                      zip(ids, cits)]

    if 'abs' in t[3]:
        placeholder1 = 'abstracts'
        placeholder2 = 'abstract'
    else:
        placeholder1 = 'titles'
        placeholder2 = 'title'

    if k > 1:
        k_prec_placeholder = 'top ' + str(t[0])
        placeholder_final_chain = 'top ' + str(t[0]) + ' references'
        fit_placeholder = 'fit'
        placeholder2 = placeholder1

    else:
        placeholder_final_chain = 'correct reference'
        k_prec_placeholder = ''
        fit_placeholder = 'fits'

    prompt = f'''Given a sentence with a placeholder '[cit]' representing a reference to a scholarly paper,  
    please evaluate the relevance of several provided citation {placeholder1} and return the {k_prec_placeholder} {placeholder2} that best {fit_placeholder} the 
    placeholder. \n\nThe possible references are listed in the following\n\n'''
    prompt += "".join(cits_formatted)
    prompt += f'''Return the {k_prec_placeholder} reference that best {fit_placeholder} the [cit] in the following sentence\n\n'''
    prompt += sentence
    prompt += f'''\n\nPerform the following tasks:\n 1) Discuss the sentence and the [cit]\n 2) State the {placeholder_final_chain} for the sentence. Consider the topic, key concepts, and overall meaning of the sentence when making your decision.
'''
    prompt = prompt.replace('  ', ' ')
    return prompt


def parse_COT(correct, cits, response, cont, k):
    ref = [s for s in response.split() if 'REFERENCE' in s][0:k]
    logList = []
    if len(ref) > 0:
        for r in ref:
            nums = re.findall("\d+", r)
            if len(nums) > 0:
                num = nums[0]
                if int(num) <= 10:
                    if correct in cits[int(num) - 1] and int(num) not in logList:
                        cont = cont + 1
                        logList.append(int(num))

    return cont


def citation_suggestor(dataset, mode, cli, tu):
    contSentences = 0
    contCorrect = 0
    answers = []
    k = tu[0]
    for it in dataset:
        cits = it[tu[3]].copy()
        sentence = it[tu[1]]
        correct = it[tu[2]][0]
        prompt = format_prompt_COT(cits, sentence, tu, k)
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        chat_response = cli.chat(
            model=mode,
            messages=messages,
            temperature=0.0
        )
        response = chat_response.choices[0].message.content

        el = {'prompt': prompt, 'response': response, 'correct': correct,
              'sentence': sentence, 'list_of_en': cits}
        answers.append(el)
        print(response)
        print('-------')
        contCorrect = parse_COT(correct, cits, response, contCorrect, k)
        print(contSentences)

        contSentences = contSentences + 1
        print(contCorrect / contSentences)

    return contCorrect / contSentences, answers


if __name__ == '__main__':
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "open-mixtral-8x7b"
    client = MistralClient(api_key=api_key)
    path = args.path_test_set
    dataset = load_dataset(path)
    for tu in tuples:
        acc, list_of_resp = citation_suggestor(dataset, model, client, tu)
        path_result = args.path_result
        with open(
                path_result,
                'w',
                encoding='utf-8') as f:
            json.dump(list_of_resp, f, indent=2, ensure_ascii=False)
        acc = round(acc, 4) * 100
        print(acc)
