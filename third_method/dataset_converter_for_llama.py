import json
import argparse

parser = argparse.ArgumentParser(description="parameters for experiments")
parser.add_argument('--path_dataset', type=str, required=True, help="path_dataset")
parser.add_argument('--path_result', type=str, required=True, help="path_result")
args = parser.parse_args()

def load_dataset(p):
    try:
        with open(p, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
    except FileNotFoundError:
        print('File does not exist')
        return {}
    except Exception as e:
        print(e)
        return {}
    return dataset

path = args.path_dataset
dataset = load_dataset(path)
newData = []
for el in dataset:
    '''
    <|start_header_id|>system<|end_header_id|>

    {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {{ input }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {{ output }}<|eot_id|>
    '''
    prompt = el['text']
    instr = prompt[prompt.find('[INST]')+len('[INST]'):prompt.rfind('[/INST]')]
    answer = prompt[prompt.find('[/INST]')+len('[/INST]'):prompt.rfind('</s>')]
    prompt_LLama3 = f'''<|start_header_id|>system<|end_header_id|>You are a researcher writing an article 
    <|eot_id|><|start_header_id|>user<|end_header_id|> 
{instr}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>'''
    newData.append({'text': prompt_LLama3})

path_result = args.path_result
with open(path_result, 'w',
          encoding='utf-8') as f:
    json.dump(newData, f, indent=2, ensure_ascii=False)
