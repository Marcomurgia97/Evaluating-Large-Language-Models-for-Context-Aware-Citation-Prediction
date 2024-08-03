import json
import random
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


def slice_sentence(s):
    s = s.split('.')
    t = [i for i in s if '[cit]' in i]
    return t[0]


def slice_ab(s, en):
    if 'abs' in en:
        if len(s.split()) > 150:
            s = s.split('.')
            if len(s) > 2:
                return s[0] + '.' + s[1] + '.' + s[2] + '.'
            elif len(s) == 2:
                return s[0] + '.' + s[1] + '.'
            else:
                return s[0] + '.'
        else:
            return s
    else:
        return s


def make_sentence_context(s):
    idc = 0
    splitted = s.split('.')
    if len(splitted) == 1:
        return s

    for id, txt in enumerate(splitted):
        if '[cit]' in txt:
            idc = id
    if idc == 0:
        return splitted[idc] + splitted[idc + 1]
    elif idc == len(splitted) - 1:
        return splitted[idc - 1] + splitted[idc]
    else:
        return splitted[idc - 1] + splitted[idc] + splitted[idc + 1]


def dataset_processor_base(d, en):
    new_data = []
    for el in d:
        for k, v in el.items():
            for i in v:
                cits_formatted = ["\"" + el[en] + "\"" for el in i['references_same_paper']]
                random.shuffle(cits_formatted)
                instr = "Given a sentence with a placeholder '[cit]' representing a reference to a scholarly paper, " \
                        f"please evaluate the relevance of several provided citation {en}s and return the " \
                        f"{en} that best " \
                        "fits " \
                        "the placeholder. Consider the topic, key concepts, and overall meaning of the sentence when " \
                        "making your " \
                        "decision. \n\n"""
                instr += f"{en.capitalize()}s of the citations are: \n\n" + ", ".join(
                    cits_formatted) + "\n\nand the sentence is: \n\n" + "\"" + slice_sentence(i[
                                                                                                  'sentenceWithContext']) + "\""
                prompt = '<s>[INST]' + instr + ' [/INST]' + f' The {en} that best fits the placeholder is ' + \
                         "\"" + i['correct_reference'][0][en] + "\"."' </s>'
                el = {'text': prompt}
                new_data.append(el)
    return new_data


def dataset_processor_COT(d, en):
    new_data = []
    for el in d:
        for k, v in el.items():
            for i in v:
                ids = [i for i in range(1, 11)]
                random.shuffle(i['references_same_paper'])

                cits_formatted = ["REFERENCE" + str(id) + ': ' + el[en] + "\n\n" for id, el in
                                  zip(ids, i['references_same_paper'])]
                # random.shuffle(cits_formatted)
                instr = f'''Given a sentence with a placeholder '[cit]' representing a reference to a scholarly paper,  
                        please evaluate the relevance of several provided citation {en}s and return the {en} that best fits the 
                        placeholder. \n\nThe possible references are listed in the following\n\n'''
                instr += "".join(cits_formatted)
                instr += f'''Return the reference that best fits the [cit] in the following sentence\n\n'''
                instr += i['sentenceWithContext']
                instr += f'''\n\nPerform the following task:\n 1) State the 
                correct reference for the sentence. Consider the topic, key concepts, and overall meaning of the 
                sentence when making your decision.'''
                prompt = '<s>[INST]' + instr + ' [/INST]' + f' The reference that best fits the placeholder is REFERENCE{get_correct(i, cits_formatted, en)}. </s>'
                el = {'text': prompt}
                new_data.append(el)
    return new_data


def get_correct(it, l, en):
    idx = 0
    for el in l:
        idx = idx + 1
        if it['correct_reference'][0][en] in el:
            return idx
    return -1


if __name__ == '__main__':
    path = args.path_dataset
    data = load_dataset(path)
    titles = dataset_processor_COT(data, 'title')
    abstracts = dataset_processor_COT(data, 'abstract')
    titles = titles[250: int(len(titles) / 2)]
    abstracts = abstracts[int(len(abstracts) / 2):]
    titles.extend(abstracts)
    dataset_for_tuning = titles
    random.shuffle(dataset_for_tuning)
    contT, contA = 0, 0
    path_result = args.path_result
    with open(path_result, 'w',
              encoding='utf-8') as f:
        json.dump(dataset_for_tuning, f, indent=2, ensure_ascii=False)

# datasets.Dataset.from_pandas(pd.DataFrame(data=data)).
