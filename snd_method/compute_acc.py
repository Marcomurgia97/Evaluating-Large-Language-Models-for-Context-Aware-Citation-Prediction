import json
import re
import argparse


def load_file(path):
    return json.load(open(path, 'r', encoding='UTF-8'))


parser = argparse.ArgumentParser(description="parameters for experiments")
parser.add_argument('--k', type=int, required=True, help="K for precision")
parser.add_argument('--path', type=str, required=True, help="path")

args = parser.parse_args()

k = args.k
context = args.sentence
entity = args.entity

def parse_COT(response, k, correct, l_en, c):
    ref = [s for s in response.split() if 'REFERENCE' in s][0:k]
    # ref.reverse()
    # ref = ref[0:k]
    # print(ref)
    logList = []
    if len(ref) > 0:
        for r in ref:
            nums = re.findall("\d+", r)
            if len(nums) > 0:
                num = nums[0]
                if int(num) <= 10:
                    if correct.lower() in l_en[int(num) - 1].lower() and int(num) not in logList:
                        c = c + 1
                        logList.append(int(num))

    return c


path = args.path
data = load_file(path)
cont = 0
contR = 0
contE = 0
for el in data:
    contR = contR + 1

    s = el['response']

    cont = parse_COT(s, k, el['correct'], el['list_of_en'], cont)

print(cont, cont / len(data), print(len(data)), contE)
