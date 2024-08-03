import json, argparse

parser = argparse.ArgumentParser(description="parameters for experiments")
parser.add_argument('--k', type=int, required=True, help="K for precision")
parser.add_argument('--path', type=str, required=True, help="path")

args = parser.parse_args()

k = args.k
path = args.path



def load_file(path):
    return json.load(open(path, 'r', encoding='UTF-8'))


def slice_string(s, k):
    index = s.find(f'''<[|{k + 1}|]>''')
    if k == 5 or index == -1:
        return s
    else:
        return s[0:index]


def parser(c, r):
    if c.lower() in r.lower():
        return True
    return False


#path = args.path
data = load_file(path)


contCorrect = 0
contSentences = 0
for el in data:
    correct = el['correct']
    response = slice_string(el['response'], 1)
    print(response)
    print('----')
    print(correct)
    print('----')
    if parser(correct, response):
        contCorrect = contCorrect + 1
    contSentences = contSentences + 1
    print(contCorrect / contSentences)
