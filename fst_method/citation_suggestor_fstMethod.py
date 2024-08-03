import json
import compute_similarity, argparse

Nums = [10]
model = 'sentence-transformers/all-mpnet-base-v2'

parser = argparse.ArgumentParser(description="parameters for experiments")
parser.add_argument('--k', type=int, required=True, help="K for precision")
parser.add_argument('--sentence', type=str, required=True, help="sentence with or without context")
parser.add_argument('--entity', type=str, required=True, help="type of entity required")
parser.add_argument('--path_test_set', type=str, required=True, help="path_test_set")
parser.add_argument('--path_result', type=str, required=True, help="path")

args = parser.parse_args()

if args.entity.startswith('titles'):
    correct = 'titles_correct_refe'
elif args.entity.startswith('abstract_'):
    correct = 'abstract_correct_refe'
else:
    correct = 'abstract&title_correct'

tuples = [
    (args.k, args.sentence, correct, args.entity)
]


def load_dicts(p):
    try:
        cache = open(p, 'r', encoding='UTF-8')
        cache_dict = json.load(cache)
        cache.close()
        return cache_dict
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def load_dataset(p):
    try:
        with open(p, 'r', encoding='utf-8') as file:
            d = json.load(file)
    except FileNotFoundError:
        print('File does not exist')
        return {}
    except Exception:
        print('An error occurred')
        return {}
    return d


def check_similarity(title, list_of_titles):
    cp = compute_similarity.SimilarityComputer(model)
    top_match = cp.get_most_similar(title, list_of_titles)
    return top_match


def get_list_of_tuples(tensor, sentence, correct, cits):
    li = []
    for f, b in zip(tensor[0], tensor[1]):
        t = (sentence, correct, cits[b], f, b)
        li.append(t)
    return sorted(li, key=lambda x: x[3], reverse=True)


def citation_suggestor(dataset, tup):
    contSentences = 0
    contCorrect = 0
    list_of_tuples = []
    print(len(dataset))
    for it in dataset:
        contSentences = contSentences + 1
        sentence = it[tup[1]]
        correct = it[tup[2]][0]
        cits = it[tup[3]]
        k = int(tup[0])
        top_match = check_similarity(sentence, cits)
        list_of_tuples.extend(get_list_of_tuples(top_match, sentence, correct, cits))
        '''for el in list_of_tuples:
            print(el)'''
        print('-----')
        '''while len(list_of_tuples) > 0:
            currSent = list_of_tuples[0][0]
            tmpList = list_of_tuples[0:k]

            for t in tmpList:
                if t[1] == t[2]:
                    contCorrect = contCorrect + 1

            list_of_tuples = [t for t in list_of_tuples if t[0] != currSent]'''
        tmpList = list_of_tuples[0:k]

        for t in tmpList:
            if t[1] == t[2]:
                contCorrect = contCorrect + 1
        list_of_tuples = []
        print(contCorrect / contSentences, contSentences)
    return contCorrect / contSentences


def manage_t(t):
    key = ''
    if 'titles' in t[3]:
        key = 'title'
    elif 'abstract_' in t[3]:
        key = 'abstract'
    elif 'abstract&' in t[3]:
        key = 'abstract&title'
    return key

if __name__ == '__main__':
    path = args.path_test_set
    dataset = load_dataset(path)
    pathOfRes = args.path_result
    result = load_dicts(pathOfRes)
    keyP, keyT = '', ''
    for tu in tuples:
        acc = citation_suggestor(dataset, tu)
        acc = round(acc, 4) * 100
        print(acc)

        if 'same_paper' in tu[3]:
            keyP = 'same_paper'
            keyT = manage_t(tu)
        elif 'other_papers' in tu[3]:
            keyP = 'other_papers'
            keyT = manage_t(tu)

        if tu[0] not in result.keys():
            result[tu[0]] = {tu[1]: {keyP: {keyT: acc}}}
        elif result.get(tu[0]).get(tu[1]) is None:
            result[tu[0]][tu[1]] = {keyP: {keyT: acc}}
        elif result.get(tu[0]).get(tu[1]).get(keyP) is None:
            result[tu[0]][tu[1]][keyP] = {keyT: acc}
        else:
            result[tu[0]][tu[1]][keyP][keyT] = acc
    print(result)

    with open(pathOfRes, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
