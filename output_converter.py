import json
import codecs

list = ['train','dev','test']

for i in range(3):
    pdtb_file = codecs.open(list[i]+'/relations.json', encoding='utf8')
    relations = [json.loads(x) for x in pdtb_file]

    for relation in relations:
        data = {
            'DocID': relation["DocID"],
            'Arg1': {
                'TokenList': [x for [_, _, x, _, _] in relation["Arg1"]["TokenList"]]
            },
            'Arg2': {
                'TokenList': [x for [_, _, x, _, _] in relation["Arg2"]["TokenList"]]
            },
            'Connective': {
                'TokenList': [x for [_, _, x, _, _] in relation["Connective"]["TokenList"]]
            },
            'Sense': relation["Sense"],
            'Type': relation["Type"]
        }

        data_file = json.dumps(data)
        # data_indented = json.dumps(d, indent=4)

        with open(list[i]+"/output_relations.json", "a") as f:
            f.write(data_file)
            f.write('\n')
