import requests

import argparse

from PyInquirer import style_from_dict, Token, prompt


style = style_from_dict({
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})

replce_map = {'she': {'남성':'he', '여성':'she'},
              'he'  : {'남성':'he', '여성':'she'},
              'her' : {'남성':'him', '여성':'her'},
              'him' : {'남성':'him', '여성':'her'},
              'hers': {'남성':'his', '여성':'hers'},
              'his' : {'남성':'his', '여성':'hers'}}


parser = argparse.ArgumentParser()

parser.add_argument('--host', type=str)
parser.add_argument('--port', type=str)


def color_sentence(sentence, color_code):
    END = "\033[0m"
    return color_code+sentence+END

def color_words(words, ids, color_code):
    END = "\033[0m"
    colored = []
    for i, w in enumerate(words):
        if i in ids:
            colored.append("\033[4m"+color_code+w+END+END)
        else:
            colored.append(w)
    return ' '.join(colored)

def gender_ambiguous_adjustment(args):
    print("")
    kor_input = str(input(color_sentence("> 번역할 한국어 문장을 입력해주세요: ", '\033[92m')))


    resp = requests.post(f"http://{args.host}:{args.port}/predict",
                        json={'korean_sentence':kor_input})

    translated_texts = resp.json()['translated_text']
    translated_tokens = resp.json()['translated_tokens']
    ambiguity_tags = resp.json()['ambiguity_tags']

    # translated_text = [num_sentences, num_words]
    # translated_tokens = [num_sentences]
    # ambiguity_tags = [num_sentences, max_len]

    is_ambiguous = False

    ent_idx, prn_idx = {}, {}

    tokens = []
    tags = []

    for sent_id in range(len(translated_texts)):
        
        tokens.append(translated_tokens[sent_id])
        tags.append(ambiguity_tags[sent_id])
        
        if ('ENT' in tags[sent_id]) and ('PRN' in tags[sent_id]):
            is_ambiguous = True
            
            ent_ids = []
            while 'ENT' in tags[sent_id]:
                id = tags[sent_id].index('ENT')
                ent_ids.append(id)
                tags[sent_id][id] = '_ENT'

            prn_ids = []
            while 'PRN' in tags[sent_id]:
                id = tags[sent_id].index('PRN')
                prn_ids.append(id)
                tags[sent_id][id] = '_PRN'

            ent_idx[sent_id] = ent_ids
            prn_idx[sent_id] = prn_ids


    if is_ambiguous:
        print("")
        colored_sentences = []
        for sent_id in ent_idx.keys():
            colored_sentences.append( color_words(tokens[sent_id], ent_idx[sent_id] + prn_idx[sent_id], "\033[91m") )

        print(color_sentence('> 영어로 번역된 문장입니다: ', '\033[92m'), ' '.join(colored_sentences))
        print("")
        print(color_sentence("> 성별이 모호한 단어가 감지되었습니다.", '\033[92m'))

        # ask the gender of ambiguous words
        answers = {}
        for sent_id in ent_idx.keys():
            embiguous_entity = ' '.join([tokens[sent_id][id] for id in ent_idx[sent_id]])
            question = [{'type': 'checkbox',
                        'message': f'단어 {embiguous_entity}의 성별을 골라주세요.',
                        'name': 'gender',
                        'choices': [{'name':'여성'}, {'name':'남성'}, {'name':'해당없음'}]}]

            print("")
            answer = prompt(question, style=style)

            while len(answer['gender']) != 1:
                print("")
                print(color_sentence("> 한 가지를 선택해주세요.", '\033[93m'))
                answer = prompt(question, style=style)

            answers[sent_id] = answer['gender'][0]

        # replace the coreferenced pronoun
        for sent_id in prn_idx.keys():
            original_pronoun = tokens[sent_id][prn_idx[sent_id][0]]
            tokens[sent_id][prn_idx[sent_id][0]] = replce_map[original_pronoun][answers[sent_id]]

        translated_sentences = []
        for s in tokens:
            translated_sentences.append(' '.join(s[:-1]))
            translated_sentences += ['.']
        
        print("")
        print(color_sentence('> 응답에 맞게 번역이 수정되었습니다.', '\033[92m'))
        print("")
        print(color_sentence('> 영어로 번역된 문장입니다: ', '\033[92m'), ' '.join(translated_sentences))

    else:
        print("")
        print(color_sentence('> 영어로 번역된 문장입니다: ', '\033[92m'), ' '.join(translated_texts))

    print("")


if __name__ == "__main__":

    args = parser.parse_args()

    while True:
        gender_ambiguous_adjustment(args)
    