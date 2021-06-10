import os, sys
import argparse
from timeit import default_timer as timer

import torch
import finetune_layer
from utils import utils

from bert_serving.client import BertClient
from BertServer_setting.BertServer import start_BertServer

from flask import Flask, request, jsonify
import requests, json

# Define the App
app = Flask(__name__)


'''
Arguments
'''
parser = argparse.ArgumentParser()

# GPU ID for inferencing
parser.add_argument('--GPU_ID', type=int)

# Translation APIs choices
parser.add_argument('--translation_api', type=str, choices=['naver', 'kakao', 'google'])
parser.add_argument('--translation_api_keys', type=str, default='./API_keys.json')

# BertServer arguments
parser.add_argument('--bert_model_dir', type=str, default='./temp/uncased_L-24_H-1024_A-16/',
                    help='path to the pretrained model')

parser.add_argument('--port', type=str, default='5555', 
                    help='port for pushing data from client to server')
parser.add_argument('--port_out', type=str, default='5556',
                    help='port for publishing results from server to client')
parser.add_argument('--max_seq_len', type=str, default='24')

parser.add_argument('--bert_hidden_dim', type=int, default=1024)

# Finetune layer arguments
parser.add_argument('--finetune_layer_dir', type=str, default='./experiments/ckpts')
parser.add_argument('--data_dir', type=str, default='./WinoBias_Dataset/processed',
                    help='directory including `tags.txt`')

parser.add_argument('--layer_hidden_dims', type=int, nargs='+', default=[1024, 256, 128, 16, 3])
parser.add_argument('--max_len', type=int, default=22)

parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--weight_decay', type=float, default=0)

# server arugments
parser.add_argument('--host', type=str)

# Parse arguments
args = parser.parse_args()


'''
- Start BertServer
'''
start_BertServer(args)
bc = BertClient()

'''
= Define and load the fintune layer
'''
# set device
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device(f"cuda:{args.GPU_ID}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# set model path
comment = f'BATCH{args.train_batch_size}_HIDDENS_{"_".join([str(i) for i in args.layer_hidden_dims])}_LR{args.learning_rate}_WD{args.weight_decay}'
ckpt_path = os.path.join(args.finetune_layer_dir, f'{comment}/model.best.ckpt')

# Define model and load the trained parameters
print("- Load Model ...")
start = timer()
model = finetune_layer.finetune_layer(args.layer_hidden_dims, args.bert_hidden_dim, num_tags=3)
checkpoint = utils.load_checkpoint(ckpt_path, model)
model.to(device)
model.eval()
end = timer()
print("- Loading model completed ({:04.3f}s elapsed)".format(end-start))


'''
- Translation by using APIs
    [Input] : (str) Korean sentences
    [Output]: (list) of (str) translated english sentences
'''
with open(args.translation_api_keys, 'rb') as f:
    keys = json.load(f)

def naver(kor_input):
    url = 'https://openapi.naver.com/v1/papago/n2mt'
    headers = {"X-Naver-Client-Id" : keys['NAVER']['ClientID'], 
            "X-Naver-Client-Secret" : keys['NAVER']['ClientSecret']}
    data = {"source" : "ko",
            "target" : "en",
            "text" : kor_input }
    response = requests.post(url, headers=headers, data=data)
    translated_text = response.json()['message']['result']['translatedText']
    
    eng_sentences = [s+"." if s[-1]!="." else s for s in translated_text.split('. ')]

    return eng_sentences

def kakao(kor_input):

    url = 'https://dapi.kakao.com/v2/translation/translate'
    headers={ "Authorization" : f"KakaoAK {keys['KAKAO']['REST_API']}"}
    data = {"src_lang" : "kr",
            "target_lang" : "en",
            "query" : kor_input }
    response = requests.post(url, headers=headers, data=data)
    translated_text = response.json()['translated_text'] # List of paragraphs
    
    eng_sentences = []
    for paragraph in translated_text:
        for p in paragraph:
            eng_sentences += [s+"." if s[-1]!="." else s for s in p.split('. ')]

    return eng_sentences

from google_trans_new import google_translator
def google(kor_input):
    translator = google_translator()
    translated_text = translator.translate(kor_input, lang_src="ko", lang_tgt="en")

    eng_sentences = [s+"." if s[-1]!="." else s for s in translated_text.split('. ')]

    return eng_sentences


'''
- Server App
    [Request]
    [Response]
'''
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        korean_input = request.json['korean_sentence']
        eng_sentences = getattr(sys.modules[__name__], args.translation_api)(korean_input)
        
        # tokenize the translated sentence
        tokens = [utils.tokenizer(sentence) for sentence in eng_sentences]
        
        # get embeddings from BertServer
        embeddings = torch.Tensor(bc.encode(tokens, is_tokenized=True)).to(device)
        embeddings = embeddings[:,1:-1,:] # exclude [CLS] and [SEP] tokens from embeddings

        # predict the gender-ambiguous translations
        preds = model(embeddings)
        pred_idxs = torch.argmax(preds, dim=2)

        # convert them to tags
        i2tag = utils.load_tag_dict(args.data_dir)
        tags = []
        for idxs in pred_idxs:
            tags.append( [i2tag[int(p)] for p in idxs] )

    return jsonify({'translated_text': eng_sentences, 'translated_tokens': tokens, 'ambiguity_tags': tags})


if __name__ == '__main__':
    app.run(host = args.host)