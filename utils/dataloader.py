import os
import random
import pickle

import torch
from torch.autograd import Variable

from bert_serving.client import BertClient

from . import utils

class DataLoader(object):
    '''
    Load the datasets preprocessed by `../WinoBias_preprocessing.py`
    Returns train/valid/test data_loaders
    Also generate list of tags to `./WinoBias_Dataset/` directory as `tags.txt`
    '''
    def __init__(self, data_dir, max_len, random_seed=None):
        
        # Load tags and make tag2i and i2tag
        self.tag2i = {}
        self.i2tag = {}
        with open(os.path.join(data_dir, 'tags.txt'), 'r') as f:
            for idx, tag in enumerate(f.read().splitlines()):
                self.tag2i[tag] = idx
                self.i2tag[idx] = tag
        
        self.data_dir = data_dir
        self.max_len = max_len
        self.num_tags = len(self.tag2i)
        self.random_seed = random_seed


    def make_dataset(self, sentences, targets, data):
        # sentences: (list) of sentences(str)
        # targets  : (list) of targets(str)

        tokens = [utils.tokenizer(sentence) for sentence in sentences]
        
        # tokens = [data_size, seq_length]

        bc = BertClient()
        embeddings = bc.encode(tokens, is_tokenized=True) 

        # embeddings = [data_size, bert_max_len, bert_hidden_dim] (numpy array)
        # bert_max_len includes the [CLS] at the beginning and the [SEP] at the last
        # exclude [CLS] and [SEP] embeddings
        # which are set to be zero as in the `BertServer.py` ('-mask_cls_sep')
        embeddings = embeddings[:, 1:-1, :]
        
        target_idxs = []
        for target in targets:
            
            ids = [self.tag2i[tag] for tag in target.split()]
            
            # clpping the sequence with max_len
            if len(ids) >= self.max_len:
                target_idxs.append(ids[:self.max_len])
            
            # add padding index -1 until max_len
            else:
                while len(ids) < self.max_len:
                    ids.append(-1)
                target_idxs.append(ids)

        # targets = [data_size, max_len]

        data['inputs'] = embeddings
        data['targets'] = target_idxs
        data['size'] = len(target_idxs)


    def data_split(self):
        '''
            Return:
                - split_data (dict):
                    [keys] 'train', 'valid', 'test'
                    [values] data(dict) with keys 'sentences', 'targets', and 'size'
        '''

        data_splited = True
        for split in ['test', 'valid', 'train']:
            if not os.path.isfile( os.path.join(self.data_dir, f"{split}.dat") ):
                data_splited = False
                print(f'- split data `{split}.dat` is missing')
    
        # if the train/valid/test data splits are exist
        if data_splited:
            print(f"- load split data (train/valid/test).dat from {self.data_dir}")
            data = {}
            for split in ['test', 'valid', 'train']:
                with open(os.path.join(self.data_dir, split+".dat"), 'rb') as f:
                    data[split] = pickle.load(f)

        # if data splits are NOT exist
        else:                    
            # Load sentences and targets
            sentences_filepath = os.path.join(self.data_dir, 'sentences.txt')
            targets_filepath = os.path.join(self.data_dir, 'targets.txt')
            with open(sentences_filepath, 'r') as f:
                sentences = f.read().splitlines()
            with open(targets_filepath, 'r') as f:
                targets = f.read().splitlines()

            assert len(sentences) == len(targets), "The number of sentences and targets should be the same"

            # Shuffle
            order = list(range(len(sentences)))
            random.seed(self.random_seed)
            random.shuffle(order)
            sentences = [sentences[i] for i in order] 
            targets = [targets[i] for i in order] 

            # split train/valid/test datasets with ratio of 8:1:1
            data = {'test':{}, 'valid':{}, 'train':{}}
            
            test_val_size = len(sentences) // 10
            print('- building test data...')
            self.make_dataset(sentences[:test_val_size], targets[:test_val_size], data['test'])
            print('- building valid data...')
            self.make_dataset(sentences[test_val_size:test_val_size*2], targets[test_val_size:test_val_size*2], data['valid'])
            print('- building train data...')
            self.make_dataset(sentences[test_val_size*2:], targets[test_val_size*2:], data['train'])

            for split in data.keys():
                with open(os.path.join(self.data_dir, f"{split}.dat"), 'wb') as f:
                    pickle.dump(data[split], f)

        return data

    def data_iterator(self, data, batch_size, device, cuda=False, shuffle=False):
        
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.random_seed)
            random.shuffle(order)
        
        n_batch = data['size'] // batch_size + 1

        for i in range(n_batch):
            
            batch_inputs = torch.Tensor([data['inputs'][idx] for idx in order[i*batch_size:(i+1)*batch_size]])
            batch_targets = torch.LongTensor([data['targets'][idx] for idx in order[i*batch_size:(i+1)*batch_size]])

            # shift tensors to GPU if avaliable
            if cuda:
                batch_inputs, batch_targets = batch_inputs.cuda(), batch_targets.cuda()

            # convert them to Variables to record operations in the computational graph
            batch_inputs, batch_targets = Variable(batch_inputs), Variable(batch_targets)

            # return one batch at a time

            # batch_inputs = [batch_size, max_len, bert_hidden_dim]
            # batch_targets = [batch_size, max_len]

            yield batch_inputs.to(device), batch_targets.to(device)