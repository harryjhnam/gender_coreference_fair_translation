import re
import os

import torch

def tokenizer(sentence):
    tokens = []
    for match in re.finditer(r'[^.,?!\s]+|[.,?!]', sentence):
        tokens.append(match.group())
    # split the sentence by spaces and punctuations
    return tokens

class running_avg_accuracy(object):
    def __init__(self):
        self.n_items = 0
        self.n_corrects = 0
        
    def update(self, preds, targets):

        # targets = [batch_size, max_len]
        # preds = [batch_size, max_len, num_tags]
        
        num_tags = preds.size(2)

        flat_targets = targets.view(-1)
        flat_preds = torch.argmax(preds.view(-1, num_tags), dim=1)

        # flat_targets / preds = [batch_size*max_len]

        for i in range(len(flat_targets)):

            t = int(flat_targets[i].detach().clone())
            p = int(flat_preds[i].detach().clone())

            if t == -1:
                continue
            else:
                self.n_items += 1
                self.n_corrects += (t==p)

    def __call__(self):
        return self.n_corrects / self.n_items


def save_checkpoint(state, model_dir, model_path):
    
    model_dir = os.path.join(model_dir, state['comment'])

    if not os.path.exists(model_dir):
        print(f"- Checkpoint directory does not exists! making directory {model_dir}")
        os.mkdir(model_dir)
    else:
        print(f"- Checkpoint Directory exists!")

    torch.save(state, model_path)


def load_checkpoint(ckpt, model, optimizer=None):
    
    if not os.path.exists(ckpt):
        raise ValueError(f"File doesn't exists {ckpt}")
    
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def load_tag_dict(data_dir):
    with open(os.path.join(data_dir, 'tags.txt'), 'r') as f:
        i2tag = {}
        for idx, tag in enumerate(f.read().splitlines()):
            i2tag[idx] = tag
    return i2tag