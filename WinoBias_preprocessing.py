"""
Preprocessing the WinoBias datasets from https://github.com/uclanlp/corefBias

Arguments:
    --cloned_dir (str) : path to the cloned directory `corefBias/`
    --data_dir   (str) : path to the directory where the `sentences.txt` and `targets.txt` will be saved
    
Output:
    modified WinoBias dataset containing `sentences.txt` and `targets.txt`
    `sentences.txt` file contains the sentences from the original dataset without brackets
        ex) 'The developer argued with the designer because he did not like the design.'
    `targets.txt` file contains the targets corresponding to the sentences in `sentences.txt` file
        ex) 'ENT ENT O O O O O PRN O O O O O'
        where 'ENT' tag means the entity referenced by 'PRN' which is pronoun tag, otherwise tagged by O
"""

import os, argparse
import re
import pickle

from utils import utils

parser = argparse.ArgumentParser()

parser.add_argument('--cloned_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    original_data_path = os.path.join(args.cloned_dir, 'WinoBias/wino/data')
    txt_filenames = ['pro_stereotyped_type1.txt.dev',
                     'pro_stereotyped_type1.txt.test',
                     'pro_stereotyped_type2.txt.dev',
                     'pro_stereotyped_type2.txt.test',
                     'anti_stereotyped_type1.txt.dev',
                     'anti_stereotyped_type1.txt.test',
                     'anti_stereotyped_type2.txt.dev',
                     'anti_stereotyped_type2.txt.test']
    
    print("- Loading the original WinoBias datasets")
    
    lines = []
    for filename in txt_filenames:
        with open(os.path.join(original_data_path, filename), 'r') as f:
            lines += f.read().splitlines()
            
    print("- Processing the loaded datasets")
            
    sentences = []
    targets = []

    # pronouns
    tag_map = {w:"PRN" for w in ['[she]', '[he]', '[her]', '[his]', '[him]']}

    # for occupations
    pattern = re.compile("\[([^]]+)\]")
    
    for l in lines:

        # tokenize the sentence
        tokens = utils.tokenizer(l[:-1])[1:]

        # replace pronoun with the tag PRN
        target = ' '.join([tag_map[w] if w in tag_map else w for w in tokens])
        
        # replace the entity with the tag ENT
        ent_tags = ' '.join(["ENT"] * len(pattern.findall(target)[0].split()))
        target = re.sub(r"\[([^]]+)\]", ent_tags, target)

        # replace all other words to the tag O
        target = ' '.join(["O" if w not in ["PRN","ENT"] else w for w in target.split()])

        # remove '[' and ']' from the sentence
        sentence = ' '.join(l.replace('[', '').replace(']','').split()[1:])
        
        # check whether both PRN and ENT tags are in tagged line
        assert "PRN" in target.split(), f"There's no PRN tag in target(tagged) line '{target}' corresponds to the sentence '{sentence}'"
        assert "ENT" in target.split(), f"There's no ENT tag in target(tagged) line '{target}' corresponds to the sentence '{sentence}'"
        
        # add the sentence and the targets
        sentences.append(sentence.lower())
        targets.append(target)
        
    print(f"- Saving the dataset to {args.data_dir}")
    with open(os.path.join(args.data_dir, "sentences.txt"), 'w') as f:
        f.writelines([s+'\n' for s in sentences])
    with open(os.path.join(args.data_dir, "targets.txt"), 'w') as f:
        f.writelines([t+'\n' for t in targets])

    with open(os.path.join(args.data_dir, 'tags.txt'), 'w') as f:
        f.writelines(["O\n", "ENT\n", "PRN\n"])
    
    print("- Saving done.")
    
    assert len(sentences) == len(targets), f"the number of sentences({len(sentences)}) mismatch with the number of targets({len(targets)})"
    
    print("")
    print("* DATASET INFO ========================================================")
    print(f"- Total {len(sentences)} lines in the dataset")
    print(f"- Maximum sentence length : {max([len( utils.tokenizer(l) ) for l in sentences])}")
    print("- Tags: [ENT] Entity, [PRN] Pronoun, [O] otherwise")
    print("- Example:")
    print(f"     sentence : {sentences[-1]}")
    print(f"     target   : {targets[-1]}")
    print("=======================================================================")
    print("")

    