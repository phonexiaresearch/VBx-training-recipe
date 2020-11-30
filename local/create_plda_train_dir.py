import argparse
import os
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data-dir', required=True)
    parser.add_argument('--output-data-dir', required=True)

    args = parser.parse_args()

    utt2fea = {}
    clean2augs = {}
    with open(os.path.join(args.input_data_dir, 'feats.scp')) as f:
        for line in f:
            key, fea = line.split()
            utt2fea[key] = fea
            clean_key = key.replace('-noise', '').replace('-music', '').replace('-reverb', '').replace('-babble', '')
            if clean_key not in clean2augs:
                clean2augs[clean_key] = []
            clean2augs[clean_key].append(key)
    with open(os.path.join(args.output_data_dir, 'feats.scp'), 'w') as f:
        for key in clean2augs:
            random_idx = random.randint(0, len(clean2augs[key]) - 1)
            random_key = clean2augs[key][random_idx]
            f.write(f'{random_key} {utt2fea[random_key]}{os.linesep}')    
