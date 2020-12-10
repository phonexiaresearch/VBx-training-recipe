#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import argparse
import csv
import os


def add_file(args, speaker, video_link, video_chunk, video_chunk_path):
    if args.format == 'wav':
        if not video_chunk_path.endswith('.wav'):
            return
        utt = os.path.join(args.dataset_name, speaker, video_link, video_chunk).replace('.wav', '')
    elif args.format == 'raw':
        if not video_chunk_path.endswith('.raw'):
            return
        utt = os.path.join(args.dataset_name, speaker, video_link, video_chunk).replace('.raw', '')
    utt2spk[utt] = f'{args.dataset_name}/{speaker}'
    if args.format == 'wav':
        utt2physical[utt] = video_chunk_path
    elif args.format == 'raw':
        utt2physical[utt] = f' sox -e signed -t raw -b 16 -c 1 -r {args.rate} {video_chunk_path} -r {args.rate} -t wav -'    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True, help='name/prefix of dataset')
    parser.add_argument('--dataset-path', type=str, required=True, help='path to datasets directory')
    parser.add_argument('--out-data-dir', type=str, required=True, help='path to output data directory in kaldi format')
    parser.add_argument('--format', choices=['wav', 'raw'], default='wav')
    parser.add_argument('--no-links', action='store_true', default=False, help='use for directories without YT links structure')
    parser.add_argument('--rate', choices=['8k', '16k'], default='8k', help='bitrate of audios')
    parser.add_argument('--remove-speakers', required=False, help='path with a list of speaker to remove (voxceleb1 test, ...)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.out_data_dir):
        os.makedirs(args.out_data_dir)

    utt2spk, utt2physical = {}, {}

    speakers_to_remove = []
    if args.remove_speakers:
        with open(args.remove_speakers) as f:
            for line in f:
                speakers_to_remove.append(line.rstrip())

    for speaker in os.listdir(args.dataset_path):
        speaker_dir = os.path.join(args.dataset_path, speaker)
        if len(speakers_to_remove) > 0:
            if speaker in speakers_to_remove:
                continue
        assert os.path.isdir(speaker_dir)
        for video_link in os.listdir(speaker_dir):
            if args.no_links:
                video_chunk_path = os.path.join(speaker_dir, video_link)
                add_file(args, speaker, '', video_link, video_chunk_path)        
            else:
                video_link_dir = os.path.join(speaker_dir, video_link)
                if not os.path.isdir(video_link_dir):
                    continue
                for video_chunk in os.listdir(video_link_dir):
                    video_chunk_path = os.path.join(video_link_dir, video_chunk)
                    add_file(args, speaker, video_link, video_chunk, video_chunk_path)
    with open(os.path.join(args.out_data_dir, 'wav.scp'), 'w') as f:
        for utt in sorted(utt2spk):
            f.write(f'{utt}{" cat " if args.format == "wav" else ""}{utt2physical[utt]} |\n')

    with open(os.path.join(args.out_data_dir, 'utt2spk'), 'w') as f:
        for utt in sorted(utt2spk):
            f.write(f'{utt} {utt2spk[utt]}\n')
