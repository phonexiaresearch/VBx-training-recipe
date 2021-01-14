#!/bin/bash

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

nnet_dir=exp/xvector_nnet
stage=0
train_stage=-1

. ./cmd.sh || exit 1
. ./path.sh || exit 1
set -e
. ./utils/parse_options.sh

vaddir=mfcc
mfccdir=mfcc
fbankdir=fbank
plda_train_dir=data/plda_train
min_len=400
rate=16k
all_data_dir=all_combined

# set directory for corresponding datasets
voxceleb1_path=
voxceleb2_dev_path=
voxceleb_cn_path=


if [ ${stage} -le 1 ]; then
  # prepare voxceleb1, voxceleb2 dev data and voxceleb-cn
  # parameter --remove-speakers removes test speakers from voxceleb1
  # please see script utils/make_data_dir_from_voxceleb.py to adapt it to your needs
  python utils/make_data_dir_from_voxceleb.py --out-data-dir data/voxceleb1 \
    --dataset-name voxceleb1 --remove-speakers local/voxceleb1-test_speakers.txt \
    --dataset-path ${voxceleb1_path} --rate ${rate}
  utils/fix_data_dir.sh data/voxceleb1
  python utils/make_data_dir_from_voxceleb.py --out-data-dir data/voxceleb2 \
    --dataset-name voxceleb2 --format raw --rate ${rate} \
    --dataset-path ${voxceleb2_dev_path} 
  utils/fix_data_dir.sh data/voxceleb2
  python utils/make_data_dir_from_voxceleb.py --out-data-dir data/voxcelebcn \
    --dataset-name voxcelebcn --no-links --rate ${rate} \
    --dataset-path ${voxceleb_cn_path}
  utils/fix_data_dir.sh data/voxcelebcn

  # combine all data into one data directory
  utils/combine_data.sh data/${all_data_dir} data/voxceleb1 data/voxceleb2 data/voxcelebcn
fi


if [ ${stage} -le 2 ]; then
  # in this stage, we compute VAD and prepare features for both clean and augmented audio

  # make mfccs from clean audios (will be only used to compute vad afterwards)
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_${rate}.conf --nj 500 --cmd \
    "${feats_cmd}" data/${all_data_dir} exp/make_fbank ${mfccdir}
  utils/fix_data_dir.sh data/${all_data_dir}

  # compute VAD for clean audio
  local/compute_vad_decision.sh --nj 500 --cmd \
    "${vad_cmd}" data/${all_data_dir} exp/make_vad ${vaddir}
  utils/fix_data_dir.sh data/${all_data_dir}

  # make fbanks from clean audios
  steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank_${rate}.conf --nj 500 --cmd \
    "${feats_cmd}" data/${all_data_dir} exp/make_fbank ${fbankdir}
  utils/fix_data_dir.sh data/${all_data_dir}
  
  # augment directory
  utils/augment_data_dir.sh ${all_data_dir}
  
  # extract features from augmented data
  steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank_${rate}.conf --nj 500 --cmd \
    "${feats_cmd}" data/${all_data_dir}_aug exp/make_fbank ${fbankdir}
  utils/fix_data_dir.sh data/${all_data_dir}_aug

  utils/combine_data.sh data/${all_data_dir}_aug_and_clean data/${all_data_dir}_aug data/${all_data_dir}
fi


name=${all_data_dir}_aug_and_clean
if [ ${stage} -le 3 ]; then
  # Now we prepare the features to generate examples for xvector training.
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 100 --cmd "${train_cmd}" \
    data/${name} data/${name}_with_aug_no_sil exp/${name}_with_aug_no_sil
  utils/fix_data_dir.sh data/${name}_with_aug_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want at least 4s (400 frames) per utterance.
  mv data/${name}_with_aug_no_sil/utt2num_frames data/${name}_with_aug_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/${name}_with_aug_no_sil/utt2num_frames.bak > data/${name}_with_aug_no_sil/utt2num_frames
  utils/filter_scp.pl data/${name}_with_aug_no_sil/utt2num_frames data/${name}_with_aug_no_sil/utt2spk > data/${name}_with_aug_no_sil/utt2spk.new
  mv data/${name}_with_aug_no_sil/utt2spk.new data/${name}_with_aug_no_sil/utt2spk
  utils/fix_data_dir.sh data/${name}_with_aug_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/${name}_with_aug_no_sil/spk2utt > data/${name}_with_aug_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/${name}_with_aug_no_sil/spk2num | utils/filter_scp.pl - data/${name}_with_aug_no_sil/spk2utt > data/${name}_with_aug_no_sil/spk2utt.new
  mv data/${name}_with_aug_no_sil/spk2utt.new data/${name}_with_aug_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/${name}_with_aug_no_sil/spk2utt > data/${name}_with_aug_no_sil/utt2spk

  utils/filter_scp.pl data/${name}_with_aug_no_sil/utt2spk data/${name}_with_aug_no_sil/utt2num_frames > data/${name}_with_aug_no_sil/utt2num_frames.new
  mv data/${name}_with_aug_no_sil/utt2num_frames.new data/${name}_with_aug_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/${name}_with_aug_no_sil
fi


if [ ${stage} -le 4 ]; then
  echo "$0: Getting neural network training egs";
  local/nnet3/xvector/get_egs_but.sh --cmd "$train_cmd" \
    --nj 16 \
    --stage 0 \
    --frames-per-chunk 400 \
    --not-used-frames-percentage 40 \
    --num-archives 1000 \
    --num-diagnostic-archives 1 \
    --num-repeats 10 \
    data/${name}_with_aug_no_sil exp/egs
fi

num_gpus=2
if [ ${stage} -le 5 ]; then
  # set all needed parameters in train.sh script
  # this will start NN training
  ./train.sh /media/ssd-local/profant/exp/egs exp/nnet

  # convert pytoch model to onnx (much faster)
  # if this end with error it should be fine, just check if onnx file is present
  python local/convert_resnet2onnx.py -i exp/nnet/ResNet101_add_margin_embed256_${num_gpus}gpu/models/model_final -o exp/nnet/ResNet101_add_margin_embed256_${num_gpus}gpu/models/model_final.onnx
fi


if [ ${stage} -le 6 ]; then
  # create data directory for training of PLDA
  # randomly pick one only clean or augmented utterance
  mkdir -p data/plda_train
  cp data/${name}_with_aug_no_sil/* data/plda_train
  python local/create_plda_train_dir.py --input-data-dir data/${name}_with_aug_no_sil --output-data-dir data/plda_train
  utils/fix_data_dir.sh data/plda_train

  # split plda_train data dir to how many gpus you are gonna use for extraction
  utils/split_data.sh data/plda_train/ ${num_gpus}
  
  # extract embedding for PLDA training
  # hardcoded path to model, modify if needed
  for i in $(seq 0 $((num_gpus-1)))
  do
    python local/predict.py \
      --model exp/nnet/ResNet101_add_margin_embed256_${num_gpus}gpu/models/model_final.onnx \
      --kaldi-data-dir data/plda_train/split${num_gpus}/$((i+1)) \
      --emb-out-dir exp/xvectors_plda_train_$((i+1)) \
      --gpus ${i} &
  done
  wait
fi


if [ ${stage} -le 7 ]; then
  # train PLDA
  for i in $(seq 0 $((num_gpus-1)))
  do
    cat exp/xvectors_plda_train_$((i+1))/*.txt
  done | python local/train_transform.py --utt2spk data/plda_train/utt2spk --output-h5 exp/transform.h5 | ivector-compute-plda ark:data/plda_train/spk2utt ark,cs:- exp/plda
fi

exit 0
