#!/bin/bash

nnet_dir=exp/xvector_nnet
stage=2
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

voxceleb1_path=/media/marvin/_datasets/transfer/jose/evaluation/VoxCeleb1-16k_01/data
voxceleb2_dev_path=/media/marvin/_datasets/transfer/jose/evaluation/VoxCeleb2-16k_01/data/dev/aac
voxceleb_cn_path=/media/marvin/_riders/jan.profant/tmp/CN-Celeb/data


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
exit 0

if [ ${stage} -le 2 ]; then
  # make mfccs from clean audios (will be only used to compute vad afterwards)
  steps/make_mfcc.sh --write-utt2num-frames true --fbank-config conf/mfcc_${rate}.conf --nj 500 --cmd \
    "${feats_cmd}" data/${all_data_dir} exp/make_fbank ${mfccdir}
  utils/fix_data_dir.sh data/${all_data_dir}

  # compute VAD for clean audio
  local/compute_vad.sh --nj 500 --cmd \
    "${vad_cmd}" data/${all_data_dir} exp/make_vad ${vaddir}
  utils/fix_data_dir.sh data/${all_data_dir}
  
  # make fbanks from clean audios
  steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank_${rate}.conf --nj 500 --cmd \
    "${feats_cmd}" data/${all_data_dir} exp/make_fbank ${fbankdir}
  utils/fix_data_dir.sh data/${all_data_dir}
  
  # augment directory
  utils/augment_data_dir.sh ${all_data_dir}
  
  # extract features from augmented data
  steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf --nj 500 --cmd \
    "${feats_cmd}" data/${all_data_dir}_aug exp/make_fbank ${fbankdir}
  utils/fix_data_dir.sh data/${all_data_dir}_aug

  utils/combine_data.sh data/${all_data_dir}_aug_and_clean data/${all_data_dir}_aug data/${all_data_dir}
fi


name=${all_data_dir}_aug_and_clean
# Now we prepare the features to generate examples for xvector training.
if [ ${stage} -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 100 --cmd "${train_cmd}" \
    data/${name} data/${name}_with_aug_no_sil ${fbankdir}/../exp/${name}_with_aug_no_sil
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
    --num-repeats 5 \
    data/${name}_with_aug_no_sil exp/egs
fi

exit 0


if [ ${stage} -le 5 ]; then
  # make mfccs from clean audios
  #steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf --nj 500 --cmd \
  #  "${feats_cmd}" ${plda_train_dir} exp/make_fbank ${fbankdir}
  #utils/fix_data_dir.sh ${plda_train_dir}

  # compute VAD
  #sid/compute_vad_decision_BUT.sh --nj 500 --cmd \
  #  "${vad_cmd}" ${plda_train_dir} exp/make_vad ${vaddir}
  utils/fix_data_dir.sh ${plda_train_dir}

  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 100 --cmd "${train_cmd}" \
    ${plda_train_dir} ${plda_train_dir}_with_aug_no_sil ${fbankdir}/../exp/plda_train__with_aug_no_sil
  utils/fix_data_dir.sh ${plda_train_dir}
fi

exit 0
