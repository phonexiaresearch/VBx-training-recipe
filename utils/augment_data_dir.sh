name=$1

if [ ! -d "RIRS_NOISES" ]; then
  # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
  wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
  unzip rirs_noises.zip
fi

if [ ! -d "data/musan" ]; then
  if [ ! -d ${musan_dir} ]; then
    echo "Can't find musan directory at ${musan_dir}"
  fi
  # Prepare the MUSAN corpus, which consists of music, speech, and noise suitable for augmentation.
  local/make_musan_16k.sh ${musan_dir} data || exit 1

  # Get the duration of the MUSAN recordings.  This will be used by the script augment_data_dir.py.
  for dname in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${dname}
    mv data/musan_${dname}/utt2dur data/musan_${dname}/reco2dur
  done
fi

# Make a version with reverberated speech
rvb_opts=()
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

frame_shift=0.01
awk -v frame_shift=${frame_shift} '{print $1, $2*frame_shift;}' data/${name}/utt2num_frames > \
  data/${name}/reco2dur

# Make a reverberated version of the SWBD+SRE list.  Note that we don't add any additive noise here.
python3 steps/data/reverberate_data_dir.py \
  "${rvb_opts[@]}" \
  --speech-rvb-probability 1 \
  --pointsource-noise-addition-probability 0 \
  --isotropic-noise-addition-probability 0 \
  --num-replications 1 \
  --source-sampling-rate 16000 \
  data/${name} data/${name}_reverb

cp data/${name}/vad.scp data/${name}_reverb/
utils/copy_data_dir.sh --utt-suffix "-reverb" data/${name}_reverb data/${name}_reverb.new
rm -rf data/${name}_reverb
mv data/${name}_reverb.new data/${name}_reverb

# Augment with musan_noise
python3 steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "10:5" --fg-noise-dir \
  "data/musan_noise" data/${name} data/${name}_noise

# Augment with musan_music
python3 steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "10:7:5" --num-bg-noises "1" \
  --bg-noise-dir "data/musan_music" data/${name} data/${name}_music

# Augment with musan_speech
python3 steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "19:17:15:13:11" --num-bg-noises "3:4:5:6:7" \
  --bg-noise-dir "data/musan_speech" data/${name} data/${name}_babble

# Combine comp, reverb, noise, music, and babble into one directory.
utils/combine_data.sh data/${name}_aug \
  data/${name}_reverb data/${name}_noise \
  data/${name}_music data/${name}_babble

utils/fix_data_dir.sh data/${name}_aug
