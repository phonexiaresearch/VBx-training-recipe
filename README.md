# VBx-training-recipe
Training recipe for VBx repository. For results and more info please see [VBx](https://github.com/BUTSpeechFIT/VBx).

## Installation
This is [Kaldi](https://github.com/kaldi-asr/kaldi) based recipe, using most of the tools from `egs/sre16/v2`, therefore it is required to have Kaldi compiled.

From Kaldi root dir:
```bash
cd egs/sre16
git clone https://github.com/Jamiroquai88/VBx-training-recipe.git
cd VBx-training-recipe
```

For NN training it is also needed to compile binary which converts egs into arks - `utils/nnet3-copy-egs-to-feats.cc`. Please, copy this source code into `KALDI/src/nnet3bin` and compile it.
```bash
cp utils/nnet3-copy-egs-to-feats.cc ../../../src/nnet3bin/
```

Please see `run.sh` script for basic overview, minimal modification is to set correct VoxCeleb directories for your setup [here](https://github.com/phonexiaresearch/VBx-training-recipe/blob/b699b5f1969e08faa7c0869d506b324613ef5c20/run.sh#L28). 

When set, you can run the main script:
```
./run.sh
```


For more details see `run.sh`. If you are familiar with `sre16/v2` recipe, it should be straightforward.

## Citations
In case of using the software please cite:

F. Landini, J. Profant, M. Diez, L. Burget: [*Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks*](https://arxiv.org/abs/2012.14952)

## Contact
If you have questions regarding the training recipe, email jan.j.profant@gmail.com or michal.klco@phonexia.com.
