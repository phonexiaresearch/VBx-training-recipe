# VBx-training-recipe
Training recipe for VBx repository. @TODO link

## Installation
This is [Kaldi](https://github.com/kaldi-asr/kaldi) based recipe, using most of the tools from `egs/sre16/v2`, therefore it is required to have Kaldi compiled.

From Kaldi root dir:
```bash
cd egs/sre16
git clone https://github.com/Jamiroquai88/VBx-training-recipe.git
cd VBx-training-recipe
```

Please see `run.sh` script for basic overview, minimal modification is to set correct VoxCeleb directories for your setup [here](https://github.com/Jamiroquai88/VBx-training-recipe/blob/028526ef763b63d24bbbc5b2f1fb882c2ceb3581/run.sh#L28). 

When set, you can run the main script:
```
./run.sh
```

For more details see `run.sh`. If you are familiar with `sre16/v2` recipe, it should be straightforward.

When making use of this repository, please cite: @TODO
