// nnet3bin/nnet3-copy-egs-to-feats.cc

// Copyright      2019  Shuai Wang
//                2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-example.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Transform the egs file format (prepared for nnet training) into \n"
        "Normal feature format\n"
        "\n"
        "Usage:  nnet3-copy-egs-to-feats [options] <egs-rspecifier> <feats-wspecifier>\n"
        "\n"
        "nnet3-copy-egs-to-feats ark:train.egs ark:feats.ark\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
        feats_wspecifier = po.GetArg(2);

    int64 num_done = 0;

    std::vector<std::pair<std::string, NnetExample*> > egs;

    SequentialNnetExampleReader example_reader(examples_rspecifier);

    BaseFloatMatrixWriter feats_writer(feats_wspecifier);

    for (; !example_reader.Done(); example_reader.Next())
        egs.push_back(std::make_pair(example_reader.Key(),
                           new NnetExample(example_reader.Value())));

    for (size_t i = 0; i < egs.size(); i++) {
      if (egs[i].second != NULL) {
        Matrix<BaseFloat> output(egs[i].second->io.front().features.NumRows(), egs[i].second->io.front().features.NumCols());
        egs[i].second->io.front().features.GetMatrix(&output);
        feats_writer.Write(egs[i].first, output);

        delete egs[i].second;
        num_done++;
      }
    }

    KALDI_LOG << "Converted " << num_done
              << " neural-network training examples "
              << " into normal feature archive files";

    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
