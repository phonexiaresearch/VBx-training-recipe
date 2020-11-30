#!/usr/bin/env python3

# Copyright      2017 Johns Hopkins University (Author: Daniel Povey)
#                2017 Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017 David Snyder
# Apache 2.0

# This script, which is used in getting training examples, decides
# which examples will come from which recordings, and at what point
# during the training.

# You call it as (e.g.)
#
#  allocate_egs.py --min-frames-per-chunk=50 --max-frames-per-chunk=200 \
#   --frames-per-iter=1000000 --num-repeats=60 --num-archives=169 \
#   --num-jobs=24  exp/xvector_a/egs/temp/utt2len.train exp/xvector_a/egs
#
# The program outputs certain things to the temp directory (e.g.,
# exp/xvector_a/egs/temp) that will enable you to dump the chunks for xvector
# training.  What we'll eventually be doing is invoking the following program
# with something like the following args:
#
#  nnet3-xvector-get-egs [options] exp/xvector_a/temp/ranges.1 \
#    scp:data/train/feats.scp ark:exp/xvector_a/egs/egs_temp.1.ark \
#    ark:exp/xvector_a/egs/egs_temp.2.ark ark:exp/xvector_a/egs/egs_temp.3.ark
#
# where exp/xvector_a/temp/ranges.1 contains something like the following:
#
#   utt1  0  1  0   65 0
#   utt1  6  7  160 50 0
#   utt2  ...
#
# where each line is interpreted as follows:
#  <source-utterance> <relative-archive-index> <absolute-archive-index> \
#    <start-frame-index> <num-frames> <spkr-label>
#
# Note: <relative-archive-index> is the zero-based offset of the archive-index
# within the subset of archives that a particular ranges file corresponds to;
# and <absolute-archive-index> is the 1-based numeric index of the destination
# archive among the entire list of archives, which will form part of the
# archive's filename (e.g. egs/egs.<absolute-archive-index>.ark);
# <absolute-archive-index> is only kept for debug purposes so you can see which
# archive each line corresponds to.
#
# For each line of the ranges file, we specify an eg containing a chunk of data
# from a given utterane, the corresponding speaker label, and the output
# archive.  The list of archives corresponding to ranges.n will be written to
# output.n, so in exp/xvector_a/temp/outputs.1 we'd have:
#
#  ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark \
#    ark:exp/xvector_a/egs/egs_temp.3.ark
#
# The number of these files will equal 'num-jobs'.  If you add up the
# word-counts of all the outputs.* files you'll get 'num-archives'.  The number
# of frames in each archive will be about the --frames-per-iter.
#
# This program will also output to the temp directory a file called
# archive_chunk_length which tells you the frame-length associated with
# each archive, e.g.,
# 1   60
# 2   120
# the format is:  <archive-index> <num-frames>.  The <num-frames> will always
# be in the range [min-frames-per-chunk, max-frames-per-chunk].


# We're using python 3.x style print but want it to work in python 2.x.
from __future__ import print_function

import argparse
import os
import random
import sys


def get_args():
    parser = argparse.ArgumentParser(description="Writes ranges.*, outputs.* and archive_chunk_lengths files "
                                                 "in preparation for dumping egs for xvector training.",
                                     epilog="Called by sid/nnet3/xvector/get_egs.sh")
    parser.add_argument("--prefix", type=str, default="",
                        help="Adds a prefix to the output files. This is used to distinguish between the train "
                             "and diagnostic files.")
    parser.add_argument("--num-repeats", type=int, default=10,
                        help="Number of times each speaker repeats within an archive.")
    parser.add_argument("--frames-per-chunk", type=int, default=200,
                        help="Minimum number of frames-per-chunk used for any archive")
    parser.add_argument("--num-archives", type=int, default=-1,
                        help="Number of archives to generate. If don't provided, it will estimated based on"
                             "the all features and other parameters")
    parser.add_argument("--not-used-frames-percentage", type=int, default=-1,
                        help="Maximum allowed not used frames. The program start using frames and when not used "
                             "frames reach to this number, stop archive generation")
    parser.add_argument("--reused-frames-percentage", type=int, default=-1,
                        help="Maximum allowed reused frames. The program start using frames and when reused "
                             "frames reach to this number, stop archive generation. This number can be any value"
                             "grater than zero.")
    parser.add_argument("--use-spk-chunks-in-one-minibatch", type=str, default="no", choices=['yes', 'no'],
                        help="If yes, all of the spk examples in each archive is presented in one or at most two"
                             "consequent minibatches. If no, all speakers examples randomized before creating"
                             "minibatches and we don't care about seeing several examples of a speaker in each"
                             "minibatch. Default value is no.", dest="spk_in_one_minibatch")
    parser.add_argument("--num-jobs", type=int, default=-1,
                        help="Number of jobs we're going to use to write the archives; the ranges.* "
                             "and outputs.* files are indexed by job.  Must be <= the --num-archives option.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Seed for random number generator")
    parser.add_argument("--num-pdfs", type=int, required=True, help="Num pdfs")

    # now the positional arguments
    parser.add_argument("--utt2len-filename", type=str, required=True,
                        help="utt2len file of the features to be used as input (format is: "
                             "<utterance-id> <num-frames>)")
    parser.add_argument("--utt2int-filename", type=str, required=True,
                        help="utt2int file of the features to be used as input (format is: "
                             "<utterance-id> <id>)")
    parser.add_argument("--egs-dir", type=str, required=True,
                        help="Name of egs directory, e.g. exp/xvector_a/egs")

    print(' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = process_args(args)
    return args


def process_args(args):
    if args.num_repeats < 1:
        raise Exception("--num-repeats should have a minimum value of 1")
    if not os.path.exists(args.utt2int_filename):
        raise Exception("This script expects --utt2int-filename to exist")
    if not os.path.exists(args.utt2len_filename):
        raise Exception("This script expects --utt2len-filename to exist")
    if args.frames_per_chunk <= 1:
        raise Exception("--frames-per-chunk is invalid.")
    if args.num_archives < 1 and args.not_used_frames_percentage < 0 and args.reused_frames_percentage < 0:
        raise Exception("Invalid archive generation stopping parameters. One of the arguments "
                        "{--num-archives, --not-used-frames-percentage or --reused-frames-percentage "
                        "should be grater than zero")
    if args.num_jobs > args.num_archives:
        raise Exception("--num-jobs is invalid (must not exceed num-archives)")
    args.spk_in_one_minibatch = args.spk_in_one_minibatch == 'yes'
    return args


def get_utt2len(utt2len_filename):
    print("Starting get_utt2len")
    utt2len = {}
    f = open(utt2len_filename, "r")
    if f is None:
        sys.exit("Error opening utt2len file " + str(utt2len_filename))
    for line in f:
        tokens = line.split()
        if len(tokens) != 2:
            sys.exit("bad line in utt2len file " + line)
        utt2len[tokens[0]] = int(tokens[1])
    f.close()
    return utt2len


# Handle utt2int, create spk2utt, spks
def get_labels(utt2int_filename):
    print("Starting get_labels")
    f = open(utt2int_filename, "r")
    if f is None:
        sys.exit("Error opening utt2int file " + str(utt2int_filename))
    spk2utt = {}
    utt2spk = {}
    for line in f:
        tokens = line.split()
        if len(tokens) != 2:
            sys.exit("bad line in utt2int file " + line)
        spk = int(tokens[1])
        utt = tokens[0]
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = [utt]
        else:
            spk2utt[spk].append(utt)
    f.close()
    return spk2utt, utt2spk
    # Done utt2int


def get_random_chunk_without_replacement(spk, spk2chunk, spk2utt, utt2len, spk2start_idx, frames_per_chunk):
    spk_chunks = spk2chunk[spk]
    spk_num_chunks = len(spk_chunks)
    if spk_num_chunks <= 1:
        chunks = gen_spk_chunks(spk, spk2utt[spk], utt2len, spk2start_idx, frames_per_chunk)
        if len(chunks) > 0:
            # first shuffle the chunks.
            random.shuffle(chunks)
            spk2chunk[spk] = chunks
    i = random.randint(0, spk_num_chunks - 1)
    chunk = spk_chunks.pop(i)
    return chunk


# given an utterance length utt_length (in frames) and two desired chunk lengths
# (length1 and length2) whose sum is <= utt_length,
# this function randomly picks the starting points of the chunks for you.
# the chunks may appear randomly in either order.
def get_random_offset(utt_length, length):
    if length > utt_length:
        sys.exit("code error: length > utt-length")
    free_length = utt_length - length

    offset = random.randint(0, free_length)
    return offset


def gen_spk_chunks(spk, spk_utts, utt2len, spk2start_idx, frames_per_chunk):
    chunks = []
    if spk in spk2start_idx:
        inf = spk2start_idx[spk]
        s = int(float(frames_per_chunk) / inf[0])
        if s <= 10:
            _start_idx = [0]
            spk2start_idx[spk] = [2, [0]]
        else:
            i = 1
            _start_idx = []
            while i * s < frames_per_chunk:
                t = i * s
                i += 1
                if t not in inf[1]:
                    _start_idx.append(t)
                    inf[1].append(t)
            inf[0] *= 2
    else:
        _start_idx = [0]
        spk2start_idx[spk] = [2, [0]]
    for _s_idx in _start_idx:
        for utt in spk_utts:
            _len = utt2len[utt]
            start_idx = _s_idx
            while start_idx < _len - frames_per_chunk:
                chunks.append((utt, start_idx))
                start_idx += frames_per_chunk
    return chunks


def generate_chunks(spk2utt, utt2len, spk2start_idx, frames_per_chunk):
    # allowed-chunk-overlap
    spk2chunks, spk2num_chunks, spk2used_chunks = {}, {}, {}
    for spk in spk2utt.keys():
        chunks = gen_spk_chunks(spk, spk2utt[spk], utt2len, spk2start_idx, frames_per_chunk)
        if len(chunks) > 0:
            # first shuffle the chunks.
            random.shuffle(chunks)
            spk2chunks[spk] = chunks
            spk2num_chunks[spk] = len(chunks)
            spk2used_chunks[spk] = 0
    return spk2chunks, spk2num_chunks, spk2used_chunks


def splitting(args, spk2utt, utt2spk, utt2len, prefix):
    # all_egs contains 2-tuples of the form (utt-id, offset)
    all_egs = []
    frames_per_chunk = args.frames_per_chunk
    spk2start_idx = {}
    spk2chunks, spk2num_chunks, spk2used_chunks = generate_chunks(spk2utt, utt2len, spk2start_idx, frames_per_chunk)
    info_f = open(args.egs_dir + "/temp/" + prefix + "archive_chunk_lengths", "w")
    if info_f is None:
        sys.exit(str("Error opening file {0}/temp/" + prefix + "archive_chunk_lengths").format(args.egs_dir))
    archive_index = 1
    this_num_egs = args.num_repeats * len(spk2chunks)
    all_arks_num_egs = set()
    while True:
        print("Processing archive {0}".format(archive_index))
        print("{0} {1}".format(archive_index, frames_per_chunk), file=info_f)
        this_egs = []
        # if args.spk_in_one_minibatch:
        #     _speakers = list(spk2chunks.keys())
        #     random.shuffle(_speakers)
        #     speakers = []
        #     for spk in _speakers:
        #         speakers += [spk] * args.num_repeats
        # else:
        speakers = args.num_repeats * list(spk2chunks.keys())
        random.shuffle(speakers)
        ark_num_egs = 0
        for n in range(this_num_egs):
            if len(speakers) == 0:
                print("Ran out of speakers for archive {0}".format(archive_index))
                break
            _flag = True
            while _flag:
                try:
                    spk = speakers.pop()
                    chunk = get_random_chunk_without_replacement(spk, spk2chunks, spk2utt, utt2len, spk2start_idx,
                                                                 frames_per_chunk)
                    _flag = False
                except Exception as exp:
                    if archive_index > 85:
                        print(exp)
                    pass
            this_egs.append(chunk)
            spk2used_chunks[spk] += 1
            ark_num_egs += 1
        all_arks_num_egs.add(ark_num_egs)
        all_egs.append(this_egs)
        total_chunks, reused_chunks, not_used_chunks = 0, 0, 0
        for spk in spk2num_chunks.keys():
            num_chunks = spk2num_chunks[spk]
            total_chunks += num_chunks
            used_chunks = spk2used_chunks[spk]
            if used_chunks > num_chunks:
                reused_chunks += used_chunks - num_chunks
            else:
                not_used_chunks += num_chunks - used_chunks
        reused = float(reused_chunks) * 100 / total_chunks
        not_used = float(not_used_chunks) * 100 / total_chunks
        print('Archive # {:4} , reused frames: {:6.2f} % and not-used frames: {:5.2f}'.format(archive_index,
                                                                                              reused, not_used))
        if archive_index >= args.num_archives > 0:
            num_archives = args.num_archives
            break
        elif args.not_used_frames_percentage > 0 and not_used < args.not_used_frames_percentage:
            num_archives = archive_index
            break
        elif reused > args.reused_frames_percentage > 0:
            num_archives = archive_index
            break
        archive_index += 1
    info_f.close()

    if not os.path.exists(os.path.join(args.egs_dir, 'info')):
        os.makedirs(os.path.join(args.egs_dir, 'info'))

    #if len(all_arks_num_egs) > 1:
    #    raise Exception("All ark files should contain same number of egs")

    with open(os.path.join(args.egs_dir, 'info/%sarks_num_egs' % prefix), 'wt') as fid:
        fid.write('%d' % all_arks_num_egs.pop())

    with open(os.path.join(args.egs_dir, 'info/%snum_archives' % prefix), 'wt') as fid:
        fid.write('%d' % num_archives)

    # work out how many archives we assign to each job in an equitable way.
    num_archives_per_job = [0] * args.num_jobs
    for i in range(0, num_archives):
        num_archives_per_job[i % args.num_jobs] = num_archives_per_job[i % args.num_jobs] + 1

    pdf2num = {}
    cur_archive = 0
    for job in range(args.num_jobs):
        this_ranges = []
        this_archives_for_job = []
        this_num_archives = num_archives_per_job[job]

        for i in range(0, this_num_archives):
            this_archives_for_job.append(cur_archive)
            for (utterance_index, offset) in all_egs[cur_archive]:
                this_ranges.append((utterance_index, i, offset))
            cur_archive = cur_archive + 1

        f = open(args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1), "w")
        if f is None:
            sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1))
        for (utterance_index, i, offset) in sorted(this_ranges):
            archive_index = this_archives_for_job[i]
            print("{0} {1} {2} {3} {4} {5}".format(utterance_index,
                                                   i,
                                                   archive_index + 1,
                                                   offset,
                                                   frames_per_chunk,
                                                   utt2spk[utterance_index]),
                  file=f)
            if utt2spk[utterance_index] in pdf2num:
                pdf2num[utt2spk[utterance_index]] += 1
            else:
                pdf2num[utt2spk[utterance_index]] = 1
        f.close()

        f = open(args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1), "w")
        if f is None:
            sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1))
        print(" ".join(
            [str("{0}/" + prefix + "egs_temp.{1}.ark").format(args.egs_dir, n + 1) for n in this_archives_for_job]),
            file=f)
        f.close()

    f = open(args.egs_dir + "/" + prefix + "pdf2num", "w")
    nums = []
    for k in range(0, args.num_pdfs):
        if k in pdf2num:
            nums.append(pdf2num[k])
        else:
            nums.append(0)

    print(" ".join(map(str, nums)), file=f)
    f.close()


def main():
    args = get_args()
    if not os.path.exists(args.egs_dir + "/temp"):
        os.makedirs(args.egs_dir + "/temp")
    random.seed(args.seed)
    utt2len = get_utt2len(args.utt2len_filename)
    spk2utt, utt2spk = get_labels(args.utt2int_filename)
    if args.num_pdfs == -1:
        spks = spk2utt.keys()
        args.num_pdfs = max(spks) + 1

    prefix = ""
    if args.prefix != "":
        prefix = args.prefix + "_"

    splitting(args, spk2utt, utt2spk, utt2len, prefix)

    print("allocate_egs.py: finished generating " + prefix + "ranges.* and " + prefix + "outputs.* files")


if __name__ == "__main__":
    main()
