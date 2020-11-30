import os
import queue
import struct
import subprocess
from threading import Thread

import kaldi_io

import numpy as np

from torch.utils.data import Dataset


class KaldiArkLoaderOld(object):

    def __init__(self, ark_file, queue_size=1280):
        self._ark_file = ark_file
        self._queue = queue.Queue(queue_size)
        self._reading_finished = False
        self._killed = False
        self._thread = Thread(target=self.__load_data)
        self._thread.daemon = True
        self._thread.start()

    @staticmethod
    def __read_token(fid):
        token = ''
        while True:
            char = fid.read(1)
            if char == '' or char == ' ':
                break
            token += char
        return token

    @staticmethod
    def __expect_token(fid, expected_token):
        token = fid.read(len(expected_token)).decode()
        assert token == expected_token

    @staticmethod
    def __read_index_vector(fid):
        def read_index(fid, prev_index=None):
            c = struct.unpack('b', fid.read(1))[0]
            if prev_index is None:
                if abs(c) < 125:
                    n = x = 0
                    t = int(c)
                else:
                    assert c == 127
                    _, n, _, t, _, x = struct.unpack('=bibibi', fid.read(15))
            else:
                if abs(c) < 125:
                    n, t, x = prev_index[0], prev_index[1] + c, prev_index[2]
                else:
                    assert c == 127
                    _, n, _, t, _, x = struct.unpack('=bibibi', fid.read(15))
            return n, t, x

        KaldiArkLoaderOld.__expect_token(fid, "<I1V> ")
        _, size = struct.unpack('=bi', fid.read(5))
        prev_index = None
        for i in range(size):
            prev_index = read_index(fid, prev_index)

    def __load_data(self):
        with kaldi_io.open_or_fd(self._ark_file, 'rb') as fid:
            key = KaldiArkLoaderOld.__read_token(fid)
            cnt = 0
            while key:
                # print(key)
                binary = fid.read(2).decode()
                assert binary == '\0B'
                KaldiArkLoaderOld.__expect_token(fid, "<Nnet3Eg> ")
                KaldiArkLoaderOld.__expect_token(fid, "<NumIo> ")
                _, examples_count = np.frombuffer(fid.read(5), dtype='int8,int32', count=1)[0]
                assert examples_count == 2
                KaldiArkLoaderOld.__expect_token(fid, "<NnetIo> ")
                KaldiArkLoaderOld.__expect_token(fid, "input ")
                KaldiArkLoaderOld.__read_index_vector(fid)
                # read mat (i.e. example, a 2D matrix) here
                mat = kaldi_io.read_mat_binary(fid)
                KaldiArkLoaderOld.__expect_token(fid, "</NnetIo> ")
                # output
                KaldiArkLoaderOld.__expect_token(fid, "<NnetIo> ")
                KaldiArkLoaderOld.__expect_token(fid, "output ")
                KaldiArkLoaderOld.__read_index_vector(fid)
                sparse_lab = kaldi_io.read_mat_binary(fid)
                assert len(sparse_lab.indices) == 1
                # read speaker label here
                label = sparse_lab.indices[0]
                KaldiArkLoaderOld.__expect_token(fid, "</NnetIo> ")
                KaldiArkLoaderOld.__expect_token(fid, "</Nnet3Eg> ")
                # put the mat and label in the queue
                self._queue.put((mat, label, key))
                cnt += 1
                # print('Queue size: %d' % self._queue.qsize())
                if self._killed:
                    # print('Killing the thread')
                    # stop reading from this file
                    break
                # if cnt % 128 == 0:
                #     print('Read count: %d, Queue size: %d' % (cnt, self._queue.qsize()))
                key = KaldiArkLoaderOld.__read_token(fid)

        print('Reading finished')
        self._reading_finished = True

    def next(self, timeout=30):
        if self._reading_finished and self._queue.empty():
            return None, None, None
        next_element = self._queue.get(block=True, timeout=timeout)
        # print('Queue size: %d' % self._queue.qsize())
        return next_element

    def stop_reading(self):
        print('stop_reading was called')
        self._killed = True
        # here we just removing unprocessed items from the queue
        # to make garbage collector happy.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break


class KaldiArkLoader(object):

    def __init__(self, ark_file, apply_cmn, queue_size=1280):
        self._ark_file = ark_file
        self._queue = queue.Queue(queue_size)
        self._reading_finished = False
        self._killed = False
        self._apply_cmn = apply_cmn
        self._thread = Thread(target=self._load_data)
        self._thread.daemon = True
        self._thread.start()

    def _load_data(self):
        def read_token(_fid):
            token = ''
            while True:
                char = _fid.read(1).decode('utf-8')
                if char == '' or char == ' ':
                    break
                token += char
            return token

        def cleanup(_process, _cmd):
            ret = _process.wait()
            if ret > 0:
                raise Exception('cmd %s returned %d !' % (_cmd, ret))
            return

        cmn_command = ""
        if self._apply_cmn:
            cmn_command = " | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:-"
        cmd = "nnet3-copy-egs-to-feats --print-args=false ark:{} ark:-{}".format(self._ark_file, cmn_command)
        fh = open("/dev/null", "w")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=fh)
        cleanup_thread = Thread(target=cleanup, args=(process, cmd))
        cleanup_thread.daemon = True
        cleanup_thread.start()
        fid = process.stdout

        key = read_token(fid)
        cnt = 0
        # note: here for speed we assume that the size if 4 bytes, so we do not check others
        while key:
            # print(key)
            _, _, _, rows, _, cols = struct.unpack('=bibibi', fid.read(15))
            buf = fid.read(rows * cols * 4)
            vec = np.frombuffer(buf, dtype=np.float32)
            mat = np.reshape(vec, (rows, cols))
            # extract speaker label from key
            label = int(key.split("-")[-1])
            self._queue.put((mat, label, key))
            cnt += 1
            # self._queue.append((mat, label, key))
            if self._killed:
                # stop reading from this file
                break
            key = read_token(fid)
        fid.close()
        fh.close()
        self._reading_finished = True
        # print('Reading finished. Count: %d' % cnt)

    def next(self, timeout=900):
        if self._reading_finished and self._queue.empty():
            return None, None, None
        next_element = self._queue.get(block=True, timeout=timeout)
        # print('Queue size: %d' % self._queue.qsize())
        return next_element

    def stop_reading(self):
        # print('stop_reading was called')
        self._killed = True
        # here we just removing unprocessed items from the queue
        # to make garbage collector happy.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break


class KaldiArkDataset(Dataset):
    def __init__(self, egs_dir, num_archives, num_workers, rank, num_examples_in_each_ark, apply_cmn,
                 finished_iterations=0, processed_archives=0, prefix=''):
        super(KaldiArkDataset, self).__init__()
        self.data_loader = None
        self.egs_dir = egs_dir
        self.num_archives = num_archives
        self.num_workers = num_workers
        self.rank = rank
        self.apply_cmn = apply_cmn
        self.finished_iterations = finished_iterations
        self.processed_archives = processed_archives
        self.num_examples_in_each_ark = num_examples_in_each_ark
        self.length = num_workers * num_examples_in_each_ark
        self.prefix = prefix

    def __getitem__(self, idx):
        # read the next example from the corresponding loader
        mat, spk_id, name = self.data_loader.next()
        return mat, spk_id

    def __len__(self):
        return self.length

    def set_iteration(self, iteration):
        assert iteration > self.finished_iterations
        iteration -= self.finished_iterations
        ark_idx = (self.processed_archives + (iteration - 1) *
                   self.num_workers + self.rank) % self.num_archives + 1
        ark_file = os.path.join(self.egs_dir, f'{self.prefix}egs.{ark_idx}.ark')
        assert os.path.isfile(ark_file), f'Path to ark with egs `{ark_file}` not found.'
        if self.data_loader is not None:
            # first stop the running thread
            self.data_loader.stop_reading()
        self.data_loader = KaldiArkLoader(ark_file, self.apply_cmn)


class KaldiArkDatasetOld(Dataset):
    def __init__(self, egs_dir, num_archives, num_workers, finished_iterations=0,
                 processed_archives=0, num_examples_in_each_ark=107190):
        super(KaldiArkDatasetOld, self).__init__()
        self.data_loaders = [None] * num_workers
        self.egs_dir = egs_dir
        self.num_archives = num_archives
        self.num_workers = num_workers
        self.finished_iterations = finished_iterations
        self.processed_archives = processed_archives
        self.num_examples_in_each_ark = num_examples_in_each_ark
        self.length = num_workers * num_examples_in_each_ark
        self._define_ark_loaders(iteration=finished_iterations + 1)

    def __getitem__(self, idx):
        # convert idx to corresponding ark file
        loader_idx = int(idx / self.num_examples_in_each_ark)
        assert loader_idx < self.num_workers
        # read the next example from corresponding loader
        mat, spk_id, name = self.data_loaders[loader_idx].next()
        return mat, spk_id

    def __len__(self):
        return self.length

    def _define_ark_loaders(self, iteration):
        assert iteration > self.finished_iterations
        iteration -= self.finished_iterations
        for i in range(self.num_workers):
            ark_idx = (self.processed_archives + (iteration - 1) *
                       self.num_workers + i + 1) % self.num_archives
            ark_file = os.path.join(self.egs_dir, 'egs.{0}.ark'.format(ark_idx))
            assert os.path.exists(ark_file)
            if self.data_loaders[i] is not None:
                # first stop the running thread
                self.data_loaders[i].stop_reading()
            self.data_loaders[i] = KaldiArkLoader(ark_file)

    def set_iteration(self, iteration):
        assert iteration > 1
        self._define_ark_loaders(iteration)
