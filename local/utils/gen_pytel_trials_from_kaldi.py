#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Created Time:   2019-03-12
# @Author: Shuai Wang
# @Email: wsstriving@gmail.com
# @Last Modified Time: 2019-03-12

import sys, os
sys.path.append("/mnt/matylda6/rohdin/expts/pytel/")
import pytel.scoring

text_key_file=sys.argv[1]

key=pytel.scoring.Key.fromfile(text_key_file)
key.save(os.path.splitext(text_key_file)[0] + ".h5")