#!/bin/bash

# Copyright (c) 2018 Intel Corporation.
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Install the dependencies
sudo apt-get update
sudo apt-get install python3-pip
sudo pip3 install numpy jupyter

BASE_DIR=`pwd`

#Optimize the model
cd /opt/intel/openvino/deployment_tools/model_optimizer/
python3 mo_tf.py -m $BASE_DIR/resources/model/model.pb --input_shape=[1,224,224,3] --data_type FP32 -o $BASE_DIR/resources/FP32 --mean_values [123.75,116.28,103.58] --scale_values [58.395,57.12,57.375]
python3 mo_tf.py -m $BASE_DIR/resources/model/model.pb --input_shape=[1,224,224,3] --data_type FP16 -o $BASE_DIR/resources/FP16 --mean_values [123.75,116.28,103.58] --scale_values [58.395,57.12,57.375]

