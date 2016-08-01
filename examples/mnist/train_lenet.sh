#!/usr/bin/env sh
set -e

./build/tools/OPT_caffe train_opt --solver=examples/mnist/opt_train.prototxt $@
