#!/bin/bash

BIN_PATH=/home/raj/g4xtools/g4Xdiagnostics.x86_64

sudo taskset -c 31 $BIN_PATH -i 2 perf -loop 10000 -delay 0


