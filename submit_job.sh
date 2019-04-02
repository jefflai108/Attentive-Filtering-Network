#!/bin/bash 
# qsub script for GPU job submission 

out100='/export/b19/jlai/cstr/spoof/model/qsub/attention/out100'
err100='/export/b19/jlai/cstr/spoof/model/qsub/attention/err100'
qsub -o $out100 -e $err100 -l "hostname=c*,gpu=1,mem_free=30g,ram_free=30g" -q g.q run.sh 0

