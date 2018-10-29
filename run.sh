#!/bin/bash
# running model
source /export/b18/nchen/keras/bin/activate
stage="$1" # parse first argument 

if [ $stage -eq 89 ]; then
    # main
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/b19/jlai/cstr/spoof/model/main.py \
	--train-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/train_spec_cmvn_tensor.scp \
        --train-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/train_utt2label \
        --validation-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/dev_spec_cmvn_tensor.scp \
        --validation-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/dev_utt2label \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/eval_spec_cmvn_tensor.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/eval_utt2label \
        --logging-dir /export/b19/jlai/cstr/spoof/model/snapshots/attention/ --epochs 30 --log-interval 50
fi

if [ $stage -eq 90 ]; then
    # feature_plot #eval_spec_cmvn_tensor.scp \
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/b19/jlai/cstr/spoof/model/feature_plot.py \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/new_color_map2.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/eval_utt2label \
        --logging-dir /export/b19/jlai/cstr/spoof/model/snapshots/predict_only/ --test-batch-size 1 \
	--plot-dir /export/b19/jlai/cstr/spoof/model/src/data_reader/plot/attention/new_colormap/shit/
fi

if [ $stage -eq 91 ]; then
    # predict_only 
    echo train 
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/b19/jlai/cstr/spoof/model/predict_only.py \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/train_spec_cmvn_tensor.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/train_utt2label \
        --logging-dir /export/b19/jlai/cstr/spoof/model/snapshots/predict_only/ --test-batch-size 4 \
	--scoring-txt /export/b19/jlai/cstr/spoof/model/snapshots/scoring/train_attention8_pred.txt \
	--label-txt /export/b19/jlai/cstr/spoof/model/snapshots/scoring/train_attention8_label.txt

    echo dev 
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/b19/jlai/cstr/spoof/model/predict_only.py \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/dev_spec_cmvn_tensor.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/dev_utt2label \
        --logging-dir /export/b19/jlai/cstr/spoof/model/snapshots/predict_only/ --test-batch-size 4 \
	--scoring-txt /export/b19/jlai/cstr/spoof/model/snapshots/scoring/dev_attention8_pred.txt \
	--label-txt /export/b19/jlai/cstr/spoof/model/snapshots/scoring/dev_attention8_label.txt

    echo eval 
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/b19/jlai/cstr/spoof/model/predict_only.py \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/eval_spec_cmvn_tensor.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/eval_utt2label \
        --logging-dir /export/b19/jlai/cstr/spoof/model/snapshots/predict_only/ --test-batch-size 4 \
	--scoring-txt /export/b19/jlai/cstr/spoof/model/snapshots/scoring/eval_attention8_pred.txt \
	--label-txt /export/b19/jlai/cstr/spoof/model/snapshots/scoring/eval_attention8_label.txt
fi

############################################### OBSOLETE BELOW ##################################################
if [ $stage -eq 92 ]; then
    # utterance-based CNN train + predict
    # cqcc spectrogram (+sliding-cmvn, 863 by 1278): 26G ram
    CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
	--train-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/train_cqcc_spectrogram_cmvn_tensor.scp \
        --train-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/utt2label/train_utt2label \
        --validation-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/dev_cqcc_spectrogram_cmvn_tensor.scp \
        --validation-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/utt2label/dev_utt2label \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/eval_cqcc_spectrogram_cmvn_tensor.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/utt2label/eval_utt2label \
        --logging-dir ./snapshots/conv_net/ --batch-size 8 --test-batch-size 8 \
        --epochs 30 --log-interval 10 
fi

if [ $stage -eq 93 ]; then
    # utterance-based CNN train + predict
    # fbank (+sliding-cmvn, 40 by 1091): 20G ram
    CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
	--train-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/fbank/train_fbank_cmvn_tensor.scp \
        --train-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/fbank/utt2label/train_utt2label \
        --validation-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/fbank/dev_fbank_cmvn_tensor.scp \
        --validation-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/fbank/utt2label/dev_utt2label \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/fbank/eval_fbank_cmvn_tensor.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/fbank/utt2label/eval_utt2label \
        --logging-dir ./snapshots/conv_net/ --batch-size 32 --test-batch-size 32 \
        --epochs 30 --log-interval 10
fi

if [ $stage -eq 94 ]; then
    # utterance-based FCNN train + predict
    # fbank (+sliding-cmvn, 40 by 1091): 20G ram
    CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
	--train-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/train_spec_cmvn_orig.scp \
        --train-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/train_utt2label \
        --validation-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/dev_spec_cmvn_orig.scp \
        --validation-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/dev_utt2label \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/eval_spec_cmvn_orig.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/eval_utt2label \
        --logging-dir ./snapshots/fconv/ --batch-size 1 --test-batch-size 1 \
        --epochs 30 --log-interval 200 
fi

if [ $stage -eq 95 ]; then
    # utterance-based CNN train + predict
    # log spectrogram (+sliding-cmvn, 257 by 1091): 30G ram
    CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
	--train-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/train_spec_cmvn_tensor.scp \
        --train-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/train_utt2label \
        --validation-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/dev_spec_cmvn_tensor.scp \
        --validation-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/dev_utt2label \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/eval_spec_cmvn_tensor.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/eval_utt2label \
        --logging-dir ./snapshots/attention/ --epochs 30 --log-interval 10 
fi

if [ $stage -eq 96 ]; then
    # utterance-based CNN train + predict
    # log spectrogram (+vad+sliding-cmvn, 257 by 1091): 30G ram
    CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
	--train-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/train_spec_vad_cmvn_tensor.scp \
        --train-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/train_utt2label \
        --validation-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/dev_spec_vad_cmvn_tensor.scp \
        --validation-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/dev_utt2label \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/eval_spec_vad_cmvn_tensor.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/eval_utt2label \
        --logging-dir ./snapshots/fconv/ --batch-size 32 --test-batch-size 32 \
        --epochs 30 --log-interval 10 
fi

if [ $stage -eq 97 ]; then
    # feed-forward train + predict
    # CQCC(30), M = 0: 15G ram
    # CQCC(30), M = 5: 30G ram
    # CQCC(30), M = 5, with eval: 100G ram
    # CQCC(30) with v6_dataset: 30G ram 
    #CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
    CUDA_VISIBLE_DEVICES=-1 python main.py \
	--train-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/data/train_orig.scp \
        --train-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/data/utt2label/train_utt2label \
        --validation-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/data/dev_orig.scp \
        --validation-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/data/utt2label/dev_utt2label \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/data/eval_orig.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/data/utt2label/eval_utt2label \
        --logging-dir ./snapshots/conv_net/ --batch-size 32 --test-batch-size 1000 \
        --epochs 30 --log-interval 500
fi

if [ $stage -eq 98 ]; then
    # feed-forward train + predict
    # CQCC(30), M = 0: 15G ram
    # CQCC(30), M = 5: 40G ram
    # .., .., with eval: 100G ram
    CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
        --train-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/data/train_orig.scp \
        --train-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/data/utt2label/train_utt2label \
        --validation-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/data/dev_orig.scp \
        --validation-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/data/utt2label/dev_utt2label \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/data/eval_orig.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/data/utt2label/eval_utt2label \
        --logging-dir ./snapshots/feed_forward/ --batch-size 32 --test-batch-size 1000 \
        --epochs 30 --log-interval 500
fi

if [ $stage -eq 99 ]; then
    # feed-forward predict-only
    # CQCC(30), M = 5: 100G ram
    CUDA_VISIBLE_DEVICES=`free-gpu` python predict_only.py \
        --model-path /export/b19/jlai/cstr/spoof/model/snapshots/feed_forward/feed-forward-2018-06-15_19-model_best.pth \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/data/eval_orig.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/data/utt2label/eval_utt2label \
        --logging-dir ./snapshots/predict_only/ --test-batch-size 1000
fi

if [ $stage -eq 100 ]; then
    # Convnet with log spectroram
    # 40G ram
    CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
        --train-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/train_spec_tensor.scp \
        --train-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/utt2label/train_utt2label \
        --validation-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/dev_spec_tensor.scp \
        --validation-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/utt2label/dev_utt2label \
        --eval-scp /export/b19/jlai/cstr/spoof/model/src/data_reader/spec/eval_spec_tensorg.scp \
        --eval-utt2label /export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/utt2label/eval_utt2label \
        --logging-dir ./snapshots/conv_net/ --batch-size 1 --test-batch-size 1 \
        --epochs 20 --log-interval 500 --hidden-dim 64
fi
##################################################################################################################
