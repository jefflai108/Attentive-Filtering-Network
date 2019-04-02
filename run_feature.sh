#!/bin/bash
# extract fbank, mfcc, plp, ivector features for:
# ASVspoof2017 train, dev, eval, train_dev
# Jeff Lai
. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
fbankdir=`pwd`/fbank
plpdir=`pwd`/plp
specdir=`pwd`/logspec
vadir=`pwd`/vad
stage=0
num_components=128 # UBM
ivector_dim=200 # ivector

if [ $stage -eq 0 ]; then
    # cqcc_spectrogram (863) feature extraction
    # apply cmvn sliding window 
    for name in train dev eval; do
      utils/fix_data_dir.sh data/${name}_cqcc_spectrogram
      utils/copy_data_dir.sh data/${name}_cqcc_spectrogram data/${name}_cqcc_spectrogram_cmvn
      feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_cqcc_spectrogram/feats.scp ark:- |"
      copy-feats "$feats" ark,scp:`pwd`/data/${name}_cqcc_spectrogram_cmvn/feats.ark,`pwd`/data/${name}_cqcc_spectrogram_cmvn/feats.scp
done 
fi 

if [ $stage -eq -6 ]; then
    # fbank (40) feature extraction
    # apply cmvn sliding window 
    for name in train dev eval train_dev; do
      utils/fix_data_dir.sh data/${name}
      utils/copy_data_dir.sh data/${name} data/${name}_fbank
      steps/make_fbank.sh --fbank-config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
	data/${name}_fbank exp/make_fbank $fbankdir
     
      utils/copy_data_dir.sh data/${name}_fbank data/${name}_fbank_cmvn
      feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_fbank/feats.scp ark:- |"
      copy-feats "$feats" ark,scp:`pwd`/data/${name}_fbank_cmvn/feats.ark,`pwd`/data/${name}_fbank_cmvn/feats.scp
done 
fi 

if [ $stage -eq -5 ]; then
    # logspec (257) feature extraction
    # apply cmvn sliding window 
    for name in train dev eval train_dev; do
      #utils/copy_data_dir.sh data/${name}_spec data/${name}_spec_cmvn
      feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_spec/feats.scp ark:- |"
      copy-feats "$feats" ark,scp:`pwd`/data/${name}_spec_cmvn/feats.ark,`pwd`/data/${name}_spec_cmvn/feats.scp
    done
fi 

if [ $stage -eq -4 ]; then
    # logspec (257) feature extraction
    for name in train dev eval train_dev; do
      utils/fix_data_dir.sh data/${name}
      utils/copy_data_dir.sh data/${name} data/${name}_spec
      
      local/make_spectrogram.sh --fbank-config conf/spec.conf --nj 40 --cmd "$train_cmd" \
          data/${name}_spec exp/make_spec $specdir
    done
fi 

if [ $stage -eq -3 ]; then
    # logspec (257) feature extraction
    # apply vad 
    for name in train dev eval train_dev; do
      utils/fix_data_dir.sh data/${name}
      utils/copy_data_dir.sh data/${name} data/${name}_spec
      
      local/make_spectrogram.sh --fbank-config conf/spec.conf --nj 40 --cmd "$train_cmd" \
          data/${name}_spec exp/make_spec $specdir
      sid/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
	  data/${name}_spec exp/make_vad $vadir
      utils/copy_data_dir.sh data/${name}_spec data/${name}_spec_vad
      feats="ark:select-voiced-frames scp:`pwd`/data/${name}_spec/feats.scp scp:`pwd`/data/${name}_spec/vad.scp ark:- |"
      copy-feats "$feats" ark,scp:`pwd`/data/${name}_spec_vad/feats.ark,`pwd`/data/${name}_spec_vad/feats.scp
    done
fi 

if [ $stage -eq -2 ]; then
    # logspec (257) feature extraction
    # apply vad --> cmvn sliding window 
    for name in train dev eval train_dev; do
      utils/fix_data_dir.sh data/${name}
      utils/copy_data_dir.sh data/${name} data/${name}_spec
      
      local/make_spectrogram.sh --fbank-config conf/spec.conf --nj 40 --cmd "$train_cmd" \
          data/${name}_spec exp/make_spec $specdir
      sid/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
	  data/${name}_spec exp/make_vad $vadir
      utils/copy_data_dir.sh data/${name}_spec data/${name}_spec_vad_cmvn
      feats="ark:select-voiced-frames scp:`pwd`/data/${name}_spec/feats.scp scp:`pwd`/data/${name}_spec/vad.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
      copy-feats "$feats" ark,scp:`pwd`/data/${name}_spec_vad_cmvn/feats.ark,`pwd`/data/${name}_spec_vad_cmvn/feats.scp
    done
fi 

if [ $stage -eq -1 ]; then
    # feature extraction
    # mfcc (24), fbank (40), plp (13)

    # Create a copy of data/train,dev,eval,train_dev for MFCC, Fbank & PLP
    for name in train dev eval train_dev; do
      utils/fix_data_dir.sh data/${name}
      
      # mfcc
      utils/copy_data_dir.sh data/${name} data/${name}_mfcc
      steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
          data/${name}_mfcc exp/make_mfcc $mfccdir
      # fbank
      utils/copy_data_dir.sh data/${name} data/${name}_fbank
      steps/make_fbank.sh --fbank-config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
          data/${name}_fbank exp/make_fbank $fbankdir
      # plp
      utils/copy_data_dir.sh data/${name} data/${name}_plp
      steps/make_plp.sh --plp-config conf/plp.conf --nj 40 --cmd "$train_cmd" \
          data/${name}_plp exp/make_plp $plpdir
    done
fi

if [ $stage -eq 1 ]; then
    # compute utterance-level cmvn stats for
    # mfcc (24), fbank (40), plp (13)

    for name in train dev eval train_dev; do
      # mfcc
      compute-cmvn-stats scp:data/${name}_mfcc/feats.scp ark,scp:`pwd`/data/${name}_mfcc/cmvn.ark,`pwd`/data/${name}_mfcc/cmvn.scp
      # fbank
      compute-cmvn-stats scp:data/${name}_fbank/feats.scp ark,scp:`pwd`/data/${name}_fbank/cmvn.ark,`pwd`/data/${name}_fbank/cmvn.scp
      # plp
      compute-cmvn-stats scp:data/${name}_plp/feats.scp ark,scp:`pwd`/data/${name}_plp/cmvn.ark,`pwd`/data/${name}_plp/cmvn.scp
    done
fi

if [ $stage -eq 2 ]; then
    # ivector extractor with UBM trained with 24-dim mfcc
    # 1) data/train_mfcc
    # 2) data/dev_mfcc
    # 3) data/train_dev_mfcc

    for name in train dev train_dev; do
        # GMM training
        local/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
            --nj 4 --num-threads 8 \
            data/${name}_mfcc $num_components \
            exp/${num_components}_diag_ubm_${name}_mfcc
        local/train_full_ubm.sh --nj 4 --remove-low-count-gaussians false \
            --cmd "$train_cmd --mem 25G" data/${name}_mfcc \
            exp/${num_components}_diag_ubm_${name}_mfcc exp/${num_components}_full_ubm_${name}_mfcc

        # ivector training
        local/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" --nj 1 \
          --ivector-dim $ivector_dim \
          --num-iters 5 exp/${num_components}_full_ubm_${name}_mfcc/final.ubm data/${name}_mfcc \
          exp/${ivector_dim}_extractor_${name}_mfcc
    done
fi

if [ $stage -eq 3 ]; then
    # Extract 24-dim-mfcc-based i-vectors
    # extractor trained on:
    # T D T_D
    # extracts for:
    # T D T_D E

    for name in train dev eval train_dev; do
        local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 4 \
            exp/${ivector_dim}_extractor_train_mfcc data/${name}_mfcc \
            exp/${ivector_dim}_ivectors_train_mfcc_${name}

         local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 4 \
            exp/${ivector_dim}_extractor_dev_mfcc data/${name}_mfcc \
            exp/${ivector_dim}_ivectors_dev_mfcc_${name}

        local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 4 \
            exp/${ivector_dim}_extractor_train_dev_mfcc data/${name}_mfcc \
            exp/${ivector_dim}_ivectors_train_dev_mfcc_${name}
    done
fi

if [ $stage -eq 4 ]; then
    # ivector extractor with UBM trained with 30-dim cqcc
    # 1) data/train_cqcc
    # 2) data/dev_cqcc
    # 3) data/train_dev_cqcc

    for name in train dev train_dev; do
        utils/fix_data_dir.sh data/${name}_cqcc
        # GMM training
        local/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
            --nj 4 --num-threads 8 \
            data/${name}_cqcc $num_components \
            exp/${num_components}_diag_ubm_${name}_cqcc
        local/train_full_ubm.sh --nj 4 --remove-low-count-gaussians false \
            --cmd "$train_cmd --mem 25G" data/${name}_cqcc \
            exp/${num_components}_diag_ubm_${name}_cqcc exp/${num_components}_full_ubm_${name}_cqcc

        # ivector training
        local/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" --nj 1 \
          --ivector-dim $ivector_dim \
          --num-iters 5 exp/${num_components}_full_ubm_${name}_cqcc/final.ubm data/${name}_cqcc \
          exp/${ivector_dim}_extractor_${name}_cqcc
    done
fi

if [ $stage -eq 5 ]; then
    # Extract 30-dim-cqcc-based i-vectors
    # extractor trained on:
    # T D T_D
    # extracts for:
    # T D T_D E

    for name in train dev eval train_dev; do
        utils/fix_data_dir.sh data/${name}_cqcc
        local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 4 \
            exp/${ivector_dim}_extractor_train_cqcc data/${name}_cqcc \
            exp/${ivector_dim}_ivectors_train_cqcc_${name}

         local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 4 \
            exp/${ivector_dim}_extractor_dev_cqcc data/${name}_cqcc \
            exp/${ivector_dim}_ivectors_dev_cqcc_${name}

        local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 4 \
            exp/${ivector_dim}_extractor_train_dev_cqcc data/${name}_cqcc \
            exp/${ivector_dim}_ivectors_train_dev_cqcc_${name}
    done
fi


if [ $stage -eq 6 ]; then
    # ivector extractor with UBM trained with 19-dim cqcc
    # 1) data/train_cqcc_19
    # 2) data/dev_cqcc_19
    # 3) data/train_dev_cqcc_19

    for name in train dev train_dev; do
        utils/fix_data_dir.sh data/${name}_cqcc_19
        # GMM training
        local/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
            --nj 4 --num-threads 8 \
            data/${name}_cqcc_19 $num_components \
            exp/${num_components}_diag_ubm_${name}_cqcc_19
        local/train_full_ubm.sh --nj 4 --remove-low-count-gaussians false \
            --cmd "$train_cmd --mem 25G" data/${name}_cqcc_19 \
            exp/${num_components}_diag_ubm_${name}_cqcc_19 exp/${num_components}_full_ubm_${name}_cqcc_19

        # ivector training
        local/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" --nj 1 \
          --ivector-dim $ivector_dim \
          --num-iters 5 exp/${num_components}_full_ubm_${name}_cqcc_19/final.ubm data/${name}_cqcc_19 \
          exp/${ivector_dim}_extractor_${name}_cqcc_19
    done
fi

if [ $stage -eq 7 ]; then
    # Extract 19-dim-cqcc-based i-vectors
    # extractor trained on:
    # T D T_D
    # extracts for:
    # T D T_D E

    for name in train dev eval train_dev; do
        utils/fix_data_dir.sh data/${name}_cqcc_19
        local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 4 \
            exp/${ivector_dim}_extractor_train_cqcc_19 data/${name}_cqcc_19 \
            exp/${ivector_dim}_ivectors_train_cqcc_19_${name}

         local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 4 \
            exp/${ivector_dim}_extractor_dev_cqcc_19 data/${name}_cqcc_19 \
            exp/${ivector_dim}_ivectors_dev_cqcc_19_${name}

        local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 4 \
            exp/${ivector_dim}_extractor_train_dev_cqcc_19 data/${name}_cqcc_19 \
            exp/${ivector_dim}_ivectors_train_dev_cqcc_19_${name}
    done
fi


