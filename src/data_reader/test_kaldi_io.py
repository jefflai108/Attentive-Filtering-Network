from __future__ import print_function
import adv_kaldi_io as ako

test_scp = '/export/b19/jlai/cstr/spoof/baseline/v1/data/train_cqcc/feats.scp'
test_utt2label = '/export/b19/jlai/cstr/spoof/baseline/v1/data/train_cqcc/utt2label'
test_key = 'M0001-T_1000027'

all_key = ako.read_all_key(test_scp)
mat     = ako.read_mat_key(test_scp, test_key)
length  = ako.read_total_len(test_scp)
print(length) # 938016
label   = ako.read_key_label(test_utt2label)
key_map = ako.read_key_len(test_scp)
print(key_map)
