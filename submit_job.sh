#!/bin/bash 
# qsub script 
clean=true
stage=6
memory=30g

if [ $stage -eq 1 ]; then 
	out1='/export/b19/jlai/cstr/spoof/model/qsub/attention/out1'
	err1='/export/b19/jlai/cstr/spoof/model/qsub/attention/err1'
	out2='/export/b19/jlai/cstr/spoof/model/qsub/attention/out2'
	err2='/export/b19/jlai/cstr/spoof/model/qsub/attention/err2'
	out3='/export/b19/jlai/cstr/spoof/model/qsub/attention/out3'
	err3='/export/b19/jlai/cstr/spoof/model/qsub/attention/err3'
	out4='/export/b19/jlai/cstr/spoof/model/qsub/attention/out4'
	err4='/export/b19/jlai/cstr/spoof/model/qsub/attention/err4'
	out5='/export/b19/jlai/cstr/spoof/model/qsub/attention/out5'
	err5='/export/b19/jlai/cstr/spoof/model/qsub/attention/err5'
	out6='/export/b19/jlai/cstr/spoof/model/qsub/attention/out6'
	err6='/export/b19/jlai/cstr/spoof/model/qsub/attention/err6'
	out7='/export/b19/jlai/cstr/spoof/model/qsub/attention/out7'
	err7='/export/b19/jlai/cstr/spoof/model/qsub/attention/err7'
	out8='/export/b19/jlai/cstr/spoof/model/qsub/attention/out8'
	err8='/export/b19/jlai/cstr/spoof/model/qsub/attention/err8'
	out9='/export/b19/jlai/cstr/spoof/model/qsub/attention/out9'
	err9='/export/b19/jlai/cstr/spoof/model/qsub/attention/err9'
	out10='/export/b19/jlai/cstr/spoof/model/qsub/attention/out10'
	err10='/export/b19/jlai/cstr/spoof/model/qsub/attention/err10'
	out11='/export/b19/jlai/cstr/spoof/model/qsub/attention/out11'
	err11='/export/b19/jlai/cstr/spoof/model/qsub/attention/err11'
	out12='/export/b19/jlai/cstr/spoof/model/qsub/attention/out12'
	err12='/export/b19/jlai/cstr/spoof/model/qsub/attention/err12'
	out13='/export/b19/jlai/cstr/spoof/model/qsub/attention/out13'
	err13='/export/b19/jlai/cstr/spoof/model/qsub/attention/err13'
	out14='/export/b19/jlai/cstr/spoof/model/qsub/attention/out14'
	err14='/export/b19/jlai/cstr/spoof/model/qsub/attention/err14'
	out15='/export/b19/jlai/cstr/spoof/model/qsub/attention/out15'
	err15='/export/b19/jlai/cstr/spoof/model/qsub/attention/err15'
	out16='/export/b19/jlai/cstr/spoof/model/qsub/attention/out16'
	err16='/export/b19/jlai/cstr/spoof/model/qsub/attention/err16'
	out17='/export/b19/jlai/cstr/spoof/model/qsub/attention/out17'
	err17='/export/b19/jlai/cstr/spoof/model/qsub/attention/err17'
	out18='/export/b19/jlai/cstr/spoof/model/qsub/attention/out18'
	err18='/export/b19/jlai/cstr/spoof/model/qsub/attention/err18'
	out19='/export/b19/jlai/cstr/spoof/model/qsub/attention/out19'
	err19='/export/b19/jlai/cstr/spoof/model/qsub/attention/err19'
	out20='/export/b19/jlai/cstr/spoof/model/qsub/attention/out20'
	err20='/export/b19/jlai/cstr/spoof/model/qsub/attention/err20'
	out21='/export/b19/jlai/cstr/spoof/model/qsub/attention/out21'
	err21='/export/b19/jlai/cstr/spoof/model/qsub/attention/err21'
	out22='/export/b19/jlai/cstr/spoof/model/qsub/attention/out22'
	err22='/export/b19/jlai/cstr/spoof/model/qsub/attention/err22'
	out23='/export/b19/jlai/cstr/spoof/model/qsub/attention/out23'
	err23='/export/b19/jlai/cstr/spoof/model/qsub/attention/err23'
	out24='/export/b19/jlai/cstr/spoof/model/qsub/attention/out24'
	err24='/export/b19/jlai/cstr/spoof/model/qsub/attention/err24'
	out25='/export/b19/jlai/cstr/spoof/model/qsub/attention/out25'
	err25='/export/b19/jlai/cstr/spoof/model/qsub/attention/err25'
	out26='/export/b19/jlai/cstr/spoof/model/qsub/attention/out26'
	err26='/export/b19/jlai/cstr/spoof/model/qsub/attention/err26'
	out27='/export/b19/jlai/cstr/spoof/model/qsub/attention/out27'
	err27='/export/b19/jlai/cstr/spoof/model/qsub/attention/err27'
	out28='/export/b19/jlai/cstr/spoof/model/qsub/attention/out28'
	err28='/export/b19/jlai/cstr/spoof/model/qsub/attention/err28'
	out29='/export/b19/jlai/cstr/spoof/model/qsub/attention/out29'
	err29='/export/b19/jlai/cstr/spoof/model/qsub/attention/err29'
	out30='/export/b19/jlai/cstr/spoof/model/qsub/attention/out30'
	err30='/export/b19/jlai/cstr/spoof/model/qsub/attention/err30'

	if [ "$clean" == "true" ]; then 
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/out{1..30}
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/err{1..30}
	fi 
	
	qsub -o $out1 -e $err1 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 89
	#qsub -o $out2 -e $err2 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 89
	#qsub -o $out3 -e $err3 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 89
	#qsub -o $out4 -e $err4 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 89
	#qsub -o $out5 -e $err5 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 89
	#qsub -o $out6 -e $err6 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 88
	#qsub -o $out7 -e $err7 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 88
	#qsub -o $out8 -e $err8 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 88
	#qsub -o $out9 -e $err9 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 88
	#qsub -o $out10 -e $err10 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 88
	#qsub -o $out11 -e $err11 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 87
	#qsub -o $out12 -e $err12 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 87
	#qsub -o $out13 -e $err13 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 87
	#qsub -o $out14 -e $err14 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 87
	#qsub -o $out15 -e $err15 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 87
	#qsub -o $out16 -e $err16 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 86
	#qsub -o $out17 -e $err17 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 86
	#qsub -o $out18 -e $err18 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 86
	#qsub -o $out19 -e $err19 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 86
	#qsub -o $out20 -e $err20 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 86
	#qsub -o $out21 -e $err21 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 85
	#qsub -o $out22 -e $err22 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 85
	#qsub -o $out23 -e $err23 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 85
	#qsub -o $out24 -e $err24 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 85
	#qsub -o $out25 -e $err25 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 85
	#qsub -o $out26 -e $err26 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 84
	#qsub -o $out27 -e $err27 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 84
	#qsub -o $out28 -e $err28 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 84
	#qsub -o $out29 -e $err29 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 84
	#qsub -o $out30 -e $err30 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 84
fi 

if [ $stage -eq 2 ]; then	
	out31='/export/b19/jlai/cstr/spoof/model/qsub/attention/out31'
	err31='/export/b19/jlai/cstr/spoof/model/qsub/attention/err31'
	out32='/export/b19/jlai/cstr/spoof/model/qsub/attention/out32'
	err32='/export/b19/jlai/cstr/spoof/model/qsub/attention/err32'
	out33='/export/b19/jlai/cstr/spoof/model/qsub/attention/out33'
	err33='/export/b19/jlai/cstr/spoof/model/qsub/attention/err33'
	out34='/export/b19/jlai/cstr/spoof/model/qsub/attention/out34'
	err34='/export/b19/jlai/cstr/spoof/model/qsub/attention/err34'
	out35='/export/b19/jlai/cstr/spoof/model/qsub/attention/out35'
	err35='/export/b19/jlai/cstr/spoof/model/qsub/attention/err35'
	out36='/export/b19/jlai/cstr/spoof/model/qsub/attention/out36'
	err36='/export/b19/jlai/cstr/spoof/model/qsub/attention/err36'
	out37='/export/b19/jlai/cstr/spoof/model/qsub/attention/out37'
	err37='/export/b19/jlai/cstr/spoof/model/qsub/attention/err37'
	out38='/export/b19/jlai/cstr/spoof/model/qsub/attention/out38'
	err38='/export/b19/jlai/cstr/spoof/model/qsub/attention/err38'
	out39='/export/b19/jlai/cstr/spoof/model/qsub/attention/out39'
	err39='/export/b19/jlai/cstr/spoof/model/qsub/attention/err39'
	out40='/export/b19/jlai/cstr/spoof/model/qsub/attention/out40'
	err40='/export/b19/jlai/cstr/spoof/model/qsub/attention/err40'
	out41='/export/b19/jlai/cstr/spoof/model/qsub/attention/out41'
	err41='/export/b19/jlai/cstr/spoof/model/qsub/attention/err41'
	out42='/export/b19/jlai/cstr/spoof/model/qsub/attention/out42'
	err42='/export/b19/jlai/cstr/spoof/model/qsub/attention/err42'
	out43='/export/b19/jlai/cstr/spoof/model/qsub/attention/out43'
	err43='/export/b19/jlai/cstr/spoof/model/qsub/attention/err43'

	if [ "$clean" == "true" ]; then 
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/out{31..43}
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/err{32..43}
	fi 
	
	qsub -o $out31 -e $err31 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 83
	qsub -o $out32 -e $err32 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 83
	qsub -o $out33 -e $err33 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 83
	#qsub -o $out34 -e $err34 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 87
	#qsub -o $out35 -e $err35 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 87
	qsub -o $out36 -e $err36 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 83
	qsub -o $out37 -e $err37 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 83
	#qsub -o $out38 -e $err38 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 88
	qsub -o $out39 -e $err39 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 82
	qsub -o $out40 -e $err40 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 82
	qsub -o $out41 -e $err41 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 82
	qsub -o $out42 -e $err42 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 82
	qsub -o $out43 -e $err43 -M clai24@jhu.edu -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 82
fi 

if [ $stage -eq 5 ]; then 
	out100='/export/b19/jlai/cstr/spoof/model/qsub/attention/out100'
	err100='/export/b19/jlai/cstr/spoof/model/qsub/attention/err100'
	
	if [ "$clean" == "true" ]; then 
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/out100
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/err100
	fi 
	
	qsub -o $out100 -e $err100 -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 90
fi 

if [ $stage -eq 6 ]; then 
	out101='/export/b19/jlai/cstr/spoof/model/qsub/attention/out101'
	err101='/export/b19/jlai/cstr/spoof/model/qsub/attention/err101'
	
	if [ "$clean" == "true" ]; then 
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/out101
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/err101
	fi 
	
	qsub -o $out101 -e $err101 -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 90
fi 

if [ $stage -eq 7 ]; then
	out102='/export/b19/jlai/cstr/spoof/model/qsub/attention/out102'
	err102='/export/b19/jlai/cstr/spoof/model/qsub/attention/err102'
	
	if [ "$clean" == "true" ]; then 
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/out102
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/err102
	fi 
	
	qsub -o $out102 -e $err102 -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 89

	out103='/export/b19/jlai/cstr/spoof/model/qsub/attention/out103'
	err103='/export/b19/jlai/cstr/spoof/model/qsub/attention/err103'
	
	if [ "$clean" == "true" ]; then 
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/out103
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/err103
	fi 
	
	qsub -o $out103 -e $err103 -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 88

	out104='/export/b19/jlai/cstr/spoof/model/qsub/attention/out104'
	err104='/export/b19/jlai/cstr/spoof/model/qsub/attention/err104'
	
	if [ "$clean" == "true" ]; then 
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/out104
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/err104
	fi 
	
	qsub -o $out104 -e $err104 -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 87

	out105='/export/b19/jlai/cstr/spoof/model/qsub/attention/out105'
	err105='/export/b19/jlai/cstr/spoof/model/qsub/attention/err105'
	
	if [ "$clean" == "true" ]; then 
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/out105
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/err105
	fi 
	
	qsub -o $out105 -e $err105 -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 86

	out106='/export/b19/jlai/cstr/spoof/model/qsub/attention/out106'
	err106='/export/b19/jlai/cstr/spoof/model/qsub/attention/err106'
	
	if [ "$clean" == "true" ]; then 
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/out106
		rm /export/b19/jlai/cstr/spoof/model/qsub/attention/err106
	fi 
	
	qsub -o $out106 -e $err106 -l "hostname=b1[12345678]*|c*,gpu=1,mem_free=$memory,ram_free=$memory" -q g.q run.sh 85

fi 
