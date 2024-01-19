#!/bin/bash
start=$1
end=$2
njobs=$3
folder=${4:-\$\{now:%H-%M-%S\}}
ntrials=$(((end-start+njobs-1) / njobs ))
for i in `seq 0 $((njobs-1))`;
do
	curr_start=$((start+(i*ntrials)))
	curr_end=$((start+((i+1)*ntrials)-1))
	if [ $curr_end -gt $end ]
	then
		curr_end=$end
	fi
    time sh run_configs.sh $curr_start $curr_end $folder	&
	echo -e "\n\n******** started $curr_start to $curr_end batch ******** \n\n"
	sleep 30
done
