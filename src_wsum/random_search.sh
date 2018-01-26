#!bin/bash

for lr in 0.002 0.005 0.01 0.05 0.1;
do
    for l2 in 1e-6 1e-5 5e-5 1e-4 1e-3;
    do
         log_dir=log/${lr}_${l2}/
         mkdir -p ${log_dir}
         CUDA_VISIBLE_DEVICES=1 python main.py  \
	 --function=train \
         --lr ${lr} \
	 --l2 ${l2} &> ${log_dir}/log.txt
         
    done
done
