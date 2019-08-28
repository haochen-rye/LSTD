#!/bin/bash
set -e
set -x

folder="models/voc2007/kd_oicr_SSD_300x300"
file_prefix="voc2007_ssd_300_"
model_path="models/voc2007/kd_oicr_SSD_300x300"

if [ "$#" -lt 1 ]; then
	echo "Illegal number of parameters"
	echo "Usage: train_script base_lr weight_decay template_solver.prototxt [finetuned.caffemodel/.solverstate]"
	exit
fi
shot_num=$1
solver_mode="GPU"
dataset=0712
device_id=0
base_lr=1e-5
weight_decay=1e-4
current_time=$(date +%j_%T)
current_time=${current_time// /_}
current_time=${current_time//:/-}

if [ "$#" -ge 2 ]; then
	dataset=$2
fi

if [ "$dataset" == "0712" ]; then
    template_file="${shot_num}shot_template_solver.prototxt"
else
    template_file="${shot_num}shot_${dataset}_template_solver.prototxt"
fi

if [ "$#" -ge 3 ]; then
	base_lr=$3
fi

snapshot_path=$folder/${shot_num}shot_${dataset}_${base_lr}_${weight_decay}_${current_time}
mkdir $snapshot_path

solverfile=$snapshot_path/solver.prototxt

cat $folder/${template_file} > $solverfile
echo "weight_decay: $weight_decay" >> $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile

# echo "device_id: $device_id" >> $solverfile
echo "solver_mode: $solver_mode" >> $solverfile
#echo "regularization_type: \"$regularization_type\"" >> $solverfile
#cat $solverfile

tunedmodel="../lstd_model/${shot_num}shot_kd_oicr_ssd.caffemodel"
if [ "$#" -ge 4 ]; then
tunedmodel=$4
fi

./build/tools/caffe.bin train --solver=$solverfile --weights=$model_path/$tunedmodel  --gpu $device_id 2>&1 | tee "${snapshot_path}/train.info"

# cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info
cat ${snapshot_path}/train.info | grep "Test net output" | awk '{print $0}' > ${snapshot_path}/test.info
