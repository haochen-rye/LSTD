#!/bin/bash
set -e
set -x

folder="models/voc"
file_prefix="voc2007_ssd_300_"
model_path="models/voc"

if [ "$#" -lt 2 ]; then
	echo "Illegal number of parameters"
	echo "Usage: train_script base_lr weight_decay device_id template_solver.prototxt [finetuned.caffemodel/.solverstate]"
	exit
fi
base_lr=$1
weight_decay=$2
solver_mode="GPU"
device_id=0

current_time=$(date +%j_%T)
current_time=${current_time// /_}
current_time=${current_time//:/-}

if [ "$#" -ge 3 ]; then
	device_id=$3
fi

snapshot_path=$folder/${base_lr}_${weight_decay}_${device_id}_${current_time}
mkdir $snapshot_path

solverfile=$snapshot_path/solver.prototxt
template_file='template_solver.prototxt'
if [ "$#" -ge 4 ]; then
template_file=$4
fi

cat $folder/${template_file} > $solverfile
echo "weight_decay: $weight_decay" >> $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile

# echo "device_id: $device_id" >> $solverfile
echo "solver_mode: $solver_mode" >> $solverfile
#echo "regularization_type: \"$regularization_type\"" >> $solverfile
#cat $solverfile

tunedmodel="../1shot_rela_oicr_ssd.caffemodel"
if [ "$#" -ge 5 ]; then
tunedmodel=$5
fi

./build/tools/caffe.bin train --solver=$solverfile --weights=$model_path/$tunedmodel  --gpu $device_id 2>&1 | tee "${snapshot_path}/train.info"

# cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info
cat ${snapshot_path}/train.info | grep "Test net output" | awk '{print $0}' > ${snapshot_path}/test.info
