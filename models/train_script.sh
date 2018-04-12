#!/bin/bash
set -e
set -x

folder="models/VOC2007/train5_restrain_post_kd_ssl_ft_SSD_300x300"
file_prefix="VOC2007_ssd_300_"
model_path="models/VOC2007/train5_restrain_post_kd_ssl_ft_SSD_300x300"

if [ "$#" -lt 5 ]; then
	echo "Illegal number of parameters"
	echo "Usage: train_script base_lr weight_decay kernel_shape_decay breadth_decay block_group_decay device_id template_solver.prototxt [finetuned.caffemodel/.solverstate]"
	exit
fi
base_lr=$1
weight_decay=$2
kernel_shape_decay=$3
breadth_decay=$4
block_group_decay=$5
solver_mode="GPU"
device_id=0

current_time=$(date +%j_%T)
current_time=${current_time// /_}
current_time=${current_time//:/-}

snapshot_path=$folder/${base_lr}_${weight_decay}_${kernel_shape_decay}_${breadth_decay}_${block_group_decay}_${current_time}
mkdir $snapshot_path

solverfile=$snapshot_path/solver.prototxt
template_file='template_solver.prototxt'
if [ "$#" -ge 8 ]; then
template_file=$8
fi

cat $folder/${template_file} > $solverfile
echo "block_group_decay: $block_group_decay" >> $solverfile
echo "kernel_shape_decay: $kernel_shape_decay" >> $solverfile
echo "breadth_decay: $breadth_decay" >> $solverfile
echo "weight_decay: $weight_decay" >> $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
if [ "$#" -ge 6 ]; then
	device_id=$6
fi
# echo "device_id: $device_id" >> $solverfile
echo "solver_mode: $solver_mode" >> $solverfile
#echo "regularization_type: \"$regularization_type\"" >> $solverfile
#cat $solverfile

tunedmodel="../../coco_60_post_kd.caffemodel"
if [ "$#" -ge 7 ]; then
tunedmodel=$7
fi

./build/tools/caffe.bin train --solver=$solverfile --weights=$model_path/$tunedmodel  --gpu $device_id 2>&1 | tee "${snapshot_path}/train.info" 

# cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info
cat ${snapshot_path}/train.info | grep "Test net output" | awk '{print $0}' > ${snapshot_path}/test.info
