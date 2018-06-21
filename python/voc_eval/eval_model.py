# get the test result 
import sys,os
import numpy as np
from pascal_voc import eval_voc_result
from os.path import expanduser
home = expanduser("~")
sys.path.append('python')
import caffe
# get the label name and corresponging number
with open('data/VOC0712/labelmap_voc.prototxt') as f:
     label_lines=f.readlines()
cls_list = []
for ele in label_lines:
     if "display" in ele:
         cls_list.append(ele.split('"')[1] )

model_test=sys.argv[1]
device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
year = sys.argv[3] if len(sys.argv) > 3 else '2007'
model_extra = sys.argv[4] if len(sys.argv) > 4 else '128_ave_3_'
dataset = sys.argv[5] if len(sys.argv) > 5 else 'test'

caffe.set_device(device_id)
caffe.set_mode_gpu()

# get the net and weights
net_def = 'models/voc2007/kd_oicr_SSD_300x300/{}test_deploy.prototxt'.format(model_extra)
net_weight = 'models/voc2007/{}.caffemodel'.format(model_test)
test_net = caffe.Net(net_def, net_weight, caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': test_net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

image_resize = 300
test_net.blobs['data'].reshape(1,3,image_resize,image_resize)

data_dir='{}/data/VOCdevkit/VOC{}/JPEGImages'.format(home, year)
name_size_file='data/VOC0712/voc{}_{}_name_size.txt'.format(year[-2:],dataset)

det_result={}
num_cls = len(cls_list)
for i in xrange(1,num_cls):
	det_result[i]=""

with open(name_size_file) as f:
	name_lines=f.readlines()

for ele in name_lines:
	image_index = ele.split()[0]
	image = caffe.io.load_image(os.path.join(data_dir, image_index +'.jpg'))
	height = float(ele.split()[1])
	width = float(ele.split()[2])
	transformed_image = transformer.preprocess('data', image)
	test_net.blobs['data'].data[...] = transformed_image

	# Forward pass.
	detections = test_net.forward()['bbox_detection_out']

	# Parse the outputs.
	det_label = detections[0,0,:,1]
	det_conf = detections[0,0,:,2]
	det_xmin = detections[0,0,:,3]
	det_ymin = detections[0,0,:,4]
	det_xmax = detections[0,0,:,5]
	det_ymax = detections[0,0,:,6]
	for i in xrange(len(det_label)):
		label=int(det_label[i])
		if label > 0 and label < num_cls: 
			det_result[label] += '{} {:0.2f} {:0.1f} {:0.1f} {:0.1f} {:0.1f}\n'.format(
				image_index, det_conf[i], width * det_xmin[i], height * det_ymin[i],
				width * det_xmax[i], height * det_ymax[i])

res_dir = "results/voc{}/{}{}_{}".format(year, model_extra, model_test, dataset)
if os.path.isdir(res_dir):
	for file in os.listdir(res_dir):
		os.remove(os.path.join(res_dir, file))
else:
	os.mkdir(res_dir)

for i in xrange(1,num_cls):
	class_name=cls_list[i]
	# result_file="/home/chenhao/det_analy/detections/{}/comp4_det_test_{}.txt".format(model_test,class_name)
	result_file=os.path.join(res_dir, "comp4_det_test_{}.txt".format(class_name))
	with open(result_file, 'w') as f:
		f.write(det_result[i])

eval_voc_result(model_name = model_test, model_extra = model_extra,
	image_set = dataset, year = year)
