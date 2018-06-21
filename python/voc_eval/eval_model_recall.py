# get the proposal result 
import sys,os
import numpy as np
sys.path.append('python')
import caffe

model_test=sys.argv[1]
device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dataset = sys.argv[3] if len(sys.argv) > 3 else 'test'
year = sys.argv[4] if len(sys.argv) > 4 else '2007'
caffe.set_device(device_id)
caffe.set_mode_gpu()

num_proposal=[64,128,256,512,1024]
# get the net and weights
net_def = 'models/voc{}/kd_oicr_SSD_300x300/1024_proposal_deploy.prototxt'.format(year)
net_weight = 'models/voc{}/{}.caffemodel'.format(year,model_test)
test_net = caffe.Net(net_def, net_weight, caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': test_net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

image_resize = 300
test_net.blobs['data'].reshape(1,3,image_resize,image_resize)

data_dir='/home/chenhao/data/VOCdevkit/VOC{}/JPEGImages'.format(year)
name_size_file='data/VOC0712/voc07_{}_name_size.txt'.format(dataset)

det_result={}
for ele in num_proposal:
	det_result[ele] = []

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
	detections = test_net.forward()['objectness_detection_out']

	# Parse the outputs.
	det_conf = detections[0,0,:,2]
	det_xmin = detections[0,0,:,3]
	det_ymin = detections[0,0,:,4]
	det_xmax = detections[0,0,:,5]
	det_ymax = detections[0,0,:,6]
	for ele in num_proposal:
			det_result[ele].extend(['{} {:0.2f} {:0.1f} {:0.1f} {:0.1f} {:0.1f}\n'.format(
				image_index, det_conf[i], width * det_xmin[i], height * det_ymin[i],
				width * det_xmax[i], height * det_ymax[i]) for i in xrange(ele)])

res_dir = "results/voc{}/{}_{}".format(year, model_test, dataset)
if os.path.isdir(res_dir):
	for file in os.listdir(res_dir):
		os.remove(os.path.join(res_dir, file))
else:
	os.mkdir(res_dir)

for ele in num_proposal:
	result_file=os.path.join(res_dir, "{}_proposal_out.txt".format(ele))
	with open(result_file, 'w') as f:
		for bbox in det_result[ele]:
			f.write(bbox)

eval_recall(model_name = model_test)