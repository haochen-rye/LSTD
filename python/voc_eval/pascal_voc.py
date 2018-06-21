import os, sys
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from dis_eval import dis_eval
from os.path import expanduser
home = expanduser("~")

model_name = sys.argv[1] if len(sys.argv) > 1 else '1shot'
model_extra = sys.argv[2] if len(sys.argv) > 2 else '128_ave_3_'
image_set = sys.argv[3] if len(sys.argv) > 3 else 'test'
year = sys.argv[4] if len(sys.argv) > 4 else '2007'

devkit_path = os.path.join(home,'data','VOCdevkit')
classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

def do_python_eval(output_dir,detpath, image_set='test', year='2007'):
    annopath = os.path.join(
        devkit_path,
        'VOC' + year,
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        devkit_path,
        'VOC' + year,
        'ImageSets',
        'Main',
        image_set + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache_{}_{}'.format(year, image_set))
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_str = ""        
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = os.path.join(detpath, "comp4_det_test_{}.txt".format(cls))
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        output_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        # print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    output_str += 'Mean AP = {:.4f}'.format(np.mean(aps))
    print output_str
    with open(os.path.join(output_dir,"eval_results.txt"), 'w') as f:
        f.write(output_str)    
    # print('Mean AP = {:.4f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('Results:')
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')
    # print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** Python eval code.')
    # print('Results should be very close to the official MATLAB eval code.')
    # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    # print('-- Thanks, The Management')
    # print('--------------------------------------------------------------')

def do_eval_discovery( output_dir, detpath, image_set='test', year='2007'):
    annopath = os.path.join(
        devkit_path,
        'VOC' + year,
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        devkit_path,
        'VOC' + year,
        'ImageSets',
        'Main',
        image_set + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache_{}_{}'.format(year, image_set))
    corlocs = []
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_str=""
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = os.path.join(detpath, "comp4_det_test_{}.txt".format(cls))
        corloc = dis_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
        corlocs += [corloc]
        output_str += 'CorLoc for {} = {:.4f}\n'.format(cls, corloc)
        # print('CorLoc for {} = {:.4f}'.format(cls, corloc))
        with open(os.path.join(output_dir, cls + '_corloc.pkl'), 'w') as f:
            cPickle.dump({'corloc': corloc}, f)
    # print('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))
    output_str += 'Mean CorLoc = {:.4f}\n'.format(np.mean(corlocs))
    print output_str
    with open(os.path.join(output_dir,"eval_results.txt"), 'w') as f:
        f.write(output_str)
    # print('~~~~~~~~')
    # print('Results:')
    # for corloc in corlocs:
    #     print('{:.3f}'.format(corloc))
    # print('{:.3f}'.format(np.mean(corlocs)))
    # print('~~~~~~~~')

def eval_voc_result(model_name = '1shot', model_extra = "128_",
    image_set = "test", year = "2007"):
# get the mAP or corLOC of the detection results
    respath = os.path.join('results','voc{}'.format(year))
    detpath = os.path.join(respath, model_extra + model_name+'_'+image_set)

    map_output_dir = os.path.join(respath, 'map_output', model_extra + model_name)
    loc_output_dir = os.path.join(respath, 'loc_output', model_extra + model_name)

    ########## No need to modify afterwards  
    if image_set == "test":
        eval_map = True
        eval_loc = False
    else:
        eval_map = False
        eval_loc = True    

    if eval_map:
        do_python_eval(image_set=image_set, year=year, 
            output_dir = map_output_dir, detpath = detpath)
    if eval_loc:
        do_eval_discovery(image_set=image_set, year=year,
            output_dir = loc_output_dir, detpath = detpath)         

if __name__=="__main__":
    eval_voc_result(model_name = model_name, model_extra = model_extra,
        image_set = image_set, year = year)