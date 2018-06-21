import xml.etree.ElementTree as ET
import sys,os
import cPickle
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import uuid
import pdb
from os.path import expanduser
from bbox_util import bbox_overlaps
# from eval_model_recall import num_proposal
num_proposal=[64,128,256,512,1024]

home = expanduser("~")

model_name = sys.argv[1] if len(sys.argv) > 1 else '1shot'
use_difficult = sys.argv[2] if len(sys.argv) > 2 else 'false'
image_set = sys.argv[3] if len(sys.argv) > 3 else 'test'
year = sys.argv[4] if len(sys.argv) > 4 else '2007'

respath = os.path.join('results','voc{}'.format(year))
devkit_path = os.path.join(home,'data','VOCdevkit')
output_dir = os.path.join(respath, 'proposal_output', model_name)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def pro_eval(detfile,
             annopath,
             imagesetfile,
             cachedir,
             thre_list=np.arange(0.5,0.95,0.05)):

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, '{}_{}_annots.pkl'.format(year, image_set))
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename]]
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)        
        if use_difficult == 'true':
            bbox = np.array([x['bbox'] for x in R ])
            npos += len(R)
        else:
            bbox = np.array([x['bbox'] for x in R if x['difficult'] == 0])
            npos += sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult}


    # read dets
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    det_bb = {}
    for ele in splitlines:
        img_name = ele[0]
        if img_name not in det_bb.keys():
            det_bb[img_name] = []
        det_bb[img_name].append(np.array([float(z) for z in ele[2:]]))

    gt_overlaps = np.zeros(npos)
    for i, img in enumerate(imagenames):
        bb = np.array(det_bb[img]).astype(float)
        gt_bb = class_recs[img]['bbox'].astype(float)
        overlaps = bbox_overlaps(bb.astype(np.float),
            gt_bb.astype(np.float))
        gt_overlaps = np.append(gt_overlaps, np.amax(overlaps, axis=0))
        if i % 1000 == 0:
            print 'compute recall for img:{}'.format(i)

    recalls = np.zeros_like(thre_list)
    for i, t in enumerate(thre_list):
        recalls[i] = (gt_overlaps >= t ).sum() / float(npos)

    return recalls
def eval_recall(model_name=model_name, num_proposal=num_proposal,
    use_difficult=use_difficult, image_set=image_set, year=year):
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
    cachedir = os.path.join(devkit_path, 'pro_annotations_cache')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for ele in num_proposal:
        detfile = os.path.join(respath, model_name +'_'+ image_set, 
            '{}_proposal_out.txt'.format(ele))
        outfile = os.path.join(output_dir, '{}_{}_recall_result.txt'.format(ele, use_difficult))

        thre_list = np.arange(0.5,0.95,0.05)
        recalls = pro_eval(detfile, annopath, imagesetfile, cachedir,
         thre_list = thre_list)    
        result_str = ''
        for i in  xrange(thre_list.size):
            result_str += 'Proposal Recall for {} at {} @{}= {}\n'.format(model_name,
                ele, thre_list[i], recalls[i])
        with open(outfile, 'w') as f:
            f.write(result_str)
        print result_str


if __name__=="__main__":
    eval_recall(model_name = model_name, num_proposal=num_proposal,
        use_difficult = use_difficult, 
        image_set = image_set, year = year)