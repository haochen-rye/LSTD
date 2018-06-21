import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import math
import pdb

NUM_PROPOSAL = 128

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
    if_propagate=True,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=lr_mult)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    eps = bn_params.get('eps', 0.001)
    moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
    use_global_stats = bn_params.get('use_global_stats', False)
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        'moving_average_fraction': moving_average_fraction,
        }
    bn_lr_mult = lr_mult
    if use_global_stats:
      # only specify if use_global_stats is explicitly provided;
      # otherwise, use_global_stats_ = this->phase_ == TEST;
      bn_kwargs = {
          'param': [
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0)],
          'eps': eps,
          'use_global_stats': use_global_stats,
          }
      # not updating scale/bias parameters
      bn_lr_mult = 0
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult= lr_mult),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, 
        propagate_down=[if_propagate],**kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, 
        propagate_down=[if_propagate],**kwargs)
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1, **bn_param):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  if dilation == 1:
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  else:
    pad = int((3 + (dilation - 1) * 2) - 1) / 2
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
        dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)


def InceptionTower(net, from_layer, tower_name, layer_params, **bn_param):
  use_scale = False
  for param in layer_params:
    tower_layer = '{}/{}'.format(tower_name, param['name'])
    del param['name']
    if 'pool' in tower_layer:
      net[tower_layer] = L.Pooling(net[from_layer], **param)
    else:
      param.update(bn_param)
      ConvBNLayer(net, from_layer, tower_layer, use_bn=True, use_relu=True,
          use_scale=use_scale, **param)
    from_layer = tower_layer
  return net[from_layer]

# def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
#         output_label=True, train=True, label_map_file='', anno_type=None,
#         transform_param={}, batch_sampler=[{}]):
#     if train:
#         kwargs = {
#                 'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
#                 'transform_param': transform_param,
#                 }
#     else:
#         kwargs = {
#                 'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
#                 'transform_param': transform_param,
#                 }
#     ntop = 1
#     if output_label:
#         ntop = 2
#     annotated_data_param = {
#         'label_map_file': label_map_file,
#         'batch_sampler': batch_sampler,
#         }
#     if anno_type is not None:
#         annotated_data_param.update({'anno_type': anno_type})
#     return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
#         data_param=dict(batch_size=batch_size, backend=backend, source=source),
#         ntop=ntop, **kwargs)
def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='', orient_anno=False,
         anno_type=None,
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    ntop = 1
    if output_label:
        ntop = 2
    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
        'orient_anno': orient_anno,
        }
    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})
    return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
        data_param=dict(batch_size=batch_size, backend=backend, source=source),
        ntop=ntop, **kwargs)

def ZFNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, dropout=True, need_fc8=False, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1 = L.Convolution(net[from_layer], num_output=96, pad=3, kernel_size=7, stride=2, **kwargs)
    net.relu1 = L.ReLU(net.conv1, in_place=True)

    net.norm1 = L.LRN(net.relu1, local_size=3, alpha=0.00005, beta=0.75,
            norm_region=P.LRN.WITHIN_CHANNEL, engine=P.LRN.CAFFE)

    net.pool1 = L.Pooling(net.norm1, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=2)

    net.conv2 = L.Convolution(net.pool1, num_output=256, pad=2, kernel_size=5, stride=2, **kwargs)
    net.relu2 = L.ReLU(net.conv2, in_place=True)

    net.norm2 = L.LRN(net.relu2, local_size=3, alpha=0.00005, beta=0.75,
            norm_region=P.LRN.WITHIN_CHANNEL, engine=P.LRN.CAFFE)

    net.pool2 = L.Pooling(net.norm2, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=2)

    net.conv3 = L.Convolution(net.pool2, num_output=384, pad=1, kernel_size=3, **kwargs)
    net.relu3 = L.ReLU(net.conv3, in_place=True)
    net.conv4 = L.Convolution(net.relu3, num_output=384, pad=1, kernel_size=3, **kwargs)
    net.relu4 = L.ReLU(net.conv4, in_place=True)
    net.conv5 = L.Convolution(net.relu4, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu5 = L.ReLU(net.conv5, in_place=True)

    if need_fc:
        if dilated:
            name = 'pool5'
            net[name] = L.Pooling(net.relu5, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            name = 'pool5'
            net[name] = L.Pooling(net.relu5, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=5, kernel_size=3, dilation=5, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=5, kernel_size=6, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=2, kernel_size=3, dilation=2,  **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=2, kernel_size=6, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
    if need_fc8:
        from_layer = net.keys()[-1]
        if fully_conv:
            net.fc8 = L.Convolution(net[from_layer], num_output=1000, kernel_size=1, **kwargs)
        else:
            net.fc8 = L.InnerProduct(net[from_layer], num_output=1000)
        net.prob = L.Softmax(net.fc8)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net

def SSD_FIX(net,from_layer='data', lr_mult = 0, num_classes=21,
  test_phase=False,
  use_proposal=True, proposal_postfix='', use_priorbox=True):
    kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=lr_mult), 
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    # net.conv1_1_fix = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)
    # net.relu1_1_fix = L.ReLU(net.conv1_1_fix, in_place=True)
    # net.conv1_2_fix = L.Convolution(net.relu1_1_fix, num_output=64, pad=1, kernel_size=3, **kwargs)
    # net.relu1_2_fix = L.ReLU(net.conv1_2_fix, in_place=True)
    # net.pool1_fix = L.Pooling(net.relu1_2_fix, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # net.conv2_1_fix = L.Convolution(net.pool1_fix, num_output=128, pad=1, kernel_size=3, **kwargs)
    # net.relu2_1_fix = L.ReLU(net.conv2_1_fix, in_place=True)
    # net.conv2_2_fix = L.Convolution(net.relu2_1_fix, num_output=128, pad=1, kernel_size=3, **kwargs)
    # net.relu2_2_fix = L.ReLU(net.conv2_2_fix, in_place=True)
    # net.pool2_fix = L.Pooling(net.relu2_2_fix, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # net.conv3_1_fix = L.Convolution(net.pool2_fix, num_output=256, pad=1, kernel_size=3, **kwargs)
    # net.relu3_1_fix = L.ReLU(net.conv3_1_fix, in_place=True)
    # net.conv3_2_fix = L.Convolution(net.relu3_1_fix, num_output=256, pad=1, kernel_size=3, **kwargs)
    # net.relu3_2_fix = L.ReLU(net.conv3_2_fix, in_place=True)
    # net.conv3_3_fix = L.Convolution(net.relu3_2_fix, num_output=256, pad=1, kernel_size=3, **kwargs)
    # net.relu3_3_fix = L.ReLU(net.conv3_3_fix, in_place=True)
    # net.pool3_fix = L.Pooling(net.relu3_3_fix, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1_fix = L.Convolution(net.pool3, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1_fix = L.ReLU(net.conv4_1_fix, in_place=True)
    net.conv4_2_fix = L.Convolution(net.relu4_1_fix, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2_fix = L.ReLU(net.conv4_2_fix, in_place=True)
    net.conv4_3_fix = L.Convolution(net.relu4_2_fix, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3_fix = L.ReLU(net.conv4_3_fix, in_place=True)
    net.pool4_fix = L.Pooling(net.relu4_3_fix, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv5_1_fix = L.Convolution(net.pool4_fix, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1_fix = L.ReLU(net.conv5_1_fix, in_place=True)
    net.conv5_2_fix = L.Convolution(net.relu5_1_fix, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2_fix = L.ReLU(net.conv5_2_fix, in_place=True)
    net.conv5_3_fix = L.Convolution(net.relu5_2_fix, num_output=512, pad=1, kernel_size=3,**kwargs)
    net.relu5_3_fix = L.ReLU(net.conv5_3_fix, in_place=True)
    net.pool5_fix = L.Pooling(net.relu5_3_fix, pool=P.Pooling.MAX, kernel_size=3, pad=1)
    net.fc6_fix = L.Convolution(net.pool5_fix, num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
    net.relu6_fix = L.ReLU(net.fc6_fix, in_place=True)
    net.fc7_fix = L.Convolution(net.relu6_fix, num_output=1024, kernel_size=1, **kwargs)
    net.relu7_fix = L.ReLU(net.fc7_fix, in_place=True)
    use_batchnorm=False
    use_relu=True
    from_layer='relu7_fix'
    out_layer = "conv6_1_fix"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2_fix"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1_fix"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2_fix"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1_fix"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2_fix"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_1_fix"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2_fix"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    min_dim = 300
    mbox_source_layers = ['conv4_3_fix', 'fc7_fix', 'conv6_2_fix', 'conv7_2_fix',
     'conv8_2_fix', 'conv9_2_fix']
    # in percent %
    min_ratio = 15
    max_ratio = 90
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
      min_sizes.append(min_dim * ratio / 100.)
      max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 7/100.] + min_sizes
    max_sizes = [min_dim * 15/100] + max_sizes
    steps = [8, 16, 32, 64, 100, 300]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    # L2 normalize conv4_3.
    normalizations = [20, -1, -1, -1, -1, -1]
    # variance used to encode/decode prior bboxes.
    prior_variance = [0.1, 0.1, 0.2, 0.2]

    flip = True
    clip = False    

    mbox_layers, object_layers= CreateMultiBoxHead(net,num_classes=num_classes,
        loc_lr_mult=0, multi_lr_mult=0, obj_lr_mult=0, use_objectness=True,
        use_priorbox=use_priorbox,conf_postfix='_fix',loc_postfix='_fix',
        obj_postfix='_fix', use_proposal=use_proposal,
        test_phase=test_phase,
        proposal_postfix=proposal_postfix, use_batchnorm=False,
        from_layers=mbox_source_layers, min_sizes=min_sizes, img_height=min_dim,
        img_width=min_dim, normalizations=normalizations, flip=flip, clip=clip,
        prior_variance=prior_variance,kernel_size=3, pad=1, 
        max_sizes=max_sizes, aspect_ratios=aspect_ratios, steps=steps)

    return net

def SSD_SOURCE(net,from_layer='data', lr_mult = 0, num_classes=21,
  test_phase=False,
  use_proposal=True, proposal_postfix='', use_priorbox=False):
    kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=lr_mult), 
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    # net.conv1_1_source = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)
    # net.relu1_1_source = L.ReLU(net.conv1_1_source, in_place=True)
    # net.conv1_2_source = L.Convolution(net.relu1_1_source, num_output=64, pad=1, kernel_size=3, **kwargs)
    # net.relu1_2_source = L.ReLU(net.conv1_2_source, in_place=True)
    # net.pool1_source = L.Pooling(net.relu1_2_source, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # net.conv2_1_source = L.Convolution(net.pool1_source, num_output=128, pad=1, kernel_size=3, **kwargs)
    # net.relu2_1_source = L.ReLU(net.conv2_1_source, in_place=True)
    # net.conv2_2_source = L.Convolution(net.relu2_1_source, num_output=128, pad=1, kernel_size=3, **kwargs)
    # net.relu2_2_source = L.ReLU(net.conv2_2_source, in_place=True)
    # net.pool2_source = L.Pooling(net.relu2_2_source, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # net.conv3_1_source = L.Convolution(net.pool2_source, num_output=256, pad=1, kernel_size=3, **kwargs)
    # net.relu3_1_source = L.ReLU(net.conv3_1_source, in_place=True)
    # net.conv3_2_source = L.Convolution(net.relu3_1_source, num_output=256, pad=1, kernel_size=3, **kwargs)
    # net.relu3_2_source = L.ReLU(net.conv3_2_source, in_place=True)
    # net.conv3_3_source = L.Convolution(net.relu3_2_source, num_output=256, pad=1, kernel_size=3, **kwargs)
    # net.relu3_3_source = L.ReLU(net.conv3_3_source, in_place=True)
    # net.pool3_source = L.Pooling(net.relu3_3_source, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv4_1_source = L.Convolution(net.pool3, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1_source = L.ReLU(net.conv4_1_source, in_place=True)
    net.conv4_2_source = L.Convolution(net.relu4_1_source, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2_source = L.ReLU(net.conv4_2_source, in_place=True)
    net.conv4_3_source = L.Convolution(net.relu4_2_source, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3_source = L.ReLU(net.conv4_3_source, in_place=True)
    net.pool4_source = L.Pooling(net.relu4_3_source, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv5_1_source = L.Convolution(net.pool4_source, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1_source = L.ReLU(net.conv5_1_source, in_place=True)
    net.conv5_2_source = L.Convolution(net.relu5_1_source, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2_source = L.ReLU(net.conv5_2_source, in_place=True)
    net.conv5_3_source = L.Convolution(net.relu5_2_source, num_output=512, pad=1, kernel_size=3,**kwargs)
    net.relu5_3_source = L.ReLU(net.conv5_3_source, in_place=True)
    net.pool5_source = L.Pooling(net.relu5_3_source, pool=P.Pooling.MAX, kernel_size=3, pad=1)
    net.fc6_source = L.Convolution(net.pool5_source, num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
    net.relu6_source = L.ReLU(net.fc6_source, in_place=True)
    net.fc7_source = L.Convolution(net.relu6_source, num_output=1024, kernel_size=1, **kwargs)
    net.relu7_source = L.ReLU(net.fc7_source, in_place=True)
    use_batchnorm=False
    use_relu=True
    from_layer='relu7_source'
    out_layer = "conv6_1_source"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2_source"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1_source"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2_source"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1_source"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2_source"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_1_source"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2_source"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    min_dim = 300
    mbox_source_layers = ['conv4_3_source', 'fc7_source', 'conv6_2_source', 'conv7_2_source',
     'conv8_2_source', 'conv9_2_source']
    # in percent %
    min_ratio = 15
    max_ratio = 90
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
      min_sizes.append(min_dim * ratio / 100.)
      max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 7/100.] + min_sizes
    max_sizes = [min_dim * 15/100] + max_sizes
    steps = [8, 16, 32, 64, 100, 300]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    # L2 normalize conv4_3.
    normalizations = [20, -1, -1, -1, -1, -1]
    # variance used to encode/decode prior bboxes.
    prior_variance = [0.1, 0.1, 0.2, 0.2]

    flip = True
    clip = False    

    mbox_layers, object_layers= CreateMultiBoxHead(net,num_classes=num_classes,
        loc_lr_mult=0, multi_lr_mult=0, obj_lr_mult=0, use_objectness=True,
        use_priorbox=use_priorbox,conf_postfix='_source',loc_postfix='_source',
        obj_postfix='_source', use_proposal=use_proposal,
        test_phase=test_phase,
        proposal_postfix=proposal_postfix, use_batchnorm=False,
        from_layers=mbox_source_layers, min_sizes=min_sizes, img_height=min_dim,
        img_width=min_dim, normalizations=normalizations, flip=flip, clip=clip,
        prior_variance=prior_variance,kernel_size=3, pad=1, 
        max_sizes=max_sizes, aspect_ratios=aspect_ratios, steps=steps)

    return net

def VGGNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        lr_mult=1,
        dilated=False, nopool=False, dropout=True, freeze_layers=[], dilate_pool4=False):
    kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=lr_mult), 
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        if dilate_pool4:
            net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
            net.fc6 = L.Convolution(net[name], num_output=num_output, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net




def ResNet101Body(net, from_layer, use_pool5=True, use_dilation_conv5=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True, **bn_param)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res3a'
    for i in xrange(1, 4):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res4a'
    for i in xrange(1, 23):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation, **bn_param)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def ResNet152Body(net, from_layer, use_pool5=True, use_dilation_conv5=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True, **bn_param)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res3a'
    for i in xrange(1, 8):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res4a'
    for i in xrange(1, 36):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation, **bn_param)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def InceptionV3Body(net, from_layer, output_pred=False, **bn_param):
  # scale is fixed to 1, thus we ignore it.
  use_scale = False

  out_layer = 'conv'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=2, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_1'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_2'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=64, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'pool'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  out_layer = 'conv_3'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=80, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_4'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=192, kernel_size=3, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'pool_1'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  # inceptions with 1x1, 3x3, 5x5 convolutions
  for inception_id in xrange(0, 3):
    if inception_id == 0:
      out_layer = 'mixed'
      tower_2_conv_num_output = 32
    else:
      out_layer = 'mixed_{}'.format(inception_id)
      tower_2_conv_num_output = 64
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=48, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=64, kernel_size=5, pad=2, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
        dict(name='conv_2', num_output=96, kernel_size=3, pad=1, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=tower_2_conv_num_output, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3(in sequence) convolutions
  out_layer = 'mixed_3'
  towers = []
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=384, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
      dict(name='conv_2', num_output=96, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  # inceptions with 1x1, 7x1, 1x7 convolutions
  for inception_id in xrange(4, 8):
    if inception_id == 4:
      num_output = 128
    elif inception_id == 5 or inception_id == 6:
      num_output = 160
    elif inception_id == 7:
      num_output = 192
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_2', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_3', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_4', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3, 1x7, 7x1 filters
  out_layer = 'mixed_8'
  towers = []
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=320, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}/tower_1'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
      dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
      dict(name='conv_3', num_output=192, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  for inception_id in xrange(9, 11):
    num_output = 384
    num_output2 = 448
    if inception_id == 9:
      pool = P.Pooling.AVE
    else:
      pool = P.Pooling.MAX
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)

    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ], **bn_param)
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  if output_pred:
    net.pool_3 = L.Pooling(net[from_layer], pool=P.Pooling.AVE, kernel_size=8, pad=0, stride=1)
    net.softmax = L.InnerProduct(net.pool_3, num_output=1008)
    net.softmax_prob = L.Softmax(net.softmax)

  return net

def CreateMultiOrientBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='', rotation_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    rotation_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create rotation prediction layer
        name = "{}_mbox_rotation{}".format(from_layer, rotation_postfix)
        num_rotation_output = num_priors_per_location ;
        if not share_location:
            num_rotation_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_rotation_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        rotation_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_rotation"
    net[name] = L.Concat(*rotation_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_multiclass=False,  use_priorbox=True, use_loc=True,
        loc_lr_mult=1, multi_lr_mult=1, obj_lr_mult=1,
        test_phase=False,
        weakly_regulize_feature=False, obj_loc_propagate_down=True,
        use_proposal = False, proposal_object_type = 'SOFTMAX',
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='',obj_postfix='', proposal_postfix='',
        **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=loc_lr_mult,
              num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, 
              if_propagate=obj_loc_propagate_down, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        if  use_multiclass:
            # Create confidence prediction layer.
            name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
            num_conf_output = num_priors_per_location * num_classes;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, 
              lr_mult=multi_lr_mult,
                num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        if use_priorbox:
          name = "{}_mbox_priorbox".format(from_layer)
          net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                  clip=clip, variance=prior_variance, offset=offset)
          if max_size:
              net.update(name, {'max_size': max_size})
          if aspect_ratio:
              net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
          if step:
              net.update(name, {'step': step})
          if img_height != 0 and img_width != 0:
              if img_height == img_width:
                  net.update(name, {'img_size': img_height})
              else:
                  net.update(name, {'img_h': img_height, 'img_w': img_width})
          priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, 
              lr_mult=obj_lr_mult, if_propagate=obj_loc_propagate_down,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

        # if weakly_regulize_feature:
        #   cls_name = "{}_class_conf".format(from_layer)
        #   ConvBNLayer(net, from_layer, cls_name, use_bn=use_batchnorm, 
        #     use_relu=True, lr_mult=1, num_output=num_classes - 1,
        #     kernel_size=3, pad=1, stride=1, **bn_param)
        #   pool_name = "{}_global_pol".format(from_layer)
        #   net[pool_name] = L.Pooling(net[cls_name],pool=P.Pooling.MAX,
        #     global_pooling=True)
        #   loss_name = "{}_weakly_loss".format(from_layer)
        #   net[loss_name] = L.MulticlassSigmoidLoss(net[pool_name],
        #             net.multi_label ,include_background=False) 


    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    object_out_layers = []

    name = "mbox_loc{}".format(loc_postfix)
    if use_loc:
      net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    object_out_layers.append(net[name])

    if use_multiclass:
        name = "mbox_conf{}".format(cong_postfix)
        net[name] = L.Concat(*conf_layers, axis=1)
        mbox_layers.append(net[name])

    if use_objectness:
        name = "mbox_objectness{}".format(obj_postfix)
        net[name] = L.Concat(*objectness_layers, axis=1)
        object_out_layers.append(net[name])

    name = "mbox_priorbox"
    if use_priorbox:
      net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    object_out_layers.append(net[name])

    code_type = P.PriorBox.CENTER_SIZE
    num_proposal = NUM_PROPOSAL
    top_k = num_proposal * 8
    if test_phase:
      num_proposal = 128
      top_k = 1000

    obj_score_layer=[]
    if use_proposal:
        proposal_param = {
        'num_classes': 2,
        'share_location': True,
        'background_label_id': 0,
        'nms_param': {'nms_threshold': 0.75, 'top_k': top_k},
        'keep_top_k': num_proposal,
        'code_type': code_type,
        'use_as_proposal': True,
        } 

        conf_name = "mbox_objectness{}".format(obj_postfix)
        if proposal_object_type == "SOFTMAX":
            reshape_name = "{}_reshape".format(conf_name)
            net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, 2]))
            softmax_name = "{}_softmax".format(conf_name)
            net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
            flatten_name = "{}_flatten".format(conf_name)
            net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
            from_name = flatten_name
            obj_score_layer.append(flatten_name)
        elif proposal_object_type == "LOGISTIC":
            sigmoid_name = "{}_sigmoid".format(conf_name)
            net[sigmoid_name] = L.Sigmoid(net[conf_name])  
            from_name = sigmoid_name  
            obj_score_layer.append(flatten_name)

        loc_from_layer = "mbox_loc{}".format(loc_postfix)
        proposal_layer = "proposal{}".format(proposal_postfix)
        net[proposal_layer]=L.DetectionOutput(net[loc_from_layer],net[from_name],net.mbox_priorbox,
          detection_output_param=proposal_param,propagate_down=[False,False,False])

    return mbox_layers, object_out_layers

def PostClassifier(net, roi_layer='fc7', num_classes=21, num_oicr=1,
  use_kd=False, kd_classes=61, kd_loss_weight=1,
  min_overlap = 0.1, include_ohem = True, del_neg=False,
  use_oicr=False, train_phase=True, use_wsdnn=False, output_bbox=False,
  num_rpn=1, num_proposal=NUM_PROPOSAL, inter_lr_mult=1, use_global_sum_pool=False,
  classifier_postfix=''):

  net.roi_pooling=L.ROIPooling(net[roi_layer],net.proposal,
      pooled_w=5,pooled_h=5,spatial_scale=19)
  add_layers = ["conv_add1","conv_add2"]

  stride = [1, 2]
  output_num = [256, 256]
  assert len(add_layers) == len(output_num)
  post_inter_lr_mult = 1    

  from_layer = "roi_pooling"
  for i in xrange(len(add_layers)):
    ConvBNLayer(net,  from_layer,  add_layers[i], True,  True, output_num[i], 3, 1, stride[i], 
      lr_mult=inter_lr_mult)
    from_layer = add_layers[i]

  # oicr_prob_layers = []fann
  if use_kd:
    bridge_name = "bridge_score"
    bridge_temp = bridge_name + "_temperature"
    kd_temp = "kd_score_temperature"
    kd_loss="post_class_kd_loss"
    net[bridge_name] = L.InnerProduct(net[from_layer],num_output=kd_classes)
    net[bridge_temp] = L.Scale(net[bridge_name],
     filler=dict(type='constant',value=0.5),
      param=[dict(lr_mult=0,decay_mult=0)], 
      include=[dict(phase=caffe_pb2.Phase.Value('TRAIN'))])
    net[kd_loss]=L.Softmax(net[bridge_temp],  net[kd_temp],  
      type="WeightedSoftmaxWithCrossEntropyLoss", 
      loss_weight=150,  include=[dict(phase=caffe_pb2.Phase.Value('TRAIN'))],
       propagate_down=[True,False])

  if use_oicr:
    if train_phase:
      if use_wsdnn:
        net.mid_score_proposal = L.InnerProduct(net[from_layer],
            num_output = num_classes -1, param=[dict(lr_mult=1), dict(lr_mult=2)])
        net.mid_score_class = L.InnerProduct(net[from_layer],
            num_output = num_classes -1, param=[dict(lr_mult=1), dict(lr_mult=2)])
        net.proposal_prob = L.Softmax(net.mid_score_proposal, axis = 0)
        net.class_prob = L.Softmax(net.mid_score_class)
        net.proposal_temp_score = L.Eltwise(net.proposal_prob, net.class_prob, 
            operation = P.Eltwise.PROD)
        net.proposal_score = L.Reshape(net.proposal_temp_score, 
            shape=dict(dim=[-1, num_proposal * num_rpn, (num_classes - 1)]) );

        net.global_sum_pool = L.GlobalSumPooling(net.proposal_score, 
          include_background=False,
          num_classes = num_classes - 1, num_proposal=num_proposal * num_rpn);
        # Comute the wealy supervised loss
        net.wsdnn_loss = L.MulticlassCrossEntropyLoss(net.global_sum_pool,
            net.multi_label, include_background=False)   
        # oicr_input = [net.multi_label,net.proposal, net.proposal_score]   

        net.wsdnn_label, net.wsdnn_label_weight = L.OICR( 
            net.multi_label, 
            net.proposal, net.proposal_score , ntop=2,
            num_proposal = num_proposal * num_rpn,
            num_classes = num_classes - 1, min_overlap = 0.3,
            include_background =False, del_neg = False, 
            include_ohem = True, output_bbox=False,
            propagate_down=[False,False,False])

      for i in xrange(num_oicr):
        oicr_score_name = "oicr_score_{}".format(i)
        reshape_name = oicr_score_name + "_reshape"
        prob_name = "oicr_prob_{}".format(i)
        pool_name = "oicr_pool_{}".format(i)
        mil_loss_name = "mil_loss_{}".format(i)
        label_name = "oicr_label_{}".format(i)
        weight_name = "oicr_label_weight_{}".format(i)
        bbox_layer = "oicr_bbox_{}".format(i)
        oicr_loss_weight = 50
        score_lr_mult = 1 if use_kd else 1
        net[oicr_score_name] = L.InnerProduct(net[from_layer], 
              param=[dict(lr_mult=score_lr_mult), 
              dict(lr_mult=2 * score_lr_mult)], 
              num_output=num_classes)
        # reshape the score layer to compute the class_score for every class
        net[reshape_name] = L.Reshape(net[oicr_score_name],
          shape=dict(dim=[-1, num_proposal * num_rpn , num_classes, 1]) );
        net[prob_name] = L.Softmax(net[reshape_name],axis=2)
        # compute the MIL loss for the first score
        if (i == 0):
          # sum the class_score over the proposals and softmax them for loss layer
          if use_wsdnn:
            net.oicr_loss_0 = L.WeightedSoftmaxWithLoss(net[reshape_name],
                net.wsdnn_label, net.wsdnn_label_weight, softmax_param=dict(axis=2),
                propagate_down=[True,False,False], loss_weight=oicr_loss_weight)
          # else:
          if use_global_sum_pool:
            net[pool_name] = L.GlobalSumPooling(net[prob_name], 
              num_classes = num_classes, num_proposal=num_proposal * num_rpn,
              include_background=True);
            # net[pool_name] = L.Reduction(net[oicr_permute_name], axis=2)
            # net[mil_loss_name] = L.MulticlassSigmoidLoss(
            #   net[pool_name], net.multi_label)
            net[mil_loss_name] = L.SigmoidCrossEntropyLoss(
              net[pool_name], net.multi_label )
          else:
            oicr_permute_name = "oicr_permute_{}".format(i)
            net[oicr_permute_name] = L.Permute(net[prob_name], order=[0,2,1,3])
            net[pool_name] = L.Pooling(net[oicr_permute_name], pool=P.Pooling.MAX,
              global_pooling = True )
            net[mil_loss_name] = L.MulticlassCrossEntropyLoss(
              net[pool_name], net.multi_label )


        # compute the lable and weight for oicr
        if output_bbox:
          net[label_name], net[weight_name], net[bbox_layer] = L.OICR(
            net.multi_label,net.proposal, net[prob_name],
            ntop=3, num_proposal = num_proposal * num_rpn,
            num_classes = num_classes, 
            propagate_down=[False,False,True])
          if (i == num_oicr - 1):
            net.silence = L.Silence(net[label_name],net[weight_name], ntop=0)          
        else:
          if (i < num_oicr -1):
            net[label_name], net[weight_name] = L.OICR(
              net.multi_label,net.proposal, net[prob_name],
              ntop=2, num_proposal = num_proposal * num_rpn,
              num_classes = num_classes, min_overlap = 0.3,
              del_neg = False, pos_neg_ratio = 3,
              output_bbox = False, include_ohem=True,
              propagate_down=[False,False,False])     
          else:
            net.silence = L.Silence(net[prob_name], ntop=0)          
        # Compute the OICR loss
        if (i > 0):
          oicr_loss_layer = "oicr_loss_{}".format(i)
          label_name = "oicr_label_{}".format(i - 1)
          weight_name = "oicr_label_weight_{}".format(i - 1)
          net[oicr_loss_layer] = L.WeightedSoftmaxWithLoss(net[reshape_name],
              net[label_name], net[weight_name], softmax_param=dict(axis=2),
              propagate_down=[True,False,False], loss_weight=oicr_loss_weight)

    else:
      conf_name = "oicr_score_{}".format(num_oicr - 1)
      net[conf_name] = L.InnerProduct(net.conv_add2, num_output=num_classes)

  else:
    temp_name = "bbox_score_temp{}".format(classifier_postfix)
    net[temp_name]=L.InnerProduct(net[from_layer], num_output=num_classes,
      param=[dict(lr_mult=1), dict(lr_mult=2)]) 
    net.bbox_score=L.Reshape(net[temp_name],
      shape=dict(dim=[-1, num_proposal * num_rpn * num_classes]))

  return net

def KDPostClassifier(net, roi_layer='fc7_fix', num_classes=21, num_oicr=1,
  kd_classes=61, 
  use_oicr=False, train_phase=True, use_wsdnn=False, output_bbox=False,
  num_rpn=1, num_proposal=NUM_PROPOSAL, inter_lr_mult=1, use_global_sum_pool=False,
  classifier_postfix=''):

  net.roi_pooling_kd=L.ROIPooling(net[roi_layer],net.proposal,
      pooled_w=5,pooled_h=5,spatial_scale=19)
  add_layers = ["conv_add1_kd","conv_add2_kd"]

  stride = [1, 2]
  output_num = [256, 256]
  assert len(add_layers) == len(output_num)
  post_inter_lr_mult = 1    

  from_layer = "roi_pooling_kd"
  for i in xrange(len(add_layers)):
    ConvBNLayer(net,  from_layer,  add_layers[i], True,  True, output_num[i], 3, 1, stride[i], 
      lr_mult=inter_lr_mult)
    from_layer = add_layers[i]

  kd_score_name = "kd_score"
  kd_score_temp = kd_score_name + "_temperature"
  net[kd_score_name] = L.InnerProduct(net[from_layer],num_output=kd_classes,
  param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)])
  net[kd_score_temp] = L.Scale(net[kd_score_name],
   filler=dict(type='constant',value=0.5),
    param=[dict(lr_mult=0,decay_mult=0)], 
    include=[dict(phase=caffe_pb2.Phase.Value('TRAIN'))])

  return net

def OrigCreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers