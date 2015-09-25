import numpy as np
import matplotlib.pyplot as plt
from nnreader import *
# Make sure that caffe is on the python path:
import sys
import config
sys.path.insert(0, config.CAFFE_ROOT + 'python')
import caffe

DEBUG=False

class CaffeReader(NNReader):
    def __init__(self):
        caffe.set_mode_cpu()
    def load(self, prototxt, caffemodel, mean=''):        
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        try:
            self.transformer.set_mean('data', np.load(mean).mean(1).mean(1)) # mean pixel
        except:
            print "Warning: No mean file or invalid mean file was provided"
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        #if self.net.blobs
        if self.net.blobs['data'].shape[1] == 3:
            self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    def vis_square(self, data, padsize=1, padval=0):
        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
        
        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:]) 

        return data
    def get_layer_list(self):
        return [k for k, v in self.net.params.items()]
    def normalize_filters(self, filters):
        filters -= filters.min()
        filters /= filters.max()
        return filters    
    def get_filters(self, layer_name):
        data = self.net.params[layer_name][0].data
        if data.shape[1] == 3:     
            return self.normalize_filters(data.transpose(0,2,3,1))
        else:
            return self.normalize_filters(data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3])))
    def forward_image(self, image, layer_name='data'):
        self.net.blobs[layer_name].data[...] = self.transformer.preprocess(layer_name, caffe.io.load_image(image))
        self.net.forward() 
    def get_features(self, layer_name):
        return self.normalize_filters(self.net.blobs[layer_name].data[0, :])
    def get_input(self, layer_name='data'):
        return self.transformer.deprocess('data', self.net.blobs[layer_name].data[0])
        
if DEBUG:
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    CR = CaffeReader()
    CR.load('/home/prlz7/code/caffe-branchnet/examples/mnist/lenet.prototxt',
                    '/home/prlz7/code/caffe-branchnet/examples/mnist/mnist_original/lenet_iter_10000.caffemodel')
    plt.imshow(CR.vis_square(CR.get_filters('conv1')))
    
    CR.forward_image('/media/prlz7/Dades/Pau/DB/cifar10/images/abandoned_ship_s_000260.png')
    plt.imshow(CR.vis_square(CR.get_features('pool1')))

#
#import caffe
#

#
#import os
#
#caffe.set_mode_cpu()
#net = caffe.Net('/home/prlz7/code/caffe/examples/cifar10/cifar10_full_multi/cifar10_full.prototxt',
#                '/home/prlz7/code/caffe/examples/cifar10/cifar10_full_multi/cifar10_full_iter_70000.caffemodel',
#                caffe.TEST)
#
## input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
#
#def normalize_filters(filters):
#    filters -= filters.min()
#    filters /= filters.max()
#    return filters
#
#def get_filters_layer(name):
#    return net.params['layer'][0].data
#           
## take an array of shape (n, height, width) or (n, height, width, channels)
## and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
#def vis_square(data, padsize=1, padval=0):
#   
#    # force the number of filters to be square
#    n = int(np.ceil(np.sqrt(data.shape[0])))
#    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
#    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
#    
#    # tile the filters into an image
#    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
#    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
#    
#    plt.imshow(data)
#
#    
#
## the parameters are a list of [weights, biases]
#filters = net.params['conv1'][0].data
#vis_square(normalize_filters(filters.transpose(0,2,3,1)))
#
#net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('/media/prlz7/Dades/Pau/DB/cifar10/images/abandoned_ship_s_000260.png'))
#out = net.forward()
#print("Predicted class is #{}.".format(out['prob'].argmax()))
#
#plt.figure()
#feat = normalize_filters(net.blobs['conv3'].data[0, :])
#vis_square(feat, padval=1)