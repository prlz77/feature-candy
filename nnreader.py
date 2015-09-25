# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 13:58:16 2015

@author: prlz7
"""
import abc
class NNReader(object):
    __metaclass__  = abc.ABCMeta    
    def __init__(self):
        pass
    @abc.abstractmethod
    def load(self):
        pass
    @abc.abstractmethod
    def vis_square(self, data, padsize, padval):
        pass
    @abc.abstractmethod
    def get_layer_list(self, layer):
        pass    
    @abc.abstractmethod
    def get_filters(self, layer_name):
        pass    
    @abc.abstractmethod
    def forward_image(self, image, layer_name):
        pass   
    @abc.abstractmethod
    def get_features(self, layer):
        pass    