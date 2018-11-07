#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'src')
import os
import tarfile
from six.moves import urllib
import numpy as np
import tensorflow as tf
import yaml
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import gridspec
from matplotlib import pyplot as plt
from helper import FPS2, WebcamVideoStream
from skimage import measure
from PIL import Image

import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

## LOAD CONFIG PARAMS ##
VIDEO_INPUT     = 0
FPS_INTERVAL    = 5
ALPHA           = 0.3
MODEL_NAME      = "mobilenetv2"
MODEL_PATH      = "models/mobilenetv2/frozen_inference_graph.pb"
DOWNLOAD_BASE   = "None"
BBOX            = False
MINAREA         = 1000

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

def cal_bin_mask(seg_map):
    m, n = seg_map.shape
    count = 0
    for i in range(m):
        for j in range(n):
            if(seg_map[i][j] != 15):
                seg_map[i][j] = 0
            else:
                count += 1
                seg_map[i][j] = 255
    return seg_map

def create_PASCAL_colormap(seg_map):
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap[seg_map]

def create_colormap(seg_map):
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    m, n = seg_map.shape
    for i in range(m):
        for j in range(n):
            if(seg_map[i][j] != 15):
                seg_map[i][j] = 0
    return colormap[seg_map]

def vis_segmentation(image, seg_map):
    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = create_PASCAL_colormap(FULL_LABEL_MAP)

    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = create_colormap(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)

    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(list(range(len(unique_labels))), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.savefig('segment.png')

def load_frozenmodel():
    print('> Loading frozen model into memory')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        seg_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            seg_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(seg_graph_def, name='')
    return detection_graph

def segmentation_image(detection_graph, label_names, image_path):
    print("log: Begin to do image segmentation")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image = cv2.imread(image_path)
            width, height, _ = image.shape
            resize_ratio = 1.0 * 513/ max(width, height)
            target_size = (int(resize_ratio * image.shape[1]), int(resize_ratio * image.shape[0]))
            image = cv2.resize(image, target_size)
            batch_seg_map = sess.run('SemanticPredictions:0', feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})
            seg_map = batch_seg_map[0]
            vis_segmentation(image, seg_map)
            seg_image = create_colormap(seg_map).astype(np.uint8)

            cv2.addWeighted(seg_image,ALPHA,image,1-ALPHA,0,image)
            #cv2.imshow('segmentation',image)
            #k = cv2.waitKey(0)
            bin_mask = cal_bin_mask(seg_map)
            bin_mask = bin_mask / 255
            bin_mask = cv2.resize(bin_mask, (height, width))

    return bin_mask

def style_transfer(check_point, in_path, out_path, device_t='/gpu:0', batch_size=1):
    print("log: Begin to style transfer")
    img_shape = get_img(in_path).shape
    g = tf.Graph()
    curr_num = 0
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
        batch_shape = (1,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(check_point):
            ckpt = tf.train.get_checkpoint_state(check_point)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, check_point)

        img = get_img(in_path)
        _preds = sess.run(preds, feed_dict={img_placeholder:[img]})
        save_img(out_path, _preds[0])
    return _preds[0]

from scipy.ndimage.morphology import distance_transform_edt as euc_dist
def blend_image_pair(src_img, src_mask, dst_img, dst_mask):
    m, n = src_img.shape[0], src_img.shape[1]
    left_src, left_dst, right_src, right_dst = n, n, 0, 0
    blend_img = np.zeros([src_img.shape[0], src_img.shape[1], src_img.shape[2]])
    for i in range(m):
        for j in range(n):
            if(src_mask[i][j] > 0):
                left_src = min(left_src, j)
            if(dst_mask[i][j] > 0):
                left_dst = min(left_dst, j)
    for i in range(m):
        for j in range(n, 0, -1):
            if(src_mask[i][j-1] > 0):
                right_src = max(right_src, j-1)
            if(dst_mask[i][j-1] > 0):
                right_dst = max(right_dst, j-1)
                
    left_idx = max(left_src, left_dst)
    right_idx = max(min(right_src, right_dst), left_idx + 1)
    if(left_idx == left_src):
        dist = np.concatenate([np.zeros([m, left_idx]), np.ones([m, n - left_idx])], axis=1)
    else:
        dist = np.concatenate([np.ones([m, right_idx]), np.zeros([m, n - right_idx])], axis=1)

    blend_src_mask = np.maximum(np.minimum(euc_dist(dist) / (right_idx - left_idx + 1), 1), src_mask - dst_mask)
    blend_dst_mask = np.maximum(1 - blend_src_mask, dst_mask - src_mask)
    for i in range(m):
        for j in range(n):
            for k in range(3):
                blend_img[i][j][k] = src_img[i][j][k] * blend_src_mask[i][j] + dst_img[i][j][k] * blend_dst_mask[i][j] 
    return blend_img.astype("uint8")

def binary_mask(img):
    mask = np.zeros([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j][0] > 0 and img[i][j][1] > 0 and img[i][j][2] > 0 ):
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    return mask

def blend_images(fg_path, bg_path, mask):
    print("log: Begin to blend images")
    fg = cv2.imread(fg_path)
    bg = cv2.imread(bg_path)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    #for i in range(mask.shape[0]):
    #   for j in range(mask.shape[1]):
    
    mask = np.stack((mask,) * 3, -1)
    out_image = bg
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(3):
                out_image[i][j][k] = (bg[i][j][k] * (1 - mask[i][j][k])) + (fg[i][j][k] * (mask[i][j][k]))
    #out_image = blend_image_pair(fg, binary_mask(fg), bg, binary_mask(bg))
    
    #cv2.imshow("fg", fg)
    #cv2.imshow("bg", bg)
    #cv2.imshow("mask", mask)
    #cv2.imshow("out", out_image)
    return out_image

if __name__ == '__main__':
    image_name = "hong_ps"
    image_suffix = "jpg"
    style = "wreck"

    graph = load_frozenmodel()
    bin_mask = segmentation_image(graph, LABEL_NAMES, image_path="./sample_images/%s.%s"%(image_name, image_suffix))
    style_transfer("./models/%s.ckpt"%(style), "./sample_images/%s.%s"%(image_name, image_suffix), "./outputs/%s_%s.%s"%(image_name, style, image_suffix))
    blend_img = blend_images(fg_path="./sample_images/%s.%s"%(image_name, image_suffix), bg_path="./outputs/%s_%s.%s"%(image_name, style, image_suffix), mask=bin_mask)
    cv2.imwrite("./outputs/blend_%s_%s.%s"%(image_name, style, image_suffix), blend_img)