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
import scipy

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

def vis_segmentation(image, seg_map, segmentation_save_path):
    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = create_PASCAL_colormap(FULL_LABEL_MAP)

    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('Detect image')

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
    plt.savefig(segmentation_save_path)

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

def optimize_boundary(img, mask, depth):
    img_edges = cv2.Canny(img, 200, 250)
    dist_edges, inds = scipy.ndimage.morphology.distance_transform_edt(np.where(img_edges > 0, 0, 1), return_indices = True)
    dist = scipy.ndimage.morphology.distance_transform_edt(mask)
    q = []
    visited = set()
    m, n = mask.shape
    for i in range(m):
        for j in range(n):
            if dist[i, j] == 1:
                q.append([i, j])
                visited.add(i*n+j)
    while q:
        i, j = q.pop(0)
        if dist[i, j] >= depth:
            break
        if dist_edges[i, j] >= depth or dist_edges[i, j] == 0 or dist[inds[0, i, j], inds[1, i, j]] < dist[i, j] or dist[inds[0, i, j], inds[1, i, j]] > 30:
            continue
        mask[i, j] = 0
        for x, y in [[i+1, j], [i-1, j], [i, j+1], [i, j-1]]:
            if x < 0 or x >= m or y < 0 or y >= n or x*n+y in visited or mask[x, y] == 0 or dist[x, y] < dist[i, j]:
                continue
            q.append([x, y])
            visited.add(x*n+y)
    return img_edges, mask

def segmentation_image(detection_graph, label_names, image_path, segmentation_save_path):
    print("log: Begin to do image segmentation")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image = cv2.imread(image_path)
            ori_image = image
            width, height, _ = image.shape
            resize_ratio = 1.0 * 513/ max(width, height)
            target_size = (int(resize_ratio * image.shape[1]), int(resize_ratio * image.shape[0]))
            image = cv2.resize(image, target_size)
            batch_seg_map = sess.run('SemanticPredictions:0', feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})
            seg_map = batch_seg_map[0]
            #vis_segmentation(image, seg_map, segmentation_save_path)
            seg_image = create_colormap(seg_map).astype(np.uint8)

            cv2.addWeighted(seg_image,ALPHA,image,1-ALPHA,0,image)
            #cv2.imshow('segmentation',image)
            #k = cv2.waitKey(0)
            bin_mask = cal_bin_mask(seg_map)
            bin_mask = bin_mask / 255
            bin_mask = cv2.resize(bin_mask, (height, width))
            u = min(height, width) // 30
            img_edges, bin_mask = optimize_boundary(ori_image, bin_mask, u)
            vis_segmentation(ori_image, (bin_mask*15).astype(np.int32), segmentation_save_path)
    return bin_mask

def style_transfer(check_point, in_path, out_path, device_t='/gpu:0', batch_size=1):
    print("log: Begin to style transfer")

    print(check_point, in_path, out_path)
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

def rgb_to_lab(v):
    M1 = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])
    lms = M1.dot(v)
    lms = np.where(lms > 0.0000000001, np.log10(lms), -10)
    M2 = np.array([[0.57735, 0.57735, 0.57735], [0.40825, 0.40825, -0.8165], [0.70711, -0.70711, 0]])
    lab = M2.dot(lms)
    return lab
    
def color_transfer(img_path, style_image_path, style_data_path, output_path):
    mStyle, nStyle = -1, -1
    style = None
    list_style = []
    if os.path.exists(style_data_path):
        for line in open(style_data_path):
            tmp = line.split('\t')
            if mStyle < 0:
                mStyle, nStyle = int(tmp[0].strip()), int(tmp[1].strip())
                style = np.zeros((mStyle, nStyle, 3))
                continue
            i, j, r, g, b = int(tmp[0].strip()), int(tmp[1].strip()), int(tmp[2].strip()), int(tmp[3].strip()), int(tmp[4].strip())
            list_style.append([i, j, 1])
            style[i, j] = [r, g, b]
    else:
        style = cv2.imread(style_image_path)
        mStyle, nStyle = style.shape[0], style.shape[1]
    
        # 1d PCA
        X = np.array([rgb_to_lab(v) for v in style.reshape((mStyle*nStyle, 3))])
        l, v = np.linalg.eig(X.T.dot(X))
        q = v[np.argmax(abs(l))]
        if q.dot([1,1,1]) < 0:
            q = -q
        list_style = [[i, j, rgb_to_lab(style[i, j]).dot(q)] for i in range(mStyle) for j in range(nStyle)]
    
        list_style.sort(key = lambda x:x[2])
        fout = open(style_data_path, 'w')
        fout.write("%d\t%d\n"%(mStyle, nStyle))
        for i, j, _ in list_style:
            fout.write("%d\t%d\t%d\t%d\t%d\n" % (i, j, style[i, j, 0], style[i, j, 1], style[i, j, 2]))
        fout.close()
        

    img = cv2.imread(img_path)
    mImg, nImg = img.shape[0], img.shape[1]
    
    X = np.array([rgb_to_lab(v) for v in img.reshape((mImg*nImg, 3))])
    l, v = np.linalg.eig(X.T.dot(X))
    qImg = v[np.argmax(abs(l))]
    if qImg.dot([1,1,1]) < 0:
        qImg = -qImg

    bucketImg = np.zeros(img.shape[:2])
    bucket_count = {}
    for i in range(mImg):
        for j in range(nImg):
            bucket = int(rgb_to_lab(img[i, j]).dot(qImg) * 10000)
            bucketImg[i, j] = bucket
            bucket_count.setdefault(bucket, 0)
            bucket_count[bucket] += 1
    
    curr = 0
    bucket_mapping = {}
    sorted_buckets = sorted(bucket_count.keys())
    for bucket in sorted_buckets:
        bucket_mapping[bucket] = curr + bucket_count[bucket] // 2
        curr += bucket_count[bucket]
    
    out_image = np.zeros_like(img)
    for i in range(mImg):
        for j in range(nImg):
            bucket = bucketImg[i, j]
            mapped_index = min(int(bucket_mapping[bucket] * mStyle * nStyle / mImg / nImg), mStyle*nStyle-1)
            mapped_pos = list_style[mapped_index]
            out_image[i, j] = style[mapped_pos[0], mapped_pos[1]]
    cv2.imwrite(output_path, out_image)

if __name__ == '__main__':
    STYLE_CPKT_PATH = "./models/"
    SAMPLE_PATH = "./_sample_images/"
    GLOBAL_STYLE_TRANSFER_PATH = "./_global_style_transfer_images/"
    REGION_BASED_STYLE_TRANSFER_PATH = "./_region_based_style_transfer_images/"
    SEGMENT_MASK_PATH = "./_segment_mask_images/"
    GLOBAL_COLOR_TRANSFER_PATH = "./_color_transfer/_global_color_transfer_images/"
    STYLE_DATA_PATH = "./_color_transfer/_style_data/"
    STYLE_IMAGE_PATH = "./_style_images/"

    sample_images = ["girl1", "girl2", "girl3", "man1", "man2", "VOC2010_18"]
    image_suffix_list = ["jpg", "jpg", "jpg", "jpg", "jpg", "jpg"]
    styles = ["la_muse", "rain_princess", "scream", "wreck", "udnie", "wave"]

    for image_name, image_suffix in zip(sample_images, image_suffix_list):
        for style in styles:
            graph = load_frozenmodel()
            bin_mask = segmentation_image(graph, LABEL_NAMES, image_path="%s/%s.%s"%(SAMPLE_PATH, image_name, image_suffix), segmentation_save_path="%s/%s.jpg"%(SEGMENT_MASK_PATH, image_name))
            
            style_transfer("%s%s.ckpt"%(STYLE_CPKT_PATH, style), "%s/%s.%s"%(SAMPLE_PATH, image_name, image_suffix), "%s/%s_%s.%s"%(GLOBAL_STYLE_TRANSFER_PATH, image_name, style, image_suffix))

            color_transfer("%s/%s.%s"%(SAMPLE_PATH, image_name, image_suffix), "%s/%s.jpg"%(STYLE_IMAGE_PATH, style), "%s/%s.txt"%(STYLE_DATA_PATH, style), "%s/%s_%s.%s"%(GLOBAL_COLOR_TRANSFER_PATH, image_name, style, image_suffix))
            blend_img = blend_images(fg_path="%s/%s_%s.%s"%(GLOBAL_COLOR_TRANSFER_PATH, image_name, style, image_suffix), bg_path="%s/%s_%s.%s"%(GLOBAL_STYLE_TRANSFER_PATH, image_name, style, image_suffix), mask=bin_mask)
            
            cv2.imwrite("%s/blend_%s_%s.%s"%(REGION_BASED_STYLE_TRANSFER_PATH, image_name, style, image_suffix), blend_img)