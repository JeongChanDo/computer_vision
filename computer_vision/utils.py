#!/usr/bin/env python
# coding: utf-8
import numpy as np

def Histogram(img):
    row, col = img.shape
    hist = np.zeros(256)
    for i in range(0, row):
        for j in range(0, col):
            hist[img[i, j]] += 1
    
    return hist


def Threshold(img=None, th=125):

    if type(img) is not np.ndarray:
        raise AssertionError("img is not ndarray")
    
    row, col = img.shape
    res = np.zeros((row, col))
    for i in range(0,row):
        for j in range(0,col):
            if img[i, j] >= th:
                res[i, j] = 1
            else:
                res[i, j] = 0
    
    return res

def adaptiveThresholdMean(img,  block_size=5, C=4):

    if type(img) is not np.ndarray:
        raise AssertionError("img is not ndarray")
    row, col = img.shape


    res = np.zeros((row, col))
    if (block_size % 2 == 0):
        block_size += 1
    
    for i in range(0, row):
        for j in range(0, col):
            x_min = j-block_size//2
            x_max = j+block_size//2
            y_min = i-block_size//2
            y_max = i+block_size//2
            
            if x_min <= 0:
                x_min = 0
            
            if x_max >= col:
                x_max = col
            
            if y_min <= 0:
                y_min = 0
            
            if y_max >= row:
                y_max = row

            
            val = img[y_min:y_max, x_min:x_max].mean()
            local_th = val-C
            if img[i,j] >= local_th:
                res[i, j] = 255
            else:
                res[i, j] = 0
    return res

def otsuThreshold(img):
    
    if type(img) is not np.ndarray:
        raise AssertionError("img is not ndarray")
    
    hist = Histogram(img)
    
    vars_within = []
    vars_between = []

    zero = 1.e-17
    for t in range(0, 256):
        sumb = np.sum(hist[:t]) + zero
        sumw = np.sum(hist[t:]) + zero
        sum = sumb + sumw
        wb = sumb/sum
        ww = sumw/sum
        
        mub = zero
        muw = zero
        for i in range(0, t):
            mub += i * hist[i]/sumb
        for i in range(t, 256):
            muw += i * hist[i]/sumw

        vb = zero
        vw = zero
        for i in range(0, t):
            vb += hist[i] * ((i - mub)**2)/sumb
        for i in range(t, 256):
            vw += hist[i] * ((i - muw)**2)/sumw
    
        var_within = wb * vb + ww * vw
        vars_within.append(var_within)


    th = vars_within.index(min(vars_within))
    print(th)
    res = Threshold(img, th)
    return res