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

def gaussian_kernel(k_size, sigma):
    """
    param
    k_size : Gaussian kernel size
    sigma : gaussian kernel standard variance
    
    return
    filter = k_size * k_size gaussian filter
    """
    size = k_size//2
    y, x = np.ogrid[-size:size+1, -size:size+1]
    #ref : https://en.wikipedia.org/wiki/Gaussian_filter
    filter = 1/(2*np.pi * (sigma**2)) * np.exp(-1 *(x**2 + y**2) /(2*(sigma**2)))
    sum = filter.sum()
    filter /= sum
    return filter

def padding(img, k_size):
    """
    param
    img : padding img
    k_size : kernel size
    
    return 
    res : padded img
    """
    pad_size = k_size//2
    if img.ndim == 2:
        rows, cols = img.shape
        res = np.zeros((rows + (2*pad_size), cols+(2*pad_size)), dtype=np.float)
        
    else :
        rows, cols = img.shape
        res = np.zeros((rows + (2*pad_size), cols+(2*pad_size), ch), dtype=np.float)

    if pad_size == 0:
        res = img.copy()
    else:
        res[pad_size:-pad_size, pad_size:-pad_size] = img.copy()
    return res
    
def gaussian_filtering(img, k_size=3,sigma=1):
    """
    param
    img : input img
    k_size : kernel size
    sigma : standard deviation
    
    return
    filtered_img : gaussian filtered image returned
    """
    if img.ndim == 3:
        rows, cols, channels = img.shape
        filter = gaussian_kernel(k_size, sigma)
        pad_img = padding(img,k_size)
        filtered_img = np.zeros((rows, cols, channels), dtype=np.float32)

        for ch in range(0, channels):
            for i in range(rows):
                for j in range(cols):
                    filtered_img[i, j, ch] = np.sum(filter * pad_img[i:i+k_size, j:j+k_size, ch])
    else:
        rows, cols = img.shape
        filter = gaussian_kernel(k_size, sigma)
        pad_img = padding(img,k_size)
        filtered_img = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                filtered_img[i, j] = np.sum(filter * pad_img[i:i+k_size, j:j+k_size])
    return filtered_img.astype(np.uint8)














def erosion(boundary=None, kernel=None):
    """
    erosion operation
    - a pixel element is 0 at least under kernel is 0
    -> return 0
    """
    boundary = boundary * kernel
    if (np.min(boundary) == 0):
        return 0
    else:
        return 255

def dilation(boundary=None, kernel=None):
    """
    erosion operation
    - a pixel element is not 0 at least under kernel is not 0
    -> return 255
    """
    boundary = boundary * kernel
    if (np.max(boundary) != 0):
        return 255
    else:
        return 0

def openning(img=None, k_size=None):
    """
    openning operation
    - erosion followed by dilation
    - it removes noise
    """
    erosion_img = morphology(img=img, method=1, k_size=k_size)
    opened_img = morphology(img=erosion_img, method=2, k_size=k_size)
    return opened_img

def closing(img=None, k_size=None):
    """
    closing operation
    - dilation follwed by erosion
    - it can close small holes inside the objects. 
    """
    dilation_img = morphology(img=img, method=2, k_size=k_size)
    closed_img = morphology(img=dilation_img, method=1, k_size=k_size)
    return closed_img


def morphology(img=None, method=None, k_size=None):
    """
    input
    img : input image
    method : 1(erosion), 2(dilation), 3(openning), 4(closing)
    k_size : kernel size
    
    output
    res_img : morphology operation image
    """
    rows, cols = img.shape
    pad_img = padding(img, k_size)
    kernel = np.ones((k_size,k_size))
    res_img = img.copy()
    
    if method == 1 or method == 2:
        for i in range(0, rows):
            for j in range(0, cols):
                if method == 1: #erosion operation
                    res_img[i, j] = erosion(pad_img[i:i+k_size, j:j+k_size], kernel)
                elif method == 2: #
                    res_img[i, j] = dilation(pad_img[i:i+k_size, j:j+k_size], kernel)
    if method == 3:
        res_img = openning(img, k_size=k_size)
    elif method == 4:
        res_img = closing(img, k_size=k_size)

    return res_img















"""
ref
- https://en.wikipedia.org/wiki/Edge_detection
- http://www.cs.cmu.edu/~16385/s17/Slides/4.0_Image_Gradients_and_Gradient_Filtering.pdf
- https://iskim3068.tistory.com/49
"""
def sobel_kerenl():
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    return kernel_x, kernel_y

def sobel(img, method=None):
    """
    get image gradient using sobel operater
    
    parameteres
    ------------
    img : input image applying sobel filter
    method : 1(x direction), 2(y dicrection), 3(x + y direction)
    """
    k_size = 3
    rows, cols = img.shape
    kernel_x, kernel_y = sobel_kerenl()
    pad_img = padding(img, k_size=k_size)
    res_img = np.zeros((rows,cols))
    sx, sy = 0, 0

    for i in range(0, rows):
        for j in range(0, cols):
            boundary = pad_img[i:i+k_size, j:j+k_size]
            if method == 1:
                sx = np.sum(kernel_x * boundary)
            elif method == 2:
                sy = np.sum(kernel_y * boundary)
            else:
                sx = np.sum(kernel_x * boundary)
                sy = np.sum(kernel_y * boundary)
            res_img[i,j] = np.sqrt(sx**2 + sy**2)
    
    return res_img




def laplacian_filter():
    kernel_x = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    kernel_y = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    return kernel_x, kernel_y


def laplacian(img):
    """
    get image gradient using laplacian filter
    
    parameteres
    ------------
    img : input image applying laplacian filter
    """
    k_size = 3
    rows, cols = img.shape
    kernel_x, kernel_y = laplacian_filter()
    pad_img = padding(img, k_size=k_size)
    res_img = np.zeros((rows,cols))
    sx, sy = 0, 0

    for i in range(0, rows):
        for j in range(0, cols):
            boundary = pad_img[i:i+k_size, j:j+k_size]
            sx = np.sum(kernel_x * boundary)
            sy = np.sum(kernel_y * boundary)
            res_img[i,j] = np.sqrt(sx**2 + sy**2)    
    return res_img

















"""
Canny Edge Detector
- 2020. 12.1 01:50

"""


"""
ref
- https://en.wikipedia.org/wiki/Canny_edge_detector
"""
def sobel_kerenl():
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    return kernel_x, kernel_y

def get_rounded_gradient_angle(ang):
    """
    get_rounded_gradient_angle
    - gradient direction angle round to one of four angles
    
    return : one of four angles
            representing vertical, horizontal and two diagonal direction
    
    parameteres
    ---------------
    ang : gradient direction angle
    """
    vals = [0, np.pi*1/4, np.pi*2/4, np.pi*3/4, np.pi*4/4]
    interval = [np.pi*1/8, np.pi*3/8, np.pi * 5/8, np.pi* 7/8]

    if ang < interval[0] and ang >= interval[-1]:
        ang = vals[0]
    elif ang < interval[1] and ang >= interval[0]:
        ang = vals[1]
    elif ang < interval[2] and ang >= interval[1]:
        ang = vals[2]
    elif ang < interval[3] and ang >= interval[2]:
        ang = vals[3]
    else:
        ang = vals[4]
    return ang
        

def get_gradient_intensity(img):
    """
    get gradient_intensity
    - calculate gradient direction and magnitude
    
    return (rows, cols, 2) shape of image about grad direction and mag
    
    parameteres
    ------------
    img : blured image
    """
    k_size = 3
    rows, cols = img.shape
    kernel_x, kernel_y = sobel_kerenl()
    pad_img = padding(img, k_size=k_size)
    res = np.zeros((rows, cols, 2))
    sx, sy = 0, 0

    for i in range(0, rows):
        for j in range(0, cols):
            boundary = pad_img[i:i+k_size, j:j+k_size]
            sx = np.sum(kernel_x * boundary)
            sy = np.sum(kernel_y * boundary)            
            ang = np.arctan2(sy, sx)
            mag = abs(sx) + abs(sy)
            if ang < 0:
                ang = ang + np.pi
            ang = get_rounded_gradient_angle(ang)
            res[i, j, 0] = ang
            res[i, j, 1] = mag
    return res



def check_local_maximum(direction=None, boundary=None):
    """
    check_local_maximum
    - check if center value is local maximum depend on gradient direction
    return True if center valus is local maximum
    
    parameter
    ------------
    direction : gradient direction
    boundary : region of image for finding local maximum
    """
    if direction == 0: # 0 degree, east and west direction
        kernel = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ])

    elif direction == 1: #45 degree, north east and south west direction
        kernel = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
    elif direction == 2: #90 degree, north & south direction
        kernel = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ])
    else : #135 degree, north west & south east direction
        kernel = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    
    max_val = np.max(kernel * boundary)
    if boundary[1,1] == max_val: #local maximum
        return True
    else: # not local maximu
        return False

    
    
def non_maximum_suppression(grad_intensity=grad_intensity):
    """
    non_maximum_suppression
    - check a pixel wiht it's neighbors in the direction of gradient
    - if a pixel is not local maximum, it is suppressed(means to be zero)
    """
    directions = [0, np.pi*1/4, np.pi*2/4, np.pi*3/4, np.pi*4/4]
    
    k_size = 3
    grad_direction = grad_intensity[:, :, 0]
    grad_magnitude = grad_intensity[:, :, 1]
    
    rows, cols = grad_magnitude.shape
    pad_img = padding(grad_magnitude, k_size=k_size)
    res_img = np.zeros((rows,cols))
    sx, sy = 0, 0
    
    
    for i in range(0, rows):
        for j in range(0, cols):
            direction = directions.index(grad_direction[i,j])
            boundary = pad_img[i:i+k_size, j:j+k_size]
            if check_local_maximum(direction, boundary) == True:
                res_img[i, j] = grad_magnitude[i, j]
            else:
                res_img[i, j] = 0
    return res_img



def hysteresis_threshold(suppressed_img=None, min_th=None, max_th=None):
    """
    hysterisis_threshold
    - check edge is connected to strong edge using eight connectivity
    
    parameter
    ---------------------
    - suppressed img : non maximum suppressed image
    - min_th : minimal threshold value
    - max_th : maximum threshold value
    """
    k_size = 3
    rows, cols = supressed_img.shape
    pad_img = padding(suppressed_img, k_size=k_size)
    res_img = np.zeros((rows,cols))

    eight_connectivity = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])    
    
    for i in range(0, rows):
        for j in range(0, cols):
            if pad_img[i+1, j+1] < min_th:
                res_img[i, j] = 0
            else:
                boundary = pad_img[i:i+k_size, j:j+k_size]
                max_magnitude = np.max(boundary * eight_connectivity)
                if max_magnitude >= max_th : # pixel is connected to real edge
                    res_img[i, j] = 255
                else:
                    res_img[i, j] = 0
    return res_img


def canny_edge_detector(img=None, min_th=None, max_th=None):
    """
    canny_edge_detector
    - detect canny edge from original image
    
    parameter
    - img : original image
    - min_th : minimal threshold value
    - max_th : maximum threshold value
    """
    img_blur = gaussian_filtering(img,k_size=5, sigma=1)
    grad_intensity = get_gradient_intensity(img_blur)
    suppressed_img = non_maximum_suppression(grad_intensity)
    canny_edge = hysteresis_threshold(suppressed_img=suppressed_img,
                                      min_th=min_th, max_th=max_th)
    return canny_edge


def canny_edge_visualize(img, min_th=100, max_th=200):
    """
    canny_edge_visualize
    - visualize all images from original to canny edge
    
    parameter
    - min_th : minimal threshold value
    - max_th : maximum threshold value
    """
    img_blur = gaussian_filtering(img,k_size=5, sigma=1)
    grad_intensity = get_gradient_intensity(img_blur)
    suppressed_img = non_maximum_suppression(grad_intensity)
    canny_edge = hysteresis_threshold(suppressed_img=suppressed_img,min_th=100, max_th=200)

    plt.figure(figsize=(64,32))
    plt.subplot(6, 1, 1)
    plt.title("original")
    plt.imshow(img)
    plt.subplot(6, 1, 2)
    plt.title("gaussian blur")
    plt.imshow(img_blur)
    plt.subplot(6, 1, 3)
    plt.title("gradient magnitude")
    plt.imshow(grad_intensity[:, :,0])
    plt.subplot(6, 1, 4)
    plt.title("gradient direction")
    plt.imshow(grad_intensity[:, :,1])
    plt.subplot(6, 1, 5)
    plt.title("non maximum suppression ")
    plt.imshow(suppressed_img)
    plt.subplot(6, 1, 6)
    plt.title("canny edge image")
    plt.imshow(canny_edge)

    
