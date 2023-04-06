import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
from .kit import norm, getPointsFromHeatmap
from skimage.metrics import peak_signal_noise_ratio as PSNR


def get_metric(s):
    return {
        'ssim': cal_ssim,
        'psnr': cal_psnr,
        'mse': cal_mse,
        'mre': cal_mre,
        'std': cal_std,
    }[s]


def prepare(x):
    if np.iscomplexobj(x):
        x = np.abs(x)
    return norm(x)


def cal_mse(x, y):
    ''' 
        result changes 
        if x,y are not normd to (0,1) or normd to different range
    '''
    x = prepare(x)
    y = prepare(y)
    return MSE(x, y)


def cal_ssim(x, y):
    ''' 
        result changes if x,y are not normd to (0,1)
        won't change   if normd to different range
    '''
    x = prepare(x)
    y = prepare(y)
    return SSIM(x, y, data_range=x.max() - x.min())


def cal_psnr(x, y):
    ''' 
        result rarely changes if x,y are not normd to (0,1)
        won't change   if normd to different range
    '''
    x = prepare(x)
    y = prepare(y)
    return PSNR(x, y, data_range=x.max() - x.min())


def cal_mre(x, y):
    ''' cal mean distance of the two heatmap's center
        x: numpy.ndarray heatmap  channel x imgshape
        y: numpy.ndarray heatmap  channel x imgshape
    '''
    # assert x.shape == y.shape
    # assert x.ndim >= 3
    p1 = getPointsFromHeatmap(x)
    p2 = getPointsFromHeatmap(y)

    li = [sum((i-j)**2 for i, j in zip(point, gt_point)) **
          0.5 for point, gt_point in zip(p1, p2)]
    return np.mean(li)


def cal_std(x, y):
    ''' cal std distance of the two heatmap's center
        x: numpy.ndarray heatmap  channel x imgshape
        y: numpy.ndarray heatmap  channel x imgshape
    '''
    # assert x.shape == y.shape
    # assert x.ndim >= 3
    p1 = getPointsFromHeatmap(x)
    p2 = getPointsFromHeatmap(y)

    li = [sum((i-j)**2 for i, j in zip(point, gt_point)) **
          0.5 for point, gt_point in zip(p1, p2)]
    return np.std(li)
