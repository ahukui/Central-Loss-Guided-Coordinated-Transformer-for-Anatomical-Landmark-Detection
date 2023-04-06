from .loss_and_optim import *
from .gln import GLN
from .gln2 import GLN2
from .u2net import U2Net
from .tri_unet import Tri_UNet
from .unet2d import UNet as unet2d
from .globalNet import GlobalNet
from .StructNet import CoorTransformer
from .Baseline import Baseline
# from .TransformerNet import Transformernet
# from .SwinTransformer import SiwnTransformernet
# from .DeTransformerNet import DeTransformernet
def get_net(s):
    return {
        'unet2d': unet2d,
        'u2net': U2Net,
        'gln': GLN,
        'gln2': GLN2,
        'tri_unet': Tri_UNet,
        'globalnet': GlobalNet,
        'pointdat':CoorTransformer,
        'baseline':Baseline,
        # 'transformer':Transformernet,
        # 'siwntransformernet':SiwnTransformernet,
        # 'detransformernet':DeTransformernet
    }[s.lower()]


def get_loss(s):
    return {
        'l1': l1,
        'l2': l2,
        'bce': bce,
        'cl':cl,
        'fcl':fcl,
        'wbce':wbce
    }[s.lower()]


def get_optim(s):
    return {
        'adam': adam,
        'sgd': sgd,
        'adagrad': adagrad,
        'rmsprop': rmsprop,

    }[s.lower()]


def get_scheduler(s):
    return {
        'steplr': steplr,
        'multisteplr': multisteplr,
        'cosineannealinglr': cosineannealinglr,
        'reducelronplateau': reducelronplateau,
        'lambdalr': lambdalr,
        'cycliclr': cycliclr,

    }[s.lower()]
