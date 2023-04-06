#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 21:59:31 2023

@author: kui
"""

import torch
from model.networks.model import get_model

if __name__ == "__main__":
    model = get_model('train', 12)
    a = torch.zeros((2, 3, 256, 256))#.cuda()    
    b = torch.zeros((2, 1, 256, 256))#.cuda()
    c = model({"img":a, "mask":b})
    print(c.shape)