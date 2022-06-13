import os

import sys
import torch
import torch.nn as nn
#from model import *
#from dataset import MatchDataset
import torchvision
from torchvision import models, transforms, utils
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import cv2
import math
import copy

from dexined_edges import DexiNedEdges


def color_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


def draw_rectangle(arr, topleft, psz, col, fill):

    psz -= 1
    x = topleft[0]
    y = topleft[1]

    if fill:

        #arr[x:x+psz,y:y+psz,0] = 255 #2*arr[x:x+psz,y:y+psz,0]
        arr[x:x+psz,y:y+psz,0] = 255 - arr[x:x+psz,y:y+psz,0]
        arr[x:x+psz,y:y+psz,1] = 255 - arr[x:x+psz,y:y+psz,1]
        arr[x:x+psz,y:y+psz,2] = 255 - arr[x:x+psz,y:y+psz,2]

    else:

        th = 2

        arr[x:x+th,y:y+psz,0] = col[0]
        arr[x:x+th,y:y+psz,1] = col[1]
        arr[x:x+th,y:y+psz,2] = col[2]
        arr[x+psz:x+psz+th,y:y+psz+th,0] = col[0]
        arr[x+psz:x+psz+th,y:y+psz+th,1] = col[1]
        arr[x+psz:x+psz+th,y:y+psz+th,2] = col[2]

        arr[x:x+psz+th,y:y+th,0] = col[0]
        arr[x:x+psz+th,y:y+th,1] = col[1]
        arr[x:x+psz+th,y:y+th,2] = col[2]
        arr[x:x+psz,y+psz:y+psz+th,0] = col[0]
        arr[x:x+psz,y+psz:y+psz+th,1] = col[1]
        arr[x:x+psz,y+psz:y+psz+th,2] = col[2]

    return arr


def get_max_loc(img, pat, w1, h1, w2, h2, psz2, fsz1, fsz2):
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    cs = cos(img, pat).cpu().detach().numpy()
    cs = np.squeeze(cs)
    
    cs = np.pad(cs, ((w1, w2), (h1, h2)), constant_values=((-1, -1),))
    csa3 = cv2.resize(cs, dsize=(fsz1, fsz2), interpolation=cv2.INTER_LINEAR)

    mxv = np.max(csa3)
    meanv = np.mean(csa3)
    amx = np.argmax(csa3)
    
    #close = np.where(csa3 > 0.95 * mxv)
    close = np.where(csa3 > 0.999 * mxv)
    cl0 = close[0] - psz2 + 1
    cl1 = close[1] - psz2 + 1
    amx2 = np.unravel_index(amx, csa3.shape)
    mx = amx2[0] - psz2 + 1
    my = amx2[1] - psz2 + 1

    csa3 = 255 * (csa3+1) / 2
    csa3 = csa3.astype(np.uint8)

    return mx, my, mxv, csa3, [cl0, cl1]



class MatchNet(nn.Module):
	def __init__(self, nc=32, numl=4, ksz=64):
		super().__init__()

		#nc = 32 #128 #32

		self.layers = nn.ModuleList([])

		for i in range(numl):

			if i == 0:
				self.layers.append(nn.ModuleList([
					nn.Conv2d(3, nc, kernel_size=7, padding=3),
					nn.ReLU(True)
				]))
			else:
				self.layers.append(nn.ModuleList([
					nn.Conv2d(nc, nc, kernel_size=7, padding=3),
					nn.ReLU(True)
				]))

		self.final_conv = nn.Conv2d(nc, nc, kernel_size=ksz, stride=4, padding=0)


	def forward(self, im, do_lsig=False):

		x = im

		for cnv, rlu in self.layers:
			#x = attn(x) + x
			x = cnv(x)
			x = rlu(x)
		
		x = self.final_conv(x)

		return x



class StructureSimilarity():

    def __init__(self, pth):

        self.pth = pth
        self.patch_size = 64
        self.edgify_temp = True
        self.edgify_tgt = True
        #edgify_temp = int(sys.argv[2])
        #edgify_tgt = int(sys.argv[3])
        #weight = int(sys.argv[4])
        #do_scale = 0 #int(sys.argv[5])

        # patch_size = int(sys.argv[1])
        # edgify = int(sys.argv[2])
        # pth_loc = sys.argv[3]
        
        #self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')

        model = MatchNet(128, 4, self.patch_size).to(self.device)
        model.load_state_dict(torch.load(self.pth, map_location=self.device))
        self.model = model.eval()

        self.dexi = DexiNedEdges()

        self.color_trans = color_transform()
        self.scale = 1

        #app = PhotoCtrl(model, dexi, device, patch_size, edgify_temp, edgify_tgt, weight, do_scale, split_model)
        #app.MainLoop()

    
    def locate_ref_in_tgt(self, tgt_im, sketch_im, scale):
        
        #self.scale = scale
        self.scale = float(self.patch_size) / scale
        #print(self.scale)
        tgt_im_rgb = tgt_im

        if self.edgify_temp:
            sketch_im = self.dexi.get_edges(sketch_im, self.patch_size)
            sketch_im = cv2.cvtColor(sketch_im,cv2.COLOR_GRAY2RGB)

        if self.edgify_tgt:
            tgt_im = self.dexi.get_edges(tgt_im, None)
            tgt_im = cv2.cvtColor(tgt_im,cv2.COLOR_GRAY2RGB)
            #tgt_im_rgb = cv2.imread(self.tgt_img,cv2.IMREAD_COLOR)


        sketch_im = sketch_im[0:self.patch_size, 0:self.patch_size, :]

        width = int(tgt_im.shape[1] * self.scale)
        height = int(tgt_im.shape[0] * self.scale)
        tgt_im2 = cv2.resize(tgt_im, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        #tgt_im_rgb2 = cv2.resize(tgt_im_rgb, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

        #print(sketch_im.shape)
        #print(tgt_im2.shape)

        ref = self.color_trans(sketch_im)
        tgt = self.color_trans(tgt_im2)

        ref1 = self.model(ref.to(self.device).unsqueeze(0))
        tgt1 = self.model(tgt.to(self.device).unsqueeze(0))

        #cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        psz = ref.size(2)
        psz2 = int(psz/2)

        wi = tgt.size(1)
        hi = tgt.size(2)
        wi2 = wi-psz+1
        hi2 = hi-psz+1
        w = wi-wi2
        w1 = int(w/2)
        w2 = w-w1
        h = hi-hi2
        h1 = int(h/2)
        h2 = h-h1

        tgt12 = F.interpolate(tgt1, size=[wi2,hi2], mode='bilinear')

        fshp = tgt_im_rgb.shape
        scl = fshp[0] / tgt_im.shape[0]
        npsz = int(scl*psz)
        npsz2 = int(npsz/2)

        mx11, my11, response, heatmap, close = get_max_loc(tgt12, ref1, w1, h1, w2, h2, npsz2, fshp[1], fshp[0])

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        im = cv2.cvtColor(tgt_im_rgb, cv2.COLOR_BGR2RGB)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        im = draw_rectangle(im, [mx11, my11], npsz, [255,0,0], 1)
        #im = draw_rectangles(im, close, npsz, [255,0,0], 1)

        return im, heatmap