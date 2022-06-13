import sys
import os
import time, platform
import numpy as np
import cv2

import torch

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = BASE_DIR
#sys.path.append(os.path.join(ROOT_DIR, '../DexiNed/models'))
#sys.path.append(os.path.join(ROOT_DIR, '../DexiNed/models'))

from dexined_model import DexiNed
#from model import DexiNed
#from utils import image_normalization



def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
	"""This is a typical image normalization function
	where the minimum and maximum of the image is needed
	source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

	:param img: an image could be gray scale or color
	:param img_min:  for default is 0
	:param img_max: for default is 255

	:return: a normalized image, if max is 255 the dtype is uint8
	"""

	img = np.float32(img)
	# whenever an inconsistent image
	img = (img - np.min(img)) * (img_max - img_min) / \
		((np.max(img) - np.min(img)) + epsilon) + img_min
	return img



class DexiNedEdges():

	def __init__(self):

		self.mean_bgr = [103.939, 116.779, 123.68]

		checkpoint_path = '10_model.pth'

		# Get computing device
		self.device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

		# Instantiate model and move it to the computing device
		model = DexiNed().to(self.device)
		model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
		model.eval()

		self.model = model


	def resize_only(self, img, sz):

		img = cv2.resize(img, (sz[0], sz[1]), interpolation = cv2.INTER_LINEAR)
		#img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]

		return img


	def resize_pad_resize(self, img, sz1, pad_to_sz, sz2):

		img = cv2.resize(img, (sz1[0], sz1[1]), interpolation = cv2.INTER_LINEAR)
		img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]

		diffw = pad_to_sz[0] - img.shape[1]
		diffh = pad_to_sz[1] - img.shape[0]
		img = cv2.copyMakeBorder(img,0,diffh,0,diffw,cv2.BORDER_CONSTANT,value=[255,255,255])

		img = cv2.resize(img, (sz2[0], sz2[1]), interpolation = cv2.INTER_LINEAR)
		img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]

		return img


	def ttransform(self, img):

		img = np.array(img, dtype=np.float32)
		# if self.rgb:
		#     img = img[:, :, ::-1]  # RGB->BGR
		img -= self.mean_bgr
		img = img.transpose((2, 0, 1))
		img = torch.from_numpy(img.copy()).float()

		return img


	def get_edge_image(self, tensor, img_shape):

		#output_dir_f = os.path.join(output_dir, fuse_name)
		#output_dir_a = os.path.join(output_dir, av_name)
		#os.makedirs(output_dir_f, exist_ok=True)
		#os.makedirs(output_dir_a, exist_ok=True)

		# 255.0 * (1.0 - em_a)
		edge_maps = []
		for i in tensor:
			tmp = torch.sigmoid(i).cpu().detach().numpy()
			edge_maps.append(tmp)
		tensor = np.array(edge_maps)
		# print(f"tensor shape: {tensor.shape}")

		image_shape = [x.cpu().detach().numpy() for x in img_shape]

		# (H, W) -> (W, H)
		image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]
		i_shape = image_shape[0]
		#assert len(image_shape) == len(file_names)


		idx = 0

		tmp = tensor[:, idx, ...]
		tmp = np.squeeze(tmp)

		# Iterate our all 7 NN outputs for a particular image
		preds = []
		for i in range(tmp.shape[0]):
			tmp_img = tmp[i]
			tmp_img = np.uint8(image_normalization(tmp_img))
			tmp_img = cv2.bitwise_not(tmp_img)
			# tmp_img[tmp_img < 0.0] = 0.0
			# tmp_img = 255.0 * (1.0 - tmp_img)

			#noa
			# Resize prediction to match input image size
			if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
				tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
				#tmp_img = cv2.resize(tmp_img, (500, 333))
			

			preds.append(tmp_img)

			if i == 6:
				fuse = tmp_img
				fuse = fuse.astype(np.uint8)

		return fuse


	def save_image(self, tensor, img_shape, savename):

		fuse = self.get_edge_image(tensor, img_shape)
		cv2.imwrite(savename, fuse)

		return fuse


	def get_edges(self, img, sz, savename=None):

		if type(img) == str:
			image = cv2.imread(img, cv2.IMREAD_COLOR)
			image = cv2.bilateralFilter(image,15,80,80)
		else:
			image = img

		

		with torch.no_grad():
			if sz is None:
				img_shape = [torch.tensor([image.shape[0]]), torch.tensor([image.shape[1]])]
				image = self.resize_only(image, [512,512])
			else:
				img_shape = [torch.tensor([333]), torch.tensor([500])]
				#img_shape = [torch.tensor([sz]),torch.tensor([sz])]
				image = self.resize_pad_resize(image, [sz,sz], [500,333], [512,512])
			imgt = self.ttransform(image)
			#imgt, padw, padh = transform(image, [sz,sz])
			preds = self.model(imgt.unsqueeze(0).to(self.device))
			#print(preds)
			#save_image(preds, img_shape, savename)
			
			if savename is not None:
				edges = self.save_image(preds, img_shape, savename)
			else:
				edges = self.get_edge_image(preds, img_shape)

		return edges