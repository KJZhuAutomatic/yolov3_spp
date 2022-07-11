import torch
from typing import Dict, List, Optional
import os
import warnings
import numpy as np
from PIL import Image
import cv2
from .utils import *
from tqdm import tqdm
from .draw_box_utils import draw_objs
import matplotlib.pyplot as plt
import json
import pickle

def read_annotation(label_file_name:str) -> np.ndarray:
	with open(label_file_name) as f:
		annotation = np.array([l.strip().split() for l in f.readlines()], dtype=np.float32)
	if not annotation.size:
		annotation = np.zeros((0, 5), dtype=np.float32)
	return annotation

class YoloDataset(torch.utils.data.Dataset):
	"""Process yolo format data."""
	def __init__(self, path:str, img_size:int = 416,
	             batch_size:int = 16, 
	             augment=True, rect=False,
	             aug_param:Optional[Dict[str, float]] = None, rank:int=-1):
		super(YoloDataset, self).__init__()
		assert not (img_size % 32), 'Assigned image size must be multiple of 32.'
		with open(path) as f:
			available_img_files = f.readlines()
		cache_path = path.replace('txt', 'cache.pkl')
		try:
			with open(cache_path, 'rb') as f:
				cached_dict = pickle.load(f)
			use_cache = True
		except Exception as e:
			use_cache = False
		self.labels: List[np.ndarray] = []
		self.img_files: List[str] = []
		shapes: List[Tuple[int, int]] = []  # (width, height)
		if rank in [-1, 0]:
			pbar = tqdm(enumerate(available_img_files), desc='Processing images ...')
		else:
			pbar = enumerate(available_img_files)
		for idx, fn in pbar:
			fn = fn.strip()
			if use_cache: 
				if fn in cached_dict:
					annotation, shape = cached_dict[fn]
					self.labels.append(annotation)
					self.img_files.append(fn)
					shapes.append(shape)
					continue
			lfn = fn.replace('images', 'labels').replace('.jpg', '.txt')
			if os.path.exists(lfn):
				annotation = read_annotation(lfn)
				if annotation.shape[0]:
					assert annotation.shape[1] == 5, \
					             'label file %s column not equal to 5.'%lfn
					assert (annotation >= 0).all(), \
					             'label file %s has negative value'%lfn
					assert (annotation[:, 1:] <= 1).all(), \
					             'label file %s has relative coords larger than 1.'%lfn
					self.labels.append(annotation)
					self.img_files.append(fn)
					shapes.append(Image.open(fn).size)  # .size() -> (width, height)
					if use_cache: 
						# fn not in cached_dict, add it
						cached_dict[fn] = (annotation, shapes[-1])
				else:
					warnings.warn('label file %s is empty.'%lfn)
			else:
				warnings.warn('Lack label file of %s.'%fn)
		if not use_cache: # without file *.cache.pkl create it
			cached_dict = {self.img_files[i]:(self.labels[i], shapes[i]) \
			                                for i in range(len(self.img_files))}
			with open(cache_path, 'wb') as f:
				pickle.dump(cached_dict, f)
			print('Without cached dataset, save it.')

		self.mosaic = augment and not rect
		self.rect = rect
		self.augment = augment
		self.aug_param = aug_param
		self.img_size = img_size
		self.batch_size = batch_size
		self.indexs = None # record index of images, sorted by aspect ratio, used in rect
		self.letterbox_shapes = [[1, 1]] * int(np.ceil(len(shapes)/batch_size)) if rect else None
		self.shapes = np.array(shapes, dtype=np.float32) # (width, height)
		if rect:
			shapes = self.shapes
			aspect_ratios = (shapes[:, 1] / shapes[:, 0]) # height / width
			self.indexs = aspect_ratios.argsort()
			for batch_idx, _ in enumerate(self.letterbox_shapes):
				start, end = batch_size*batch_idx, batch_size*(batch_idx+1)
				batch_aspect_ratios = aspect_ratios[self.indexs[start:end]]
				min_r = batch_aspect_ratios[0]
				max_r = batch_aspect_ratios[-1]
				if min_r > 1:
					self.letterbox_shapes[batch_idx] = [1, 1 / min_r] 
				elif max_r < 1:
					self.letterbox_shapes[batch_idx] = [max_r, 1]
			self.letterbox_shapes = np.array(self.letterbox_shapes, dtype=np.float32) * img_size
			self.letterbox_shapes = (np.ceil(self.letterbox_shapes / 32) * 32).astype(np.int32)

	def __len__(self):
		return len(self.img_files)

	def simple_mosaic_load(self, indices:List[int]):
		assert len(indices) == 4, 'mosaic load must be 4 images, get %d'%len(indices)
		size = self.img_size
		imgs = np.empty(shape=(2*size, 2*size, 3), dtype=np.uint8)
		offset_x, offset_y = 0, 0
		labels = []
		for i, idx in enumerate(indices):
			img = cv2.imread(self.img_files[idx])
			h0, w0 = img.shape[:2]
			img, ratio, (top_pad, left_pad) = letterbox_img(img, self.img_size)
			imgs[offset_y:offset_y+size, offset_x:offset_x+size, :] = img.copy()
			label = self.labels[idx].copy() # [cls, x, y, w, h]
			label[:, 1:] = xywh2xyxy(label[:, 1:])
			label[:, 1::2] *= (w0*ratio)
			label[:, 2::2] *= (h0*ratio)
			label[:, 1::2] += (left_pad+offset_x)
			label[:, 2::2] += (top_pad+offset_y)
			labels.append(label)
			if i == 0:
				offset_x += size
			elif i == 1:
				offset_y += size
			elif i == 2:
				offset_x -= size
		img = cv2.resize(imgs, (size, size))
		label = np.concatenate(labels, axis=0)
		label[:, 1:] /= 2.0
		return random_affine(img, label, **self.aug_param) 
		# return img, label

	def mosaic_load(self, indices:List[int]):
		size = self.img_size
		xc, yc = [int(np.random.uniform(0.5*size, 1.5*size))	for _ in range(2)]
		mosaic_img = np.full(shape=(2*size, 2*size, 3), fill_value=114, dtype=np.uint8)
		mosaic_labels = []
		for i,idx in enumerate(indices):
			img = cv2.imread(self.img_files[idx])  # note rect is False
			img, _ = resize_img(img, self.img_size)
			h, w = img.shape[:2]
			if i == 0:
				x1a, y1a = max(xc-w, 0), max(yc-h, 0)
				x2a, y2a = xc, yc
				x1b = w - (x2a - x1a)
				y1b = h - (y2a - y1a)
				x2b, y2b = w, h
			elif i == 1:
				x1a = xc
				y1a = max(yc-h, 0)
				x2a = min(xc+w, 2*size)
				y2a = yc
				x1b = 0
				y1b = h - (y2a - y1a)
				x2b = min(x2a - x1a, w)
				y2b = h
			elif i == 2:
				x1a = max(xc-w, 0)
				y1a = yc
				x2a = xc
				y2a = min(yc+h, 2*size)
				x1b = w - (x2a - x1a)
				y1b = 0
				x2b = w
				y2b = min(y2a - y1a, h)
			else:
				x1a, y1a = xc, yc
				x2a = min(xc+w, 2*size)
				y2a = min(yc+h, 2*size)
				x1b, y1b = 0, 0
				x2b, y2b = min(x2a - x1a, w), min(y2a - y1a, h)

			mosaic_img[y1a:y2a, x1a:x2a, :] = img[y1b:y2b, x1b:x2b, :]
			padw = x1a - x1b
			padh = y1a - y1b
			label = self.labels[idx].copy() # [cls, x, y, w, h]
			label[:, 1::2] *= w
			label[:, 2::2] *= h
			label[:, 1:] = xywh2xyxy(label[:, 1:])
			label[:, 1::2] += padw
			label[:, 2::2] += padh
			mosaic_labels.append(label)
		mosaic_label = np.concatenate(mosaic_labels, axis=0)
		np.clip(mosaic_label[:, 1:], 0, 2*size, out=mosaic_label[:, 1:])
		# mosaic_label[:, 1:] /= 2.0 # error mosaic_label / 2.0 change cls, same error as simple_mosaic_load
		return random_affine(mosaic_img, mosaic_label, border=-size//2, **self.aug_param)

	def __getitem__(self, idx):
		# the idx was changed if rect.
		# if return idx, not align with coco_index, and mAP is 0
		# this error waste a morning.
		original_idx = idx
		if self.mosaic:
			indices = [idx] + [np.random.randint(0, len(self)) for _ in range(3)]
			img, label = self.mosaic_load(indices)
			shapes = None
		else:
			if self.rect:
				batch_idx = idx // self.batch_size
				idx = self.indexs[idx]
			fn = self.img_files[idx]
			img = cv2.imread(fn)
			h0, w0 = img.shape[:2]
			img, ratio0 = resize_img(img, self.img_size)
			letterbox_shape = self.letterbox_shapes[batch_idx] if self.rect else self.img_size
			img, ratio1, (top_pad, left_pad) = letterbox_img(img, letterbox_shape)
			ratio = ratio0 * ratio1
			shapes = (h0, w0), ((ratio, ratio), (top_pad, left_pad))
			label = self.labels[idx].copy() # [cls, x, y, w, h]
			label[:, 1:] = xywh2xyxy(label[:, 1:])
			label[:, 1::2] *= (w0*ratio)
			label[:, 2::2] *= (h0*ratio)
			label[:, 1::2] += left_pad
			label[:, 2::2] += top_pad
		if self.augment:
			if not self.mosaic:
				img, label = random_affine(img, label, **self.aug_param)
			img = augment_hsv(img, **self.aug_param)
			# TODO Flip
		# breakpoint()
		# there was a every fool error in xyxy2xywh func
		# if I don't check image's label, training performance would be very pool
		label[:, 1:] = xyxy2xywh(label[:, 1:])
		label[:, 1::2] /= img.shape[1]
		label[:, 2::2] /= img.shape[0]
		# breakpoint()
		
		if self.augment:
			# note after label normalization apply random flip
			if np.random.rand() > 0.5:
				img = np.fliplr(img)
				label[:, 1:2] = 1 - label[:, 1:2]
		
		img = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))
		
		return img, label, self.img_files[idx], shapes, original_idx

	@staticmethod
	def collate_fn(batch):
		imgs, labels, paths, shapes, idxs = zip(*batch)
		imgs = torch.from_numpy(np.stack(imgs, axis=0))
		labels = list(labels)
		for i, label in enumerate(labels):
			labels[i] = np.ones(shape=(label.shape[0], 6), dtype=np.float32) * i
			labels[i][:, 1:] = label.copy()
		labels = torch.from_numpy(np.concatenate(labels, axis=0))
		return imgs, labels, paths, shapes, idxs

	def coco_index(self, idx):
		if self.rect:
			idx = self.indexs[idx]
		return torch.from_numpy(self.labels[idx].copy()), self.shapes[idx, ::-1]

if __name__ == '__main__':
	aug_param = {'scale': 0.0, 'shear': 0.0, 'translate': .0, 'degrees': 0.0, 'hsv_h':0.0138, 'hsv_s':0.678, 'hsv_v':0.36}
	dataset = YoloDataset('data/my_train_data.txt', augment=True,
	                      rect=False, aug_param=aug_param)
	json_path = "./data/pascal_voc_classes.json"  # json标签文件
	with open(json_path, 'r') as f:
		class_dict = json.load(f)
	category_index = {str(v): str(k) for k, v in class_dict.items()}
	for idx in range(0, 10):
		img, label, _, _, _ = dataset[idx]
		# 下面这两段测试，我本应该找label的错误，classes和scores也混在里面
		# 对目标不清晰，那里错，要干什么，怎么找？一定先清晰，再干
		# h, w = tuple(img.shape[1:])
		# shape = np.array([[w, h, w, h]])
		# img = Image.fromarray(img.transpose(1, 2, 0))
		# boxes = label[:, 1:] * shape
		# boxes = xywh2xyxy(boxes)
		'''
		img = Image.fromarray(img.transpose(1, 2, 0))
		boxes = label[:, 1:]
		classes = label[:, 0].astype(np.int32) + 1
		'''
		
		h, w = tuple(img.shape[1:])
		img = Image.fromarray(img.transpose(1, 2, 0))
		label[:, 1::2] *= w
		label[:, 2::2] *= h
		label[:, 1:] = xywh2xyxy(label[:, 1:])
		# breakpoint()
		classes = label[:, 0].astype(np.int32) + 1
		boxes = label[:, 1:]
		
		scores = np.ones_like(classes)
		plt.imshow(draw_objs(img, boxes, classes, scores, category_index=category_index))
		plt.show()
