import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
import torchvision.ops as box_ops
import numpy as np
import cv2

def xywh2xyxy(boxes:Union[Tensor, np.ndarray]):
	# boxes [n, 4] with [x, y, w, h] -> [x1, y1, x2, y2]
	assert boxes.shape[1] == 4, 'boxes must in shape [n, 4]'
	xc, yc, w, h = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3]/2.0, boxes[:, 3:4]/2.0
	x1, y1, x2, y2 = xc - w, yc - h, xc + w, yc + h
	if isinstance(boxes, Tensor):
		return torch.cat((x1, y1, x2, y2), axis=1)
	else:
		return np.concatenate((x1, y1, x2, y2), axis=1)

def xyxy2xywh(boxes:Union[Tensor, np.ndarray]):
	x1, y1, x2, y2 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
	xc, yc, w, h = (x1 + x2)/2.0, (y1 + y2)/2.0, x2 - x1, y2 - y1
	'''
	this is an error waste a afternoon. So fool.
	be definited to target, discard no need variable, 
	if isinstance(boxes, Tensor):
		return torch.cat((x1, y1, x2, y2), axis=1)
	else:
		return np.concatenate((x1, y1, x2, y2), axis=1)
	'''
	if isinstance(boxes, Tensor):
		return torch.cat((xc, yc, w, h), axis=1)
	else:
		return np.concatenate((xc, yc, w, h), axis=1)

def nms(pred:Tensor, conf_thres:float, iou_thres:float, multi_label:bool = True):
	# perform nms for inference pred [bs, num_anchors, num_classes+5], 
	# last dimension of pred [x, y, w, h, conf, cls0_score, ...cls19_score]
	outs:List[Tensor] = [] # Tensor shape [n, 6], [x1, y1, x2, y2, score, cls_idx]
	dtype, device = pred.dtype, pred.device
	None_det = torch.zeros(0, 6, dtype=dtype, device=device)
	num_classes = pred.shape[-1]-5
	num_anchors = pred.shape[1]
	bs = pred.shape[0]
	cls_offset = 4096 # suppose img size small than cls_offset
	img_offset = cls_offset * num_classes
	img_idxs = torch.arange(bs, dtype=dtype, device=device).view(bs, 1).repeat(1, num_anchors)
	pred = torch.cat((pred.view(-1, num_classes+5), img_idxs.view(-1, 1)), axis=1)

	# last dimension of pred [x, y, w, h, conf, cls0_score, ...cls19_score, img_idx]
	# I ignored the last img_idx and use pred[:, 5:], waste a evening.
	# Through same logits can work well in pred.shape[0] == 0 case, maybe some bug
	pred = pred[pred[:, 4] > conf_thres] # discard low confidence pred
	pred = pred[((pred[:, 2:4] > 2) & (pred[:, 2:4] < cls_offset)).all(axis=1)]
	
	if not pred.shape[0]:
		return [None_det] * bs
	
	pred[:, 5:-1] *= pred[:, 4:5]
	pred[:, :4] = xywh2xyxy(pred[:, :4])
	
	if multi_label:
		i, j = (pred[:, 5:-1] > conf_thres).nonzero(as_tuple=True)
		boxes = pred[i, :4]
		cls_idxs = j
		scores = pred[i, j+5]
		img_idxs = pred[i, -1]
	else:
		scores, cls_idxs = pred[:, 5:-1].max(1)
		mask = (scores > conf_thres)
		boxes = pred[mask, :4]
		cls_idxs = cls_idxs[mask]
		scores = scores[mask]
		img_idxs = pred[mask, -1]
	
	if not boxes.shape[0]:
		return [None_det] * bs
	
	boxes_offset = boxes + cls_idxs[:, None] * cls_offset + img_idxs[:, None] * img_offset
	keeps = box_ops.nms(boxes_offset, scores, iou_thres)
	nmsd_pred = torch.cat((boxes, scores[:, None], cls_idxs[:, None]), axis=1)
	nmsd_pred = nmsd_pred[keeps]
	img_idxs = img_idxs[keeps]
	
	if not nmsd_pred.shape[0]:
		return [None_det] * bs
	
	for i in range(bs):
		mask = (img_idxs == i)
		outs.append(nmsd_pred[mask][:100])
		# this is an error index wasting a morning
		# sorted index is not for original list, pay attention to index sort
		# outs.append(nmsd_pred[nmsd_pred[mask, 4].argsort(descending=True)][:100])

	return outs

def resize_img(img:np.ndarray, size:int):
	'''
	resize image as large size equal to input size while keep aspect ratio
	'''
	shape = img.shape[:2] # image shape [img_h, img_w, 3]
	ratio = size / max(shape)
	img = cv2.resize(img, (int(shape[1]*ratio), int(shape[0]*ratio)))
	return img, ratio

def letterbox_img(img:np.ndarray, size:Union[int, Tuple[int, int]], auto=False, 
	              pad_color:Tuple[int, int, int]=(114, 114, 114)):
	'''
	put image in a required size and record pad value 
	img: array shape [height, width, 3]
	size: expected size of image int or tuple
	auto: bool flag of whether adjust the value of pad
	pad_color: pading border color
	'''
	if isinstance(size, int):
		size = (size, size)
	ratio = min(size[0]/img.shape[0], size[1]/img.shape[1])
	if ratio != 1:
		img = cv2.resize(img, (int(img.shape[1]*ratio), int(img.shape[0]*ratio)))
	dh, dw = size[0] - img.shape[0], size[1] - img.shape[1]
	top = dh // 2
	bottom = dh - top
	left = dw // 2
	right = dw - left
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
	return img, ratio, (top, left)

def remap_coords(img_shape, coords:Tensor, orig_img_shape, ratio_pad=None):
	# shape (height, width), coords [N, 4] [x1, y1, x2, y2]
	if ratio_pad: # (int, (int, int))
		ratio = ratio_pad[0]
		pad = ratio_pad[1]
	else:
		ratio = max(img_shape) / max(orig_img_shape)
		pad = (img_shape[0] - orig_img_shape[0]*ratio) / 2, (img_shape[1] - orig_img_shape[1]*ratio) / 2
	remaped_coords = coords.clone()
	remaped_coords[:, [0, 2]] -= pad[1]
	remaped_coords[:, [1, 3]] -= pad[0]
	remaped_coords /= ratio
	return clamp_coords(remaped_coords, orig_img_shape)

def clamp_coords(coords:Tensor, orig_img_shape):
	# coords[:, [0, 2]] at first, but cannot clamp original coords tensor
	coords[:, 0::2].clamp_(min=0, max=orig_img_shape[1])
	coords[:, 1::2].clamp_(min=0, max=orig_img_shape[0]) 
	return coords

def random_affine(img:np.ndarray, label:np.ndarray, border=0, **hyp):
	h, w = img.shape[:2]
	h += (2 * border)
	w += (2 * border)
	angle = np.random.uniform(-hyp['degrees'], hyp['degrees'])
	scale = np.random.uniform(1-hyp['scale'], 1+hyp['scale'])
	translation = np.random.uniform(-hyp['translate'], hyp['translate'], size=(2,1))
	shear = np.tan(np.radians(np.random.uniform(-hyp['shear'], hyp['shear'], size=(2,))))
	M, T, S = np.eye(3), np.eye(3), np.eye(3)
	M[:2, :] = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), 
		                               angle=angle, scale=scale)
	T[:2, 2:3] = (translation * np.array([[img.shape[1]], [img.shape[0]]])) \
	                                   + np.array([[border], [border]])
	S[0, 1], S[1, 0] = shear
	transform = S @ T @ M
	transformed_img = cv2.warpAffine(src=img, M=transform[:2], 
		                             dsize=(w, h), borderValue=(114, 114, 114))
	nobj = label.shape[0]
	boxes = label[:, 1:] # x1y1x2y2
	area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
	coords = np.ones((4*nobj, 3), dtype=np.float32)
	coords[:, :2] = boxes[:, [0, 1, 2, 3, 2, 1, 0, 3]].reshape(-1, 2) # x1y1 x2y2 x2y1 x1y2
	transformed_coords = (transform @ coords.T).T  # [4*nobj, 3]
	transformed_coords = transformed_coords[:, :2].reshape(nobj, -1)
	xmin = transformed_coords[:, [0, 2, 4, 6]].min(1, keepdims=True).clip(0, w)
	xmax = transformed_coords[:, [0, 2, 4, 6]].max(1, keepdims=True).clip(0, w)
	ymin = transformed_coords[:, [1, 3, 5, 7]].min(1, keepdims=True).clip(0, h)
	ymax = transformed_coords[:, [1, 3, 5, 7]].max(1, keepdims=True).clip(0, h)
	w = (xmax - xmin).squeeze(axis=-1)
	h = (ymax - ymin).squeeze(axis=-1)
	area = w * h
	small_boxes_filter = (w > 4) & (h > 4)
	area_factor_filter = ((area / (area0*scale+1e-6)) > 0.2)
	aspect_ratio_filter = ((w / (h+1e-6)) < 10.0) & ((h / (w+1e-6)) < 10.0)
	inds = small_boxes_filter & area_factor_filter & aspect_ratio_filter
	new_boxes = np.concatenate((xmin, ymin, xmax, ymax), axis=1)
	new_label = np.concatenate((label[:, 0:1], new_boxes), axis=1)
	if inds.any():
		new_label = new_label[inds]
	else:
		new_label = np.zeros(shape=(0, 5), dtype=np.float32)
	return transformed_img, new_label
	
def augment_hsv(img:np.ndarray, **hyp):
	hsv_gain = np.random.uniform(-1, 1, 3) * np.array([hyp['hsv_h'], hyp['hsv_s'], hyp['hsv_v']]) + 1
	h_gain, s_gain, v_gain = hsv_gain
	dtype = img.dtype
	table = np.arange(256)
	h_table = (table * h_gain).clip(0, 180).astype(dtype)
	s_table = (table * s_gain).clip(0, 255).astype(dtype)
	v_table = (table * v_gain).clip(0, 255).astype(dtype)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	H, S, V = img[..., 0], img[..., 1], img[..., 2]
	img = np.stack((h_table[H], s_table[S], v_table[V]), axis=-1)
	return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def wh_iou(boxes1:Tensor, boxes2:Tensor) -> Tensor:
	# compute 2 groups wh iou 
	# boxes1 [n1, 2] boxes2 [n2, 2]
	assert boxes1.shape[1] == 2, f'box shape error: b1 {boxes1.shape[1]}'
	assert boxes2.shape[1] == 2, f'box shape error: b2 {boxes2.shape[1]}'
	n1, n2 = boxes1.shape[0], boxes2.shape[0]
	w1, h1 = boxes1.t()
	w1 = w1.reshape(-1, 1)
	h1 = h1.reshape(-1, 1)
	w2, h2 = boxes2.t() 
	w2 = w2.reshape(1, -1)
	h2 = h2.reshape(1, -1)
	area1 = w1 * h1
	area2 = w2 * h2
	sum_area = area1 + area2
	repeat_w1 = w1.repeat(1, n2)
	repeat_w2 = w2.repeat(n1, 1)
	repeat_h1 = h1.repeat(1, n2)
	repeat_h2 = h2.repeat(n1, 1)
	i_w = torch.minimum(repeat_w1, repeat_w2)
	i_h = torch.minimum(repeat_h1, repeat_h2)
	i_aera = i_w * i_h
	u_aera = sum_area - i_aera
	return i_aera / u_aera

def compute_giou(boxes1:Tensor, boxes2:Tensor):
	# compute giou across boxes1 [n, 4] with boxes2 [n, 4] in format xywh
	assert boxes1.shape[0] == boxes2.shape[0], 'number of boxes not equal.'
	assert boxes1.shape[1] == 4 and boxes2.shape[1] == 4, 'boxes coords should be in shape [n, 4]'
	area1 = boxes1[:, 2] * boxes1[:, 3]
	area2 = boxes2[:, 2] * boxes2[:, 3]
	sum_area = area1 + area2
	boxes1 = xywh2xyxy(boxes1)
	boxes2 = xywh2xyxy(boxes2)
	b1x1, b1y1, b1x2, b1y2 = boxes1.t()
	b2x1, b2y1, b2x2, b2y2 = boxes2.t()
	intersect_x1 = torch.maximum(b1x1, b2x1)
	enclose_x1 = torch.minimum(b1x1, b2x1)
	intersect_y1 = torch.maximum(b1y1, b2y1)
	enclose_y1 = torch.minimum(b1y1, b2y1)

	intersect_x2 = torch.minimum(b1x2, b2x2)
	enclose_x2 = torch.maximum(b1x2, b2x2)
	intersect_y2 = torch.minimum(b1y2, b2y2)
	enclose_y2 = torch.maximum(b1y2, b2y2)

	intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
	enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
	union_area = sum_area - intersect_area

	iou = intersect_area / union_area
	return iou - (enclose_area - union_area) / enclose_area


if __name__ == '__main__':
	boxes = np.array([[1, 2, 3, 4], [2, 5, 3, 6]])
	print(xywh2xyxy(boxes))
