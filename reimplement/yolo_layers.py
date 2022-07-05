import torch.nn as nn
from torch import Tensor
from typing import *
import torch

class SumLayer(nn.Module):
	"""Sum with shortcut feature in Residual of Darknet."""
	def __init__(self, shortcut_layer:int):
		super(SumLayer, self).__init__()
		if shortcut_layer >= 0:
			self.layer = shortcut_layer
		else:
			raise ValueError('Short cut layer index cannot be negative, but get %d.'%shortcut_layer)

	def forward(self, x:Tensor, outs:List[Optional[Tensor]]):
		shortcut_feature = outs[self.layer]
		assert x.size() == shortcut_feature.size(), f'feature size {x.size()} unidentical with shortcut {shortcut_feature.size()}'
		return x + shortcut_feature

class ConcatLayer(nn.Module):
	"""Concat the features in channel dimension."""
	def __init__(self, layers:List[int]):
		super(ConcatLayer, self).__init__()
		for l in layers:
			assert l>0, 'layer index in ConcatLayer cannot be negative, but get %d.'%l
		self.layers = layers
		
	def forward(self, x:Tensor, outs:List[Optional[Tensor]]):
		return torch.cat([outs[l] for l in self.layers], 1)

class YoloLayer(nn.Module):
	"""Given feature from conv, reshape and output directly in training, 
	   or reshape in inference."""
	def __init__(self, anchors, stride:int):
		super(YoloLayer, self).__init__()
		assert anchors.shape == (3, 2), f'yolo anchors shape should be [3, 2], get {anchors.shape}.'
		self.normalized_anchors = Tensor(anchors / stride)
		self.num_anchors = self.normalized_anchors.shape[0]
		self.anchor_vec = self.normalized_anchors
		self.stride = stride
		self.grid = None
		self.used_anchor_count = [0] * self.normalized_anchors.shape[0]

	def create_grid(self, height:int, width:int):
		y_idxs = torch.arange(height)
		x_idxs = torch.arange(width)
		grid_x, grid_y = torch.meshgrid(x_idxs, y_idxs, indexing='xy')
		self.grid = torch.cat((grid_x.unsqueeze(2), grid_y.unsqueeze(2)), 2)

	def count(self, inds):
		for idx in inds:
			self.used_anchor_count[idx] += 1

	def forward(self, x):
		na = self.normalized_anchors.shape[0]
		gh, gw = tuple(x.shape[-2:])
		bs = x.shape[0]
		nc = x.shape[1] // na - 5
		# x [bs, (nc+5)*na, gh, gw] -> [bs, na, gh, gw, nc+5]
		reshaped_pred = x.view((bs, na, -1, gh, gw)).permute((0, 1, 3, 4, 2)).contiguous()
		if self.training:
			return reshaped_pred
		else:
			# [xywh+obj+cls]
			infered_pred = reshaped_pred.clone()
			# if not (self.grid and self.grid.shape[:2] == x.shape[-2:]):
			if self.grid is None or (self.grid.shape[0]!=x.shape[-2]) or (self.grid.shape[1]!=x.shape[-1]):
				self.create_grid(gh, gw)
			self.grid = self.grid.to(x.device)
			self.normalized_anchors = self.normalized_anchors.to(x.device)
			infered_pred[..., :2].sigmoid_()
			infered_pred[..., :2] += self.grid[None, None] 
			infered_pred[..., 2:4].exp_() 
			infered_pred[..., 2:4] *= self.normalized_anchors[None, :, None, None, :]
			infered_pred[..., :4] *= self.stride
			infered_pred[..., 4:].sigmoid_()
			return infered_pred.view(bs, -1, nc+5), reshaped_pred