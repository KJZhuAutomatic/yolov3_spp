import os
import numpy as np
import torch.nn as nn
from typing import *
from .yolo_layers import *

def is_newblock(x: str):
	return x.startswith('[')

def parse_model_cfg(cfg: str):
	
	if not (os.path.exists(cfg) and cfg.endswith('.cfg')):
		raise FileNotFoundError('%s not found or name error.'%cfg)

	with open(cfg) as f:
		lines = f.readlines()

	parsed_dict_list: List[Dict[str]] = []

	for line in lines:

		line = line.strip() # remove space string at the start and end of the line

		if not line: # empty line
			continue

		if line.startswith('#'): # annotation line
			continue

		if is_newblock(line):
			new_block: Dict[str] = {}
			new_block['type'] = (line[1:-1]).strip()
			parsed_dict_list.append(new_block)
		else:
			curr_blk = parsed_dict_list[-1]
			key, val = line.split('=')
			key = key.strip()
			val = val.strip()
			if key in ['layers', 'mask', 'from']:
				curr_blk[key] = [int(s.strip()) for s in val.split(',')]
			elif key == 'anchors':
				val_ls = [float(v) for v in val.replace(' ', '').split(',')]
				anchors = np.array(val_ls, dtype=np.float32).reshape(-1, 2)
				curr_blk[key] = anchors				
			else:
				try:
					curr_blk[key] = float(val) 
				except ValueError: # val is string
					curr_blk[key] = val

	return parsed_dict_list

def create_modules(parsed_dict_list: List[Dict[str, Any]]) -> Tuple[nn.ModuleList, List[bool]]:
	'''
	Return module list and rout flag list given the parsed modules params dict list.
	'''
	modules_ls = nn.ModuleList()
	parsed_dict_list.pop(0) # pop the first layer whose type = net
	
	# record layers whose output is required by other layers.
	routs: List[bool] = [False] * len(parsed_dict_list) 
	last_layer_out_channels: List[int] = [3] # in channels list
	yolo_idx = 0

	for idx, module_param_dict in enumerate(parsed_dict_list):
		
		module_type = module_param_dict['type']
		
		if module_type == 'convolutional':
			module = nn.Sequential()

			out_channels = int(module_param_dict['filters'])
			kernel_size = int(module_param_dict['size'])
			stride = int(module_param_dict['stride']), 
			pad = bool(module_param_dict['pad'])
			pad_val = kernel_size // 2 if pad else None
			bn_enable = module_param_dict.get('batch_normalize') and module_param_dict['batch_normalize']
			conv = nn.Conv2d(in_channels=last_layer_out_channels[idx],
				             out_channels=out_channels, kernel_size=kernel_size,
				             stride=stride, padding=pad_val, bias=not bn_enable)
			module.add_module('Conv2d', conv)

			if bn_enable:
				bn = nn.BatchNorm2d(out_channels)
				module.add_module('BatchNorm2d', bn)

			activation = module_param_dict['activation']
			if activation == 'leaky':
				module.append(nn.LeakyReLU(negative_slope=0.1))
			elif activation == 'linear':
				pass
			else:
				raise NameError('Undefined activation %s.'%activation)

		elif module_type == 'shortcut':
			from_layer = module_param_dict['from'][0]
			if from_layer < 0:
				from_layer += idx
			routs[from_layer] = True
			module = SumLayer(from_layer)
			if not (module_param_dict['activation'] == 'linear'):
				raise NameError('Undefined activation %s.'%module_param_dict['activation'])

		elif module_type == 'maxpool':
			stride = int(module_param_dict['stride'])
			assert stride == 1, 'stride of maxpool should be 1, get %d.'%stride
			size = int(module_param_dict['size'])
			pad = (size - 1) // 2
			module = nn.MaxPool2d(kernel_size=size, stride=stride, padding=pad)

		elif module_type == 'route':
			out_channels = 0
			layers = module_param_dict['layers']
			# print(layers)
			for i in range(len(layers)):
				if layers[i] < 0:
					layers[i] += idx
				routs[layers[i]] = True
				out_channels += last_layer_out_channels[layers[i]+1] 
				# print(last_layer_out_channels[layers[i]+1], modules_ls[layers[i]].__class__.__name__)
			module = ConcatLayer(layers)

		elif module_type == 'yolo':
			strides_ls = [32, 16, 8]
			anchors = module_param_dict['anchors']
			mask = np.array(module_param_dict['mask'])
			num_classes = int(module_param_dict['classes'])
			module = YoloLayer(anchors[mask], strides_ls[yolo_idx])
			yolo_idx += 1

		elif module_type == 'upsample':
			module = nn.Upsample(scale_factor=int(module_param_dict['stride']))

		else:
			raise NameError('Undefined layer name %s'%module_type)

		last_layer_out_channels.append(out_channels)
		modules_ls.append(module)

	return modules_ls, routs

if __name__ == '__main__':
	create_modules(parse_model_cfg('../cfg/my_yolov3.cfg'))
	# print(os.getcwd())