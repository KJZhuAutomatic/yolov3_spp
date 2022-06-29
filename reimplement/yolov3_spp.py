from .model_utils import *
import torch
from .utils import wh_iou, compute_giou

class YoloV3SPP(nn.Module):
	"""implement of yolov3spp network."""
	def __init__(self, cfg:str):
		super(YoloV3SPP, self).__init__()
		self.module_list, self.routs = create_modules(parse_model_cfg(cfg))
		self.yolo_layers = [idx for idx, module in enumerate(self.module_list) \
		                            if module.__class__.__name__ == 'YoloLayer' ]

	def forward(self, x:Tensor):
		# x [bs, 3, img_h, img_w]
		assert not torch.fmod(torch.as_tensor(x.shape[2:]), 32).type(torch.bool).any(), f'input size must be a multiple of 32, get {x.shape[2:]}' 
		outs:List[Tensor] = []  # record each layer out which is necessary for others
		yolo_outs:List[Union[Tensor, Tuple[Tensor, Tensor]]] = [] # recore each yolo layer out
		for idx, module in enumerate(self.module_list):
			if module.__class__.__name__ in ['ConcatLayer', 'SumLayer']:
				x = module(x, outs)
			elif module.__class__.__name__ == 'YoloLayer':
				yolo_outs.append(module(x))
			else:
				x = module(x)
			if self.routs[idx]:
				outs.append(x)
			else:
				outs.append(None)
		if self.training:
			return yolo_outs
		else:
			inference_ls, _ =  tuple(zip(*yolo_outs))
			return torch.cat(inference_ls, axis=1), None

	def compute_loss(self, preds:List[Tensor], targets:Tensor, **params):
		'''
		# preds: list of Tensor[bs, na, gh, gw, nc+5] last dim [xywh, obj, cls]
		# targets: [num_targets, 6] last dim: [img_idx, cls, x, y, w, h]
		# params['giou'], params['cls'], params['obj'], params['iou_t']
		'''
		device = preds[0].device
		dtype = preds[0].dtype
		grid_sizes = [tuple(p.shape[2:4]) for p in preds]
		box_tgts, cls_tgts, pos_inds, anchor_tgts = \
		           self.build_target(targets, grid_sizes, params['iou_t'], device)
		lbox = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
		lcls = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
		lobj = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
		bce_loss = torch.nn.functional.binary_cross_entropy_with_logits
		for i, pred in enumerate(preds):
			pred_obj = pred[..., 4]
			tgt_obj = torch.zeros_like(pred_obj, dtype=dtype, device=device)
			b_ids, a_ids, g_i, g_j = pos_inds[i]
			num_pos = b_ids.shape[0] # get none sample is possible
			if num_pos:
				pos_pred = pred[b_ids, a_ids, g_i, g_j]
				pred_xy = pos_pred[:, :2].sigmoid()
				pred_wh = pos_pred[:, 2:4].exp().clamp(max=1E3) * anchor_tgts[i]
				pred_box = torch.cat((pred_xy, pred_wh), axis=1)
				giou = compute_giou(pred_box, box_tgts[i])
				tgt_obj[b_ids, a_ids, g_i, g_j] = giou.detach().clamp(min=0)
				pred_cls = pos_pred[:, 5:]
				tgt_cls = torch.zeros_like(pred_cls, dtype=dtype, device=device)
				# This is an error wasting a morning. Note the `:` get all along dimension.
				# tgt_cls[:, cls_tgts[i]] = 1
				tgt_cls[range(num_pos), cls_tgts[i]] = 1
				lcls = lcls + bce_loss(pred_cls, tgt_cls)
				# print(cls_tgts[i], 'in my loss')
				lbox = lbox + (1 - giou).mean()
			lobj = lobj + bce_loss(pred_obj, tgt_obj)
		return {'box_loss': lbox * params['giou'],
		        'obj_loss': lobj * params['obj'],
		        'class_loss': lcls * params['cls']}


	def build_target(self, targets:Tensor,
	                 grid_sizes:List[Tuple[int, int]], 
		             iou_t:float, device):
		'''
		build target for each yolo layer
		# targets [num_targets, 6] last dim: [img_idx, cls, x, y, w, h]
		# grid_sizes (grid_h, grid_w) for sequential 3 layers
		# iou_t: iou with targets and anchors threshold   
		'''
		num_tgt = targets.shape[0]
		# record positive sample (batch_size, anchor_id, grid_i, grid_j)
		pos_inds: List[Tuple[Tensor, Tensor, Tensor, Tensor]] = []
		target_cls: List[Tensor] = [] # record cls, Tensor shape [num_pos, ]
		# record box, Tensor[num_pos, 4] last dim [x-j, y-i, w, h]
		target_box: List[Tensor] = [] 
		# positive targets corresponding anchors
		target_anchors: List[Tensor] = []
		for idx, g_s in enumerate(grid_sizes):
			# adjust targets(relative) to current grid size
			gain = torch.as_tensor([1, 1, g_s[1], g_s[0], g_s[1], g_s[0]], \
				                    device=device).unsqueeze(0)
			zoomed_targets = targets * gain
			# get anchors adaptive to current grid size
			anchors = self.module_list[self.yolo_layers[idx]].normalized_anchors.to(device)
			# find positive sample
			acr_ids, tgt_ids = torch.nonzero(wh_iou(anchors, zoomed_targets[:, -2:]) > iou_t, as_tuple=True)
			# record num of positive sample corresponding anchros
			self.module_list[self.yolo_layers[idx]].count(acr_ids) 
			# choose positive targets
			pos_tgts = zoomed_targets[tgt_ids]
			img_ids, tgt_cls, grid_xy, grid_wh = pos_tgts[:, 0], pos_tgts[:, 1], pos_tgts[:, 2:4], pos_tgts[:, 4:]
			grid_ji = grid_xy.long()
			tgt_box = torch.cat((grid_xy - grid_ji, grid_wh), axis=1)
			pos_inds.append((pos_tgts[:, 0].long(), acr_ids, grid_ji[:, 1], grid_ji[:, 0]))
			target_box.append(tgt_box.to(device))
			target_cls.append(tgt_cls.long().to(device))
			target_anchors.append(anchors[acr_ids].to(device))
		return target_box, target_cls, pos_inds, target_anchors

if __name__ == '__main__':
	net = YoloV3SPP('cfg/my_yolov3.cfg')
	'''
	print(list(net.state_dict().keys())[20:25])
	print(list(torch.load('../weights/yolov3spp-voc-512.pt')['model'].keys())[20:25])
	'''
	net.load_state_dict(torch.load('weights/yolov3spp-voc-512.pt')['model'])
	net.eval()
	net(torch.zeros(1, 3, 512, 512))
