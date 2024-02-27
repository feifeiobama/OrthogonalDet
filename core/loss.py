import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from .util import box_ops
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
import copy
import numpy as np


class SetCriterionDynamicK(nn.Module):
    """ This class computes the training loss.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.start_count = 0
        self.start_iter = cfg.MODEL.CHANGE_START

        self.focal_loss_alpha = cfg.MODEL.ALPHA
        self.focal_loss_gamma = cfg.MODEL.GAMMA
        self.disentangled = cfg.MODEL.DISENTANGLED

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        if self.disentangled == 0:
            src_logits = outputs['pred_logits']
        else:
            assert 'pred_objectness' in outputs
            src_prob = torch.softmax(outputs['pred_logits'], dim=-1) * outputs['pred_objectness']
            src_logits = torch.log(src_prob / (1 - src_prob))
        batch_size = len(targets)

        if self.cfg.TEST.MASK == 2:
            seen_logits = list(range(0, self.cfg.TEST.PREV_INTRODUCED_CLS))
            masked_logit = src_logits.clone()
            masked_logit[..., seen_logits] = -10e10
            src_logits = masked_logit

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        src_logits_list = []
        target_classes_o_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        src_logits = src_logits.flatten(0, 1)
        target_classes_onehot = target_classes_onehot.flatten(0, 1)
        cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha,
                                              gamma=self.focal_loss_gamma, reduction="none")

        losses = {'loss_ce': torch.sum(cls_loss) / num_boxes}

        return losses

    def loss_nc_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        if self.disentangled == 0:
            src_logits = outputs['pred_logits']
        else:
            assert 'pred_objectness' in outputs
            src_prob = torch.softmax(outputs['pred_logits'], dim=-1) * outputs['pred_objectness']
            src_logits = torch.log(src_prob / (1 - src_prob))
        batch_size = len(targets)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        src_logits_list = []
        target_classes_o_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        src_logits = src_logits.flatten(0, 1)
        target_classes_onehot = target_classes_onehot.flatten(0, 1)
        cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha,
                                          gamma=self.focal_loss_gamma, reduction="none")

        losses = {'loss_nc_ce': torch.sum(cls_loss) / num_boxes}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes']

        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx]["boxes"]  # normalized (cx, cy, w, h)
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]  # absolute (x1, y1, x2, y2)
            pred_box_list.append(bz_src_boxes[valid_query])
            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)  # normalize (x1, y1, x2, y2)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])

        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            src_boxes_norm = torch.cat(pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = torch.cat(tgt_box_list)
            target_boxes_abs_xyxy = torch.cat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0]

            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(src_boxes_norm, box_cxcywh_to_xyxy(target_boxes), reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            # loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes_abs_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses

    def loss_decorr(self, outputs, targets, indices):
        assert self.disentangled != 0
        cls_scores = outputs['pred_logits'].softmax(-1).flatten(0, 1).detach()
        obj_score = outputs['pred_objectness'].reshape(-1)

        cls_mean, obj_mean = cls_scores.mean(dim=0), obj_score.mean()
        cov = ((cls_scores - cls_mean) * (obj_score - obj_mean)[:, np.newaxis]).sum(dim=0)
        var = ((cls_scores - cls_mean) ** 2).sum(dim=0) * ((obj_score - obj_mean) ** 2).sum()
        loss_decorr = (cov ** 2 / var).mean()

        return {'loss_decorr': loss_decorr}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'nc_labels': self.loss_nc_labels,
            'decorr': self.loss_decorr,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        self.start_count += 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, _, ow_indices, unknown_targets = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'nc_labels':
                if self.start_count > self.start_iter:
                    losses.update(self.get_loss(loss, outputs, unknown_targets, ow_indices))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _, ow_indices, unknown_targets = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'nc_labels':
                        if self.start_count > self.start_iter:
                            l_dict = self.get_loss(loss, aux_outputs, unknown_targets, ow_indices)
                            l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                            losses.update(l_dict)
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.ota_k = cfg.MODEL.OTA_K
        self.forward_k = cfg.MODEL.FORWARD_K
        self.cfg = cfg
        self.focal_loss_alpha = cfg.MODEL.ALPHA
        self.focal_loss_gamma = cfg.MODEL.GAMMA
        self.disentangled = cfg.MODEL.DISENTANGLED
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            if self.disentangled == 0:
                out_prob = outputs["pred_logits"].sigmoid()  # [batch_size, num_queries, num_classes]
            else:
                out_prob = torch.softmax(outputs['pred_logits'], dim=-1) * outputs['pred_objectness']
            out_bbox = outputs["pred_boxes"]  # [batch_size,  num_queries, 4]

            indices = []
            matched_ids = []
            unknown_labels = []
            ow_indices = []
            assert bs == len(targets)
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx]  # [num_proposals, 4]
                bz_out_prob = out_prob[batch_idx]
                bz_tgt_ids = targets[batch_idx]["labels"]
                num_insts = len(bz_tgt_ids)
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                else:
                    bz_gtboxs = targets[batch_idx]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h)
                    bz_gtboxs_abs_xyxy = targets[batch_idx]['boxes_xyxy']
                    fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                        box_xyxy_to_cxcywh(bz_boxes),  # absolute (cx, cy, w, h)
                        box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy),  # absolute (cx, cy, w, h)
                        expanded_strides=32
                    )

                    pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy)

                    # Compute the classification cost.
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]

                    # Compute the L1 cost between boxes
                    # image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
                    # image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
                    # image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

                    bz_image_size_out = targets[batch_idx]['image_size_xyxy']
                    bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt']

                    bz_out_bbox_ = bz_boxes / bz_image_size_out  # normalize (x1, y1, x2, y2)
                    bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt  # normalize (x1, y1, x2, y2)
                    cost_bbox = torch.cdist(bz_out_bbox_, bz_tgt_bbox_, p=1)

                    cost_giou = -generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy)

                    # Final cost matrix
                    cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + 100.0 * (
                        ~is_in_boxes_and_center)
                    # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
                    cost[~fg_mask] = cost[~fg_mask] + 10000.0

                    # if bz_gtboxs.shape[0]>0:
                    indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])

                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)

                if self.cfg.MODEL.NC:
                    #                 ==========================================================
                    forward_score = torch.sum(bz_out_prob, dim=1)
                    _, forward_index = torch.topk(forward_score, self.forward_k, largest=True, sorted=True)
                    forward_index_list = forward_index.cpu().numpy().tolist()
                    unknown_label = []
                    for each in forward_index_list:
                        if each not in matched_qidx.cpu().numpy().tolist():
                            unknown_label.append(each)
                    unknown_indices_batchi_a = copy.deepcopy(indices_batchi[0])
                    unknown_indices_batchi_b = torch.tensor([0] * len(unknown_label), dtype=torch.long,
                                                            device=unknown_indices_batchi_a.device)
                    for each in range(unknown_indices_batchi_a.shape[0]):
                        if each in unknown_label:
                            unknown_indices_batchi_a[each] = True
                        else:
                            unknown_indices_batchi_a[each] = False

                    unknown_labels.append(unknown_label)
                    ow_indices.append((unknown_indices_batchi_a, unknown_indices_batchi_b))
                    #                 ==========================================================

        #         ==================================================
        unknown_targets = []
        for i, unknown_label in enumerate(unknown_labels):
            unity_label = [self.cfg.MODEL.NUM_CLASSES-1] * len(unknown_label)
            unity_boxes = targets[i]['image_size_xyxy'].unsqueeze(0)
            unity_boxes_xyxy = targets[i]['image_size_xyxy'].unsqueeze(0)
            unity_image_size_xyxy_tgt = targets[i]['image_size_xyxy'].unsqueeze(0)
            unity_area = targets[i]['image_size_xyxy'].unsqueeze(0)
            unknown_target = {'labels': torch.tensor(unity_label, dtype=torch.long, device=targets[0]['labels'].device),
                              'boxes': unity_boxes.repeat(len(unknown_label), 1),
                              'image_size_xyxy': targets[i]['image_size_xyxy'],
                              'boxes_xyxy': unity_boxes_xyxy.repeat(len(unknown_label), 1),
                              'image_size_xyxy_tgt': unity_image_size_xyxy_tgt.repeat(len(unknown_label), 1),
                              'area': unity_area.repeat(len(unknown_label), 1)}
            unknown_targets.append(unknown_target)
        #         ==================================================

        return indices, matched_ids, ow_indices, unknown_targets

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)  # (x1, y1, x2, y2)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (
                    target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (
                    target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (
                    target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (
                    target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

        return (selected_query, gt_indices), matched_query_id
