import tensorflow.keras.backend as K
import tensorflow as tf
import math

from yolo.utils.box_utils import _xcycwh_to_xyxy 
from yolo.utils.box_utils import _xcycwh_to_yxyx 
from yolo.utils.box_utils import _yxyx_to_xcycwh 
from yolo.utils.box_utils import _get_area 
from yolo.utils.box_utils import _intersection_and_union
from yolo.utils.box_utils import _aspect_ratio_consistancy 
from yolo.utils.box_utils import _center_distance 

def compute_iou(box1, box2):
    # get box corners 
    with tf.name_scope("iou"):
        box1 = _xcycwh_to_xyxy(box1)
        box2 = _xcycwh_to_xyxy(box2)
        intersection, union = _intersection_and_union(box1, box2)
        
        iou = tf.math.divide_no_nan(intersection, union)
        iou = tf.clip_by_value(iou, clip_value_min = 0.0, clip_value_max = 1.0)
    return iou


def compute_giou(box1, box2):
    with tf.name_scope("giou"):
        # get box corners 
        box1 = _xcycwh_to_xyxy(box1)
        box2 = _xcycwh_to_xyxy(box2)
        
        # compute IOU
        intersection, union = _intersection_and_union(box1, box2)
        iou = tf.math.divide_no_nan(intersection, union)
        iou = tf.clip_by_value(iou, clip_value_min = 0.0, clip_value_max = 1.0)
        
        # find the smallest box to encompase both box1 and box2
        c_mins = K.minimum(box1[..., 0:2], box2[..., 0:2])
        c_maxes = K.maximum(box1[..., 2:4], box2[..., 2:4])
        c = _get_area((c_mins, c_maxes), use_tuple = True)

        # compute giou
        giou = iou - tf.math.divide_no_nan((c - union), c)
    return iou, giou

def compute_diou(box1, box2):
    with tf.name_scope("diou"):
        # compute center distance
        dist = _center_distance(box1[..., 0:2], box2[..., 0:2])
        
        # get box corners 
        box1 = _xcycwh_to_xyxy(box1)
        box2 = _xcycwh_to_xyxy(box2)

        # compute IOU
        intersection, union = _intersection_and_union(box1, box2)
        iou = tf.math.divide_no_nan(intersection, (union + 1e-16))
        iou = tf.clip_by_value(iou, clip_value_min = 0.0, clip_value_max = 1.0)
        
        # compute max diagnal of the smallest enclosing box
        c_mins = K.minimum(box1[..., 0:2], box2[..., 0:2])
        c_maxes = K.maximum(box1[..., 2:4], box2[..., 2:4])
        diag_dist = _center_distance(c_mins, c_maxes)
        
        regularization = tf.math.divide_no_nan(dist,diag_dist)  
        diou = iou + regularization
    return iou, diou

def compute_ciou(box1, box2):
    with tf.name_scope("ciou"):
        #compute DIOU and IOU
        iou, diou = compute_diou(box1, box2)
        
        # computer aspect ratio consistency
        v = _aspect_ratio_consistancy(box1[..., 2],box1[..., 3], box2[..., 2], box2[..., 3])
        
        # compute IOU regularization
        a = v/((1 - iou) + v)
        ciou = diou + v * a
    return iou, ciou