import os
import requests
import torch
import torchvision
import matplotlib.pyplot as plt
import logging
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from pytorch3d.renderer.camera_utils import join_cameras_as_batch as join_cameras
import torchshow as ts

from helper import *


def load_model(name, device, conf=0.25, iou=0.45, max_det=5):
    """Loads YOLOv5 from hub and sets to eval
    
    Args:
        name (str): name of the model ("yolov5s", "yolov5m", etc)
        device: device
        conf, iou, max_det (float): params for YOLOv5
    
    Returns:
        model: model with pretrained weights
    """
    
    model = torch.hub.load('ultralytics/yolov5', name)
    model.conf = conf
    model.iou = iou
    model.max_det = max_det
    model.eval()
    model.to(device)
    
    return model

def predict(model, image, show=False):
    """Passes image tensor through the model for prediction
    
    Args:
        model: model (must be compatible with [1, RGB, W, H] input)
        image (Tensor): image tensor for prediction [1, RGB(A), W, H] (partially processed) 
                        or [1, W, H, RGB(A)] (freshly rendered tensor)
        show (bool): if True, prints prediction results
        
    Returns:
        pred (Det): returns the detections tensor
    """
    
    transform = transforms.ToPILImage()
    image_pic = transform(preprocess(image, "pil"))
    pred = model(image_pic)
    if show:
        pred.show()
        pred.print()
    return pred


def iou(boxA, boxB):
    """Returns iou score of boxA over boxB
    
    Args:
        boxA (tuple): tuple of xmin, ymin, xmax, ymax
        boxB (tuple): tuple of xmin, ymin, xmax, ymax
    
    Returns:
        iou (float): iou score
    
    """
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def find_bbox(silhouette):
    """Finds bbox coordinates
    
    Args:
        silhouette (Tensor): tensor of the silhouette [W, H]
    
    Returns:
        x_min, y_min, x_max, y_max (int tuple): bbox coords tuple
    """
    
    nonzero_indices = torch.nonzero(silhouette)
    
    # find the minimum and maximum x and y coordinates of the non-zero elements
    x_min = nonzero_indices[:, 1].min().item()
    y_min = nonzero_indices[:, 0].min().item()
    x_max = nonzero_indices[:, 1].max().item()
    y_max = nonzero_indices[:, 0].max().item()
    return (x_min, y_min, x_max, y_max)

def draw_bbox(img_tensor, coords):
    """Draws bbox on image using coords
    
    Args:
        img_tensor (Tensor): image tensor
        coord (tuple): bbox coords tuple
    
    Returns:
        copy (Tensor): copy of img_tensor with bbox drawn
    """
    xmin, xmax, ymin, ymax = coords
    copy = preprocess(img_tensor.clone().detach(), "pil")
    copy[..., ymin, xmin:xmax+1, :] = 1.0 - copy[..., ymin, xmin:xmax+1, :]
    copy[..., ymax, xmin:xmax+1, :] = 1.0 - copy[..., ymax, xmin:xmax+1, :]
    copy[..., ymin:ymax+1, xmin, :] = 1.0 - copy[..., ymin:ymax+1, xmin, :]
    copy[..., ymin:ymax+1, xmax, :] = 1.0 - copy[..., ymin:ymax+1, xmax, :]
    return copy


def batch_predict(model, images, coords, adverse=False, adverse_classes=2, iou_thres=0.25):
    """Conducts batch prediction on batch tensor
    
    Args:
        model: model
        images (Render list): batch_size list of Renders
        coords (tuple list): list of bbox coord tuples per render
        adverse (bool): whether to mark adverse imag
        adverse_classes (int): class indices to mark as adverse
        iou_thresh (float): threshold of iou for prediction against bbox coords
    
    Returns:
        predicts (Render list): Resultant predictions on the list of Renders
        predict_count (dict): Dict containing predicted_classes : qty 
        adverse (Render list): list of adverse image Renders
    """

    predicts = []
    predict_count = {}
    adverses = []
    with open("./coco_classes.txt", "r") as f:
        classes = [s.strip() for s in f.readlines()]
    with torch.no_grad():
        for idx, render in enumerate(images):
            pred = predict(model, render.get_image())
            while True:
                if len(pred.xyxy[0]) == 0:
                    pred_class = "No Detection"
                    pred_class_idx = -1
                    break
                else:
                    pred_box = tuple(pred.xyxy[0][0][:4])
                    pred_iou = iou(pred_box, coords[idx])
                    if pred_iou >= iou_thres:
                        pred_class_idx = int(pred.xyxy[0][0][5])
                        pred.xyxy[0] = torch.unsqueeze(pred.xyxy[0][0], 0)
                        pred_class = classes[pred_class_idx]
                        break
                    else:
                        pred.xyxy[0] = pred.xyxy[0][1:]
            
            predict_count[pred_class] = predict_count.get(pred_class, 0) + 1
            
            with blockOutput():
                pred.save(exist_ok=True)
            img = Image.open("./runs/detect/exp/image0.jpg")
            pred_img = transforms.PILToTensor()(img)
            
            d, e, a = render.get_params()
            predicts.append(Render(pred_img, d, e, a, pred_class))
            if adverse and adverse_classes == pred_class_idx:
                adverses.append(Render(pred_img, d, e, a, pred_class))
    
    if adverse and adverse_classes:
        return predicts, predict_count, adverses
    else:
        return predicts, predict_count


def show_adverse(adverses, **kwargs):
    """FOR PASTE ONLY. Renders and shows the adverse viewpoints and dist, elev, azim
    
    Args:
        adverses (Render list): list of adverse Renders
        ** kwargs: kwargs for ts.show
    
    Returns:
        None
    """
    
    if not adverses:
        print("No adverse views")
    else:
        for adverse in adverses:  # adverse = (Render object, pred)
            d, e, a = adverse.get_params()
            image = adverse.get_image()
            pred = adverse.get_pred()
            ts.show(image[:3, ...], suptitle=f"{pred}: Dist {d}, Elev {e}, Azim {a}", **kwargs)