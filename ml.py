import os
import requests
import torch
import torchvision
import matplotlib.pyplot as plt
import logging
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
# from render_functions import *
import numpy as np
from pytorch3d.renderer.camera_utils import join_cameras_as_batch as join_cameras
import torchshow as ts

from helper import *


def load_model(name, device, conf, iou, max_det):
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


def batch_predict(model, images, adverse=False, adverse_classes=2):
    """Conducts batch prediction on batch tensor
    
    Args:
        model: model
        images (Render list): batch_size list of Renders
        adverse (bool): whether to mark adverse imag
        adverse_classes (int): class indices to mark as adverse
    
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
        for render in images:
            pred = predict(model, render.get_image())
            
            if len(pred.xyxy[0]) == 0:
                pred_class = "No Detection"
                pred_class_idx = -1
            else:
                pred_class_idx = int(pred.xyxy[0][0][5])
                pred_class = classes[pred_class_idx]
            
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