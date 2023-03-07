import os
import requests
import torch
import torchvision
import matplotlib.pyplot as plt
import logging
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from render_functions import *
import numpy as np
from pytorch3d.renderer.camera_utils import join_cameras_as_batch as join_cameras
import torchshow as ts
from PIL import Image


class Camera:
    def __init__(self, camera, dist, elev, azim):
        self.camera = camera
        self.dist = dist
        self.elev = elev
        self.azim = azim
    
    def get_camera(self):
        return self.camera
    
    def get_dist(self):
        return float(self.dist)
    
    def get_elev(self):
        return float(self.elev)
    
    def get_azim(self):
        return float(self.azim)
    
    def get_params(self):
        return float(self.dist), float(self.elev), float(self.azim)
    
    def render(self, mesh, renderer):
        images = renderer(mesh, cameras=self.get_camera()) # [1, W, H, RGBA]
        images = Render(torch.clamp(images, min=0.0, max=1.0), self.get_dist(), self.get_elev(), self.get_azim())
        return images

    
class Render:
    def __init__(self, image, dist, elev, azim):
        self.image = image
        self.dist = dist
        self.elev = elev
        self.azim = azim
    
    def get_image(self):
        return self.image
    
    def get_dist(self):
        return float(self.dist)
    
    def get_elev(self):
        return float(self.elev)
    
    def get_azim(self):
        return float(self.azim)
    
    def get_params(self):
        return float(self.dist), float(self.elev), float(self.azim)
    

def load_model(name, device):
    """Loads YOLOv5 from hub and sets to eval
    
    Args:
        name (str): name of the model ("yolov5s", "yolov5m", etc)
        device: device
    
    Returns:
        model: model with pretrained weights
    """
    
    model = torch.hub.load('ultralytics/yolov5', name)
    model.eval()
    model.to(device)
    
    return model


def search(dist, elev, azim, device, path="./data/", start="m1_v26_p0", end="rgb.png"):
    dist, elev, azim = int(dist), int(elev), int(azim)
    respective = path + start + f"_Dis_{dist}_Azi_{azim}_Ele_{elev}_" + end
    onto = Image.open(respective)
    onto = onto.resize((640, 640))
    tran = transforms.PILToTensor()
    onto = torch.unsqueeze(tran(onto), 0)/255
    onto = preprocess(onto, "pred").to(device)
    return onto

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


def batch_predict(model, images, adverse=False, adverse_classes=None):
    """Conducts batch prediction on batch tensor
    
    Args:
        model: model
        images (Tensor): [batch_size, W, H, RGBA] shape tensor input
        adverse (bool): whether to mark adverse imag
        adverse_classes (int): class indices to mark as adverse
    
    Returns:
        predicts (dict): either a dictionary of predicted classes (YOLOv5l with Autoshape) 
                    or a [batch_size, N, 85] Tensor (DetectMultiBackend)
        adverse (dict): list of indices where the predictions are of adverse_class (if adverse is True)
    """
    
    predicts = {}
    adverses = {}
    with open("./coco_classes.txt", "r") as f:
        classes = [s.strip() for s in f.readlines()]
    images = preprocess(images, "pred")
    with torch.no_grad():
        for img_idx in range(len(images)):
            img = torch.unsqueeze(images[img_idx], 0)
            pred = predict(model, img)
            if len(pred.xyxy[0]) == 0:
                predicts["No Detection"] = predicts.get("No Detection", 0) + 1
            else:
                pred_class_idx = int(pred.xyxy[0][0][5])
                pred_class = classes[pred_class_idx]
                predicts[pred_class] = predicts.get(pred_class, 0) + 1
                if adverse and adverse_classes == pred_class_idx:
                    adverses[img_idx] = pred_class
    print("predicts: ", predicts)
    if adverse:
        print("adverse: ", adverses)
    return predicts, adverses if adverse and adverse_classes else predicts


def show_adverse(scene, renderer, adverses, test_idxs, cameras, **kwargs):
    """Renders and shows the adverse viewpoints and dist, elev, azim
    
    Args:
        mesh (Meshes): mesh
        renderer: renderer
        adverses (dict): dict of cam_idx: pred_class
        test_idxs (int list): list of indices of test camera views w.r.t. master camera list
        cameras (Cam list): list of (camera, d, e, a)s (master camera list)
    
    Returns:
        adverse_views (tup list): list of tuples containing (
    """
    
    if not adverses:
        print("No adverse views")
    else:
        adverse_test_cams_idx = [test_idxs[idx] for idx in adverses.keys()] # using adverse index to find the camera index of adverse cameras in test_idx
        adverse_cameras = [cameras[idx] for idx in adverse_test_cams_idx] # finding the camera with d, e, a in the master camera list
        adverse_views = [(d, e.item(), a.item()) for c, d, e, a in adverse_cameras] # extracting d, e, a for adverse cams
        adverse_cams = [camera[0] for camera in adverse_cameras] # compiling adverse FOVCAMs into list

        adverse_cams_join = join_cameras(adverse_cams)
        adverse_imgs = render_batch(scene, renderer, adverse_cams_join)
        adverse_imgs = torch.clamp(adverse_imgs, min=0.0, max=1.0)
        ts.show(adverse_imgs, axes_title="{img_id_from_1}", **kwargs)
        ts.save(adverse_imgs, axes_title="{img_id_from_1}", **kwargs)
        for i, (k, v) in enumerate(adverses.items()):
            print(f"Image ID {i + 1}: Test cam {k} {adverse_views[i]} --> {v}")
        return adverse_views


def plot_progress(returns, preds, iters_no, labels, path="./images/progress.jpg"):
    """Plots and saves periodical images and detections to show progress of the training loop
    
    Args:
        returns (Tensor list): list of image tensors [W, H, 3]
        preds (str list): list of detected classes
        iters_no (int list): list of int iteration numbers
        labels (str list): list of all labels
        path (str): path to save plot to
        
        All args were extracted at regular iteration intervals during the training process
    
    Returns:
        None
    """

    length = len(returns)
    rows = length // 5 + 1 if length % 5 != 0 else length // 5
    columns = 5

    fig, ax = plt.subplots(rows, columns, figsize=(30, 3 * columns))
    
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    for idx, tensor in enumerate(returns):
        rowN = idx // 5
        columnN = idx % 5
        ax[rowN][columnN].imshow(tensor)
        label = "No Detection" if preds[idx] == None else preds[idx]
        if idx == 0:
            ax[rowN][columnN].set_title(f"Original: {label}")
        else:
            ax[rowN][columnN].set_title(f"Prediction (Iteration {iters_no[idx]}): {label}")
    plt.savefig(path)
    logger.setLevel(old_level)

        
def plot_loss(losses, path="./images/losses.jpg"):
    """Plots and saves the losses over iterations for visualisation
    
    Args:
        losses (list): list of losses over all iterations
        path (str): path to save plot to 
        
    Returns:
        None
    """
    
    plt.figure(figsize=(20, 6))
    iters = np.arange(len(losses))
    plt.plot(iters, losses)
    plt.title("Loss function (NLL_loss) over iterations")
    plt.xlabel("# of iterations")
    plt.ylabel("Loss")
    plt.savefig(path)
    plt.show()
    
    
def plot_classes(classes, path="./images/classes.jpg"):
    """Plots the progressions of detected classes over the training loop
    
    Args:
        classes (int list): list of detected classes
        path (str): path to save plot to
        
    Returns:
        None
    """
    
    iters = np.arange(0, 200, 200/len(classes))
    plt.figure(figsize=(20, 6))
    plt.plot(iters, classes)
    plt.title("Predicted Classes over time")
    plt.xlabel("# of iterations")
    plt.ylabel("Classes (0 - 79)")
    plt.ylim(0, 80)
    plt.savefig(path)
    plt.show()