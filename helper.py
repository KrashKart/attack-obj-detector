import torch
import logging
from PIL import Image
from torchvision import transforms
import torchshow as ts
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, image, dist, elev, azim, pred=None):
        self.image = image
        self.dist = dist
        self.elev = elev
        self.azim = azim
        self.pred = None
    
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
    
    def get_pred(self):
        return self.pred


class blockOutput:
    def __enter__(self):
        logging.disable(logging.INFO)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.disable(logging.NOTSET)

    
def preprocess(image, purpose):
    """Carries out processing for image to make it compatible with functions
    
    Args:
        image (Tensor): Tensor rendered by renderer [1, RGB(A), W, H] or [1, W, H, RGB(A)]
        purpose (str): whether to process for "pred" [1, RGB, W, H], "view" [W, H, RGB] or "pil" [RGB, W, H]
    
    Returns:
        processed (Tensor): processed tensor
    """
    if isinstance(image, Render):
        image = image.get_image()
    if len(image.shape) == 3:
        image = torch.unsqueeze(image, 0)
    if image.shape[3] == 3 or image.shape[3] == 4: # [1, W, H, RGB(A)] -> [1, RGB(A), W, H]
        image = image.permute(0, 3, 1, 2)
    if image.shape[1] == 4: # [1, RGB(A), W, H] -> [1, RGB, W, H]
        image = image[:, :3, :, :]
    
    if purpose == "pred": # if pred, required shape is [1, RGB, W, H]
        return image
    elif purpose == "view": # if view, required shape is [W, H, RGB]
        return image.squeeze().permute(1, 2, 0)
    elif purpose == "pil":
        return image.squeeze()
    else:
        print("purpose must be 'pred', 'view' or 'pil'")
        

def search(dist, elev, azim, device, path="./data/", start="m1_v26_p0", end="rgb.png"):
    """FOR PASTE ONLY. Finds the picture file with the correct dist, elev and azim
    
    Args:
        dist, elev, azim (float): params
        device: device
        path, start, end (str): folder to check + formatting
    
    Returns:
        onto (Tensor): picture for pasting
    """
    
    dist, elev, azim = int(dist), int(elev), int(azim)
    respective = path + start + f"_Dis_{dist}_Azi_{azim}_Ele_{elev}_" + end
    onto = Image.open(respective)
    onto = onto.resize((640, 640))
    tran = transforms.PILToTensor()
    onto = tran(onto).to(device)

    return onto

        
def img_mask(img, onto, device):
    """Takes the img and pastes the car portion onto an image
    
    Args:
        img (Tensor): image tensor of the rendered car [W, H, 3] or [3, W, H]
        onto (Tensor): image tensor to paste onto [W, H, 3] or [3, W, H]
        device: device [1, i want starbucks]
        
    Returns:
        result (Tensor): resultant image
    """
    
    if isinstance(img, Render):
        img = img.get_image()
    
    if type(onto) != torch.Tensor:
        print("onto must be of type Tensor")
        return
    elif type(img) != torch.Tensor:
        print("img must be of type Tensor")
        return
    elif torch.max(onto) > 1:
        onto  = onto / 255
    
    img = preprocess(img, "pil") 
    onto = preprocess(onto, "pil")
    
    white = torch.ones(3, device=device)
    mask = torch.all(img == 1, dim=0)
    result = torch.where(mask, onto, img)
    result = torch.unsqueeze(result, 0)
    
    return result


def save(image, path="./results", **kwargs):
    """Saves rendered image to a defined path
    
    Args:
        image (Tensor): Tensor of shape [1, RGB, W, H]
        path (str): String containing destination and image name
        **kwargs (bruh): other args for ts.save
        
    Returns:
        None
    """
    
    if type(image) == list:
        output = []
        for render in image:
            output.append(preprocess(render.get_image(), "pil"))
        output = torch.stack(output)
        ts.save(output, **kwargs)
    else:
        if isinstance(image, Render):
            seeimage = image.get_image()
        else:
            seeimage = image
        saveimage = torch.clamp(seeimage, min=0, max=1)
        saveimage = preprocess(seeimage, "pil") 
        ts.save(seeimage, **kwargs)


def see(image, **kwargs):
    """Visualises the mesh
    
    Args:
        image (Tensor): Tensor containing information about a mesh
        **kwargs (lmaooooo): other args for ts.show
        
    Returns:
        None
    """   
    
    if type(image) == list:
        output = []
        for render in image:
            output.append(preprocess(render.get_image(), "pil"))
        output = torch.stack(output)
        ts.show(output, **kwargs)
    else:
        if isinstance(image, Render):
            seeimage = image.get_image()
        else:
            seeimage = image
        seeimage = torch.clamp(seeimage, min=0, max=1)
        seeimage = preprocess(seeimage, "pil") 
        ts.show(seeimage, **kwargs)

    
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