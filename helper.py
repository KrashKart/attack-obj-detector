import torch
import logging
from PIL import Image
from torchvision import transforms
import torchshow as ts
import numpy as np
import matplotlib.pyplot as plt
from platform import python_version
import pytorch3d
from multiprocessing import Pool
import time

class Camera:
    """Camera class to store dist, elev and azim for easier handling
    """
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
    """Render class to store camera params and prediction
    """
    def __init__(self, tensor, dist, elev, azim, pred=None):
        tensor = preprocess(tensor, "render_save")
        self.tensor = tensor # [1, W, H, 4] or [1, W, H, 3]
        self.dist = dist
        self.elev = elev
        self.azim = azim
        self.pred = pred
    
    def get_tensor(self): # [1, W, H, 4]
        return self.tensor
    
    def get_image(self):  # returns img tensor [1, W, H, 3]
        return self.tensor.clone()[..., :3]
    
    def get_sil(self): # [W, H] (can be expanded to [W, H, 1] after external processing)
        return self.tensor.clone()[..., 3].squeeze(0)

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
    """Used to block logging of INFO level or lower
    """
    def __enter__(self):
        logging.disable(logging.INFO)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.disable(logging.NOTSET)

        
def specs():
    """Prints crucial package version for debug and confirmation
    
    Args:
        None
       
    Returns:
        None
    """
    
    print(f"Python version: {python_version()}")
    print(f"PyTorch3D version: {pytorch3d.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    
def set_device():
    """Sets the device to either "cuda:0" if available or "cpu" otherwise
    Args:
        None
        
    Returns:
        device: device
    """

    # check if cuda is available, set device cuda if True and cpu if False
    print(f"torch.cuda.is_available() is {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("Cuda available")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        print("Cuda not available, using cpu")
        device = torch.device("cpu")
    return device


def preprocess(image, purpose):
    """Carries out processing for image to make it compatible with functions
    
    Args:
        image (Tensor): Tensor rendered by renderer [1, RGB(A), W, H] or [1, W, H, RGB(A)]
        purpose (str): whether to process for "pred" [1, RGB, W, H], "view" [W, H, RGB] or "pil" [RGB, W, H]
    
    Returns:
        image (Tensor): processed tensor
    """
    if isinstance(image, Render):
        image = image.get_image()
    if len(image.shape) == 2:
        image = torch.unsqueeze(image, 2)
    if len(image.shape) == 3:
        image = torch.unsqueeze(image, 0)
    if image.shape[3] in [3, 4]:
        image = image.permute(0, 3, 1, 2)
    
    if purpose == "pred" or purpose == "pil":
        if image.shape[1] == 4: # [1, RGB(A), W, H] -> [1, RGB, W, H]
            image = image[:, :3, :, :]
        return image if purpose == "pred" else image.squeeze(0)
    elif purpose == "render_save":
        return image.permute(0, 2, 3, 1)
    else:
        print("purpose must be 'pred', 'view' or 'pil'")

        
def load_data(**kwargs):
    start = time.time_ns()
    selected = sorted(os.listdir("./data"))
    output = []
    
    if "elev" in kwargs:
        kwargs["Ele"] = kwargs["elev"]
        del kwargs["elev"]
    if "azim" in kwargs:
        kwargs["Azi"] = kwargs["azim"]
        del kwargs["azim"]
    if "dist" in kwargs:
        kwargs["Dis"] = kwargs["dist"]
        del kwargs["dist"]
    
    for param in ["p", "Dis", "Ele", "Azi"]:
        if param in kwargs:
            if param != "p":
                selected = [file for file in selected if param + "_" + str(kwargs[param]) in file]
            else:
                selected = [file for file in selected if param + str(kwargs[param]) in file]
 
    for filename in selected:
        temp = Image.open("./data/" + filename)
        pipe = tran.ToTensor()
        image = pipe(temp)
        filestuff = filename.split("_")
        d, e, a = filestuff[4], filestuff[6], filestuff[8]
        render = Render(image, d, e, a)
        output.append(render)
    output.sort(key=lambda x: x.get_params())
    time_taken = time.time_ns() - start
    print(f"Time taken for data load: {time_taken}")
    print(f"{len(output)} images loaded")
    return output
        
def img_mask(image, onto, device):
    """Takes the img and pastes the car portion onto an image
    
    Args:
        image (Tensor): image tensor or Render image
        onto (Tensor): image tensor
        device: device [1, i want starbucks]
        
    Returns:
        result (Tensor): resultant image
    """
    
    if isinstance(image, Render):
        img = image.get_image()
        silhouette = image.get_sil().unsqueeze(2)
    
    if isinstance(onto, Render):
        onto = onto.get_image()
    
    if type(onto) != torch.Tensor:
        print("onto must be of type Tensor")
        return
    elif torch.max(onto) > 1:
        onto  = onto / 255
    
    img = preprocess(img, "pil")
    onto = preprocess(onto, "pil")
    silhouette = preprocess(silhouette, "pil")
    
    mask = torch.all(silhouette == 0.0, dim=2)
    result = torch.where(mask, onto, img)
    result = torch.unsqueeze(result, 0)
    
    return result


def save(image, path="./results", **kwargs):
    """Saves rendered image to a defined path
    
    Args:
        image (Tensor) or (Tensor list): Tensor or batch of Tensors in a list
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
        image (Tensor) or (Tensor list): Tensor or batch of Tensors in a list
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
    
    iters = np.arange(0, 100, 100/len(classes))
    plt.figure(figsize=(20, 6))
    plt.plot(iters, classes)
    plt.title("Predicted Classes over time")
    plt.xlabel("# of iterations")
    plt.ylabel("Classes (0 - 79)")
    plt.ylim(-2, 80)
    plt.savefig(path)
    plt.show()