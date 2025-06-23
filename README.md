# Adversarial Attack on Object Detector (2023)
This repo contains code written from scratch by me for my research internship regarding adversarial attack on object detectors using gradient-based methods.

!(structure.png)

The way this was achieved was to establish a PyTorch autograd generator and a discriminator (either Yolov5 trained on COCO or VisDrone). The idea was to pass PyTorch3D-rendered structures through the discriminator using a randomly initialised texture, and for the PyTorch autograd to backpropagate the gradients from the discriminator loss back to the generator, in turn perturbating the trained texture in the direction of minimal loss.
