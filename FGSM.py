"""
 x_adv = x + ε · sign( ∇_x J(θ, x, y) )

Symbol glossary:
    x         — original clean input image (tensor of pixel values)
    y         — ground-truth label for x
    θ         — model parameters (weights & biases; fixed during FGSM)
    J(θ,x,y) — cross-entropy loss of the model at input x with label y
    ∇_x J     — gradient of the loss with respect to the INPUT pixels
    sign(·)   — element-wise sign: maps each scalar to +1 or −1
    ε         — perturbation budget (a small positive number, e.g. 0.25)
    x_adv     — adversarial image, visually similar to x but misclassified
"""

import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    criterion: nn.Module,
) -> torch.Tensor:
  
    # Step 1: enable gradient tracking on the input
    # clone() : makes a copy so we don't modify the original batch and use the clone to make the adverse
    # detach() : cuts any existing graph edges on this tensor
    # requires_grad_(True) — asks PyTorch to compute ∂loss/∂images
    images = images.clone().detach().requires_grad_(True)

    # Step 2: forward pass and compute the loss 
    outputs = model(images)               # logits: (N, num_classes)
    loss    = criterion(outputs, labels)  # scalar cross-entropy loss

    # Step 3: backward pass
    model.zero_grad()   # clear parameter gradients (θ stays fixed)
    loss.backward()     # differentiates loss with respect to every leaf with grad=True

    # images.grad now holds ∂J/∂x for every pixel — same shape as images
    data_grad = images.grad.data   # (N, C, H, W)

    # Steps 4 & 5: calculate the disturbance
    # .sign() maps each element: positive → +1, negative → -1, zero → 0
    perturbation      = epsilon * data_grad.sign()
    adversarial_images = images.detach() + perturbation

    # Step 6: clip to valid pixel range 
    # Without clamping, pixel values could exceed [0,1], which is physically impossible for normalized images.
    adversarial_images = torch.clamp(adversarial_images, min=0.0, max=1.0)

    return adversarial_images   # detached tensor, ready for inference


def fgsm_attack_batch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    epsilon: float,
    criterion: nn.Module,
    device: torch.device,
):
  
    model.eval()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = fgsm_attack(model, images, labels, epsilon, criterion)
        yield adv_images, labels