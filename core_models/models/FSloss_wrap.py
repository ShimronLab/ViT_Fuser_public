import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from fastmri.models import Unet
import torchvision
from torch.nn import functional as F
from torchvision import models, transforms


import torch

class CustomL1L2Loss(nn.Module):
    def __init__(self, a):
        """
        Initializes the custom loss function.
        Args:
            a (float): The threshold value to switch between L1 and L2 loss.
        """
        super(CustomL1L2Loss, self).__init__()
        self.a = a

    def forward(self, input, target):
        """
        Computes the custom L1/L2 loss.
        Args:
            input (torch.Tensor): Predicted tensor of shape (batch, ...).
            target (torch.Tensor): Ground truth tensor of the same shape as input.
        Returns:
            torch.Tensor: Scalar loss value.
        """
        diff = input - target
        abs_diff = torch.abs(diff)
        
        # L1 loss for |x| < a
        l1_loss = abs_diff[abs_diff < self.a]
        
        # L2 loss for |x| >= a
        l2_loss = diff[abs_diff >= self.a] ** 2
        
        # Combine the losses
        loss = l1_loss.sum() + l2_loss.sum()
        return loss / input.numel()  # Normalize by the total number of elements
    
def compute_principal_components_batched(images, num_components=None):
    """
    Compute the principal components of a batch of images using PCA.

    Args:
        images: Tensor of shape (B, H, W) representing a batch of grayscale images.
        num_components: Number of principal components to retain. If None, retain all.

    Returns:
        components: Principal components as a batch of matrices (B, W, num_components or W).
        explained_variance: Variance explained by each component (B, num_components or W).
    """
    # Step 1: Flatten each image in the batch
    B, H, W = images.shape
    flattened_images = images.view(B, H, W)  # Shape: (B, H, W)

    # Step 2: Center the data
    mean = flattened_images.mean(dim=1, keepdim=True)  # Mean along the height (H) axis
    centered_data = flattened_images - mean

    # Step 3: Compute the covariance matrix for each image
    covariance_matrices = torch.matmul(
        centered_data.transpose(1, 2), centered_data
    ) / (H - 1)  # Shape: (B, W, W)

    # Step 4: Compute eigenvalues and eigenvectors for each covariance matrix
    eigenvalues_list = []
    eigenvectors_list = []
    for cov_matrix in covariance_matrices:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # Symmetric matrices
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)

    eigenvalues = torch.stack(eigenvalues_list, dim=0)  # Shape: (B, W)
    eigenvectors = torch.stack(eigenvectors_list, dim=0)  # Shape: (B, W, W)

    # Step 5: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True, dim=1)
    eigenvalues = torch.gather(
        eigenvalues, 1, sorted_indices
    )  # Shape: (B, W)
    eigenvectors = torch.gather(
        eigenvectors, 2, sorted_indices.unsqueeze(1).expand(-1, W, -1)
    )  # Shape: (B, W, W)

    # Step 6: Select the top components
    if num_components is not None:
        eigenvalues = eigenvalues[:, :num_components]  # Shape: (B, num_components)
        eigenvectors = eigenvectors[:, :, :num_components]  # Shape: (B, W, num_components)

    return eigenvectors, eigenvalues

def reconstruct_images_batched(images, components, first_component, num_components):
    """
    Reconstruct a batch of images using selected principal components.

    Args:
        images: Tensor of shape (B, H, W) representing a batch of grayscale images.
        components: Tensor of shape (B, W, C) where C is the number of components for each image.
        first_component: The starting principal component (1-based index).
        num_components: Number of components to use for reconstruction.

    Returns:
        reconstruction: Tensor of shape (B, H, W) representing the reconstructed images.
    """
    B, H, W = images.shape
    selected_components = components[:, :, first_component - 1:num_components]  # Shape: (B, W, selected_components)

    # Step 1: Center the images
    mean = images.mean(dim=1, keepdim=True)  # Compute mean along the height dimension (H)
    centered_data = images - mean  # Shape: (B, H, W)

    # Step 2: Project centered data onto the selected components
    projection = torch.matmul(centered_data, selected_components)  # Shape: (B, H, selected_components)

    # Step 3: Reconstruct the images from the projection
    reconstruction = torch.matmul(projection, selected_components.transpose(1, 2)) + mean  # Shape: (B, H, W)

    return reconstruction



class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).

    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        #self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.l1_loss(input_feats, target_feats, reduction=self.reduction)
    
class DifferentiableMutualInformationLoss:
    def __init__(self, sigma=0.2, epsilon=1e-10):
        """
        Initializes the differentiable mutual information loss.

        Args:
            sigma: Bandwidth of the Gaussian kernel.
            epsilon: Small constant to avoid log(0).
        """
        self.sigma = sigma
        self.epsilon = epsilon

    def gaussian_kernel(self, x1, x2):
        """
        Compute the Gaussian kernel between two sets of points.
        
        Args:
            x1: Tensor of shape (N, d).
            x2: Tensor of shape (M, d).

        Returns:
            Kernel matrix of shape (N, M).
        """
        diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # Shape: (N, M, d)
        distances = torch.sum(diff ** 2, dim=-1)  # Shape: (N, M)
        return torch.exp(-distances / (2 * self.sigma ** 2))

    def compute_joint_probabilities(self, X, Y):
        """
        Compute joint probabilities using a Gaussian kernel.
        
        Args:
            X: Tensor of shape (N, d1).
            Y: Tensor of shape (N, d2).

        Returns:
            Joint probability matrix (P(X, Y)).
        """
        joint_features = torch.cat([X, Y], dim=-1)  # Shape: (N, d1 + d2)
        kernel_matrix = self.gaussian_kernel(joint_features, joint_features)
        joint_probabilities = kernel_matrix / kernel_matrix.sum()  # Normalize
        return joint_probabilities

    def compute_marginal_probabilities(self, X):
        """
        Compute marginal probabilities using a Gaussian kernel.
        
        Args:
            X: Tensor of shape (N, d).

        Returns:
            Marginal probability matrix (P(X)).
        """
        kernel_matrix = self.gaussian_kernel(X, X)
        marginal_probabilities = kernel_matrix / kernel_matrix.sum()  # Normalize
        return marginal_probabilities

    def __call__(self, X, Y):
        """
        Compute the differentiable mutual information loss.
        
        Args:
            X: Tensor of shape (N, d1).
            Y: Tensor of shape (N, d2).

        Returns:
            Negative mutual information loss (scalar).
        """
        # Compute joint and marginal probabilities
        joint_probabilities = self.compute_joint_probabilities(X, Y) + self.epsilon
        marginal_prob_X = self.compute_marginal_probabilities(X) + self.epsilon
        marginal_prob_Y = self.compute_marginal_probabilities(Y) + self.epsilon

        # Compute mutual information
        joint_entropy = -torch.sum(joint_probabilities * torch.log(joint_probabilities))
        marginal_entropy_X = -torch.sum(marginal_prob_X * torch.log(marginal_prob_X))
        marginal_entropy_Y = -torch.sum(marginal_prob_Y * torch.log(marginal_prob_Y))

        mutual_information = marginal_entropy_X + marginal_entropy_Y - joint_entropy

        # Return negative mutual information as loss
        return -mutual_information

class ResNet18Backbone(nn.Module):
    def __init__(self):
        super(ResNet18Backbone, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
        )

    def forward(self, x):
        return self.feature_extractor(x)
    

class FeatureEmbedding(nn.Module):
    def __init__(self, backbone, output_dim=128):
        super(FeatureEmbedding, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, output_dim)  # Adjust 512 if needed based on feature size

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        embedding = self.fc(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)  # L2 normalization
        return embedding

def contrastive_loss(embedding, memory_bank):
    positive = embedding.dot(memory_bank[0])  # Positive sample (or query itself)
    negatives = embedding.dot(memory_bank[1:])  # Negatives
    loss = -torch.log(torch.exp(positive) / (torch.exp(positive) + torch.sum(torch.exp(negatives))))
    return loss



class VGGPerceptualLoss(nn.Module):
    DEFAULT_FEATURE_LAYERS = [];#[0,1,2,3]#[0, 1, 2, 3]
    IMAGENET_RESIZE = (224, 224)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGENET_SHAPE = (1, 3, 1, 1)

    def __init__(self, resize=True, feature_layers=None, style_layers=None):
        super().__init__()
        self.resize = resize
        self.feature_layers = feature_layers or self.DEFAULT_FEATURE_LAYERS
        self.style_layers = style_layers or [1,2,3]  
        self.forb_factor = 20

        features = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            features[:4].eval(),
            features[4:9].eval(),
            features[9:16].eval(),
            features[16:23].eval(),
        ])
        #for param in self.parameters():
            #param.requires_grad = False
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(self.IMAGENET_SHAPE))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(self.IMAGENET_SHAPE))
        self.weights = [0, 1/100000, 1/5000, 1/100]  # [1,1,1,1] #[0, 0, 1/1000, 1/40] - good res

        # for Sairam new loss: [0, 1/100000, 1/5000, 1/100] 
        self.loss_func = CustomL1L2Loss(3)
        # [0, 0, 1/1000, 1/40] = good res 
        # [0, 1/1000, 1/5000, 1/100] - tests 2 - great res - no feature loss

    def _transform(self, tensor):
        if tensor.shape != self.IMAGENET_SHAPE:
            tensor = tensor.repeat(self.IMAGENET_SHAPE)
        """
        print(tensor.shape)
        print(self.mean.shape)
        print(self.std.shape)
        """
        tensor = (tensor - self.mean) / self.std
        if self.resize:
            tensor = nn.functional.interpolate(tensor, mode='bilinear', size=self.IMAGENET_RESIZE, align_corners=False)
        return tensor

    def _calculate_gram(self, tensor):
        act = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        return act @ act.permute(0, 2, 1)
    
    def _calculate_gram_new(self, tensor):
        act = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        return act @ act.permute(0, 2, 1)
    

    def forward(self, output, target):
        """
        output = output.squeeze(1)
        components, explained_variance = compute_principal_components_batched(output,num_components=13)
        output = reconstruct_images_batched(output, components,first_component=3, num_components=13)
        output = output.unsqueeze(1)
        target = target.squeeze(1)
        components, explained_variance = compute_principal_components_batched(target,num_components=13)
        target = reconstruct_images_batched(target, components,first_component=3, num_components=13)
        target = target.unsqueeze(1)
        """
        output, target = self._transform(output), self._transform(target)
        loss = 0.
        for i, block in enumerate(self.blocks):
            output, target = block(output), block(target)
            
            if i in self.feature_layers:
                ## l1 loss 
                loss += nn.functional.l1_loss(output, target)
                """
                ## frobenius loss
                diff = output - target
                loss += torch.norm(diff, p='fro') / diff[0].numel()
                """

            if i in self.style_layers:
                gram_output, gram_target = self._calculate_gram(output), self._calculate_gram(target)
                ## l1 loss 
                loss += nn.functional.l1_loss(gram_output, gram_target) *self.weights[i]
                ## frobenius loss
                """
                gram_diff = gram_output - gram_target
                loss += self.forb_factor*(torch.norm(gram_diff, p='fro') / gram_diff[0].numel()) * self.weights[i]
                """

                
     
        return loss
    

