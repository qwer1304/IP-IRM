import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import torch
import torch.nn as nn

from typing import Union, List

from functools import partial

class Mask():
    def __init__(self, mask_type, tau=1.0, soft=False, K=None, hard_K=False):
        self.mask_type = mask_type
        self.tau = tau
        self.soft = soft
        self.K = K
        self.hard_K = hard_K
        self.u = None

    def __call__(self, x, u=None, training=True):
        # x: (num_features,) tensor
        if self.mask_type == 'sigmoid':
            mask = torch.sigmoid(x / self.tau)
            #mask = torch.where(mask < 0.1, torch.zeros_like(mask), mask)
            return mask
        elif self.mask_type == 'ident':
            return x
        elif self.mask_type == 'gumbel':
            # Sample Gumbel noise
            if training:
                if u is None: u = torch.rand_like(x)
                self.u = u
            else: # use stored u unless it has never been stored
                if self.u is None:
                    if u is None: u = torch.rand_like(x)
                else:
                    u = self.u
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            x_soft = torch.sigmoid((x + g) / self.tau)  # (N,)

            if self.soft:
                x_ret = x_soft
            else:
                # Hard straight-through: forward 0/1, backward gradient through soft

                # Calculate threshold for the Top-K slots
                if self.hard_K:
                    _, topk_indices = torch.topk(x_soft, self.K)
                    topk_mask = torch.zeros_like(x, dtype=bool)
                    topk_mask[topk_indices] = True
                else:
                    topk_mask = torch.ones_like(x, dtype=bool)

                # 1. We never exceed K (because of threshold)
                # 2. We don't force 'on' channels that are naturally 'off' (because of 0.5)
                x_hard = ((x_soft > 0.5) & topk_mask).float()
                x_ret = x_hard + x_soft - x_soft.detach()
            return x_ret
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

class MaskModule(nn.Module):
    def __init__(self, activation_method, input_dim, trainable=True, device='cuda'):
        super().__init__()
        # Initialize the mask as a trainable parameter
        if trainable:
            init_logit = torch.rand(input_dim) # default value
            if activation_method.K:
                if activation_method.mask_type == 'gumbel' and not activation_method.soft:
                        target_p = activation_method.K / input_dim  # e.g., 256 / 2048
                        init_logit = torch.log(torch.tensor(target_p / (1 - target_p)))
                elif activation_method.mask_type == 'sigmoid' or activation_method.mask_type == 'gumbel':
                    def get_bounds(K, N=2048, W=2):
                        # The Logit of the probability
                        b = torch.log(torch.tensor(K / (N - K)))
                        L = b - (W / 2)
                        H = b + (W / 2)
                        return L, H

                    def init_mask_to_neff(n=2048, target_k=100):
                        # 1. Start with random noise in [-1, 1]
                        r = (torch.rand(n) * 2) - 1

                        # 2. We need to find the bias 'b' that shifts the distribution 
                        # so that the Sigmoid output hits the target Neff
                        def get_neff(bias):
                            mask = torch.sigmoid(r + bias)
                            return (mask.sum()**2 / torch.sum(mask**2))

                        # Binary search for the correct bias
                        low, high = get_bounds(K=target_k, N=n)
                        for _ in range(20):  # 20 iterations is enough for high precision
                            mid = (low + high) / 2
                            if get_neff(mid) < target_k:
                                high = mid # Inversely related: higher bias = higher Neff
                            else:
                                low = mid

                        # 3. Apply the found bias
                        final_bias = (low + high) / 2
                        return r + final_bias

                    init_logit = init_mask_to_neff(n=input_dim, target_k=activation_method.K)
                #end if activation_method.mask_type == 'gumbel' and not activation_method.gumbel_soft:
            # end if activation_method.K:

            # Initialize with a small variance around the target logit
            init_val = init_logit + (torch.randn(input_dim) * 0.01)
            self.mask = nn.Parameter(init_val) # no need to set device, since it will be placed at the correct device with the rest of the
        else:
            self.mask = torch.ones(input_dim, device=device)
        # end if trainable:
        self.activation_method = activation_method

    def forward(self, x, **kwargs):
        # Simply multiply the features by the mask
        return self.activation(**kwargs).to(x.device) * x

    def activation(self, **kwargs):
        return self.activation_method(self.mask, **kwargs)
        
    def sample(self):
        return torch.rand_like(self.mask)


def create_mlp(
    input_dim: int,
    output_dim: int,
    *, # all pars that follow must be named
    hidden_dims: List[int] = None,
    activation_layer: type[nn.Module] = nn.ReLU,
    norm_layer: type[nn.Module] = None,
    norm_kwargs: Union[None, dict, List[dict]] = None, 
    dropout: float = 0.0,
    bias: Union[bool, List[bool]] = True,
    last_layer_norm: bool = False,
    last_layer_act: bool = False
) -> nn.Sequential:
    """
    Returns an n-hidden layer MLP module.
    
    Args:
        bias: 
            - True: All layers have bias.
            - False/None: No layers have bias.
            - List[bool]: Individual bias setting for each linear layer.
    """
    hidden_dims = hidden_dims or []
    all_dims = [input_dim] + hidden_dims + [output_dim]
    num_linear_layers = len(all_dims) - 1
    
    # Helper to ensure we have a list of config dicts
    if isinstance(norm_kwargs, dict) or norm_kwargs is None:
        # Turn single dict into a list of identical dicts
        norm_params = [norm_kwargs or {} for _ in range(num_linear_layers)]
    else:
        # User provided a list [{}, {}, {}]
        norm_params = norm_kwargs
    
    # Resolve bias into a list of booleans (one per linear layer)
    if isinstance(bias, list):
        if len(bias) != num_linear_layers:
            raise ValueError(f"Bias list length ({len(bias)}) must match "
                             f"num_layers ({num_linear_layers})")
        bias_list = bias
    else:
        # Broadcast single bool/None to all layers
        bias_val = bool(bias) if bias is not None else False
        bias_list = [bias_val] * num_linear_layers

    layers = []
    for i in range(num_linear_layers):
        is_last = (i == num_linear_layers - 1)
        in_dim, out_dim = all_dims[i], all_dims[i+1]
        
        # 1. Linear Layer with specific bias
        layers.append(nn.Linear(in_dim, out_dim, bias=bias_list[i]))
        
        # 2. Normalization
        if norm_layer and (not is_last or last_layer_norm):
            layers.append(norm_layer(out_dim, **norm_params[i]))
            
        # 3. Activation
        if not is_last or last_layer_act:
            layers.append(activation_layer())
            
        # 4. Dropout
        if dropout > 0 and not is_last:
            layers.append(nn.Dropout(dropout))
            
    return nn.Sequential(*layers)
    
"""
A multi-arm model is comprised of:
    - backbone (e.g., Resnet50)
    - mask layer (optional), stateless; can be learnable or fixed
    - multiple "arm" layers which correspond to different network processing required for specific losses
'aliases' ({alias: target} dict) are optional alternative names for the different components s.t. they can be invoked through net.alias(...)
'blueprints' are partial model creation functions s.t. the caller instantiates all the parameters except for those that come from within
'out_transforms' are transformations of module's outputs (e.g., normalization)
'in_transform' is a transformation (e.g., augmentation) applied to the input BEFORE applying the backbone
the model (e.g., feature dimension) which the callee instantiates
It is the caller's resposibility to invoke different elements in the correct order and to pass parameters between them appropriately
"""
class MultiArmModel(nn.Module):
    def __init__(self, backbone_name='resnet50', mask_blueprint=None, arms_blueprints={}, in_transform=None, out_transforms=None, 
                 shortcuts=None, image_class='ImageNet', state_dict=None):
        super().__init__()
        self.in_transform = in_transform

        # 1. Internal Backbone logic
        if backbone_name == 'resnet50':
            self.f = resnet50(weights=None)
        else:
            raise ValueError(f"Unknown backbone name: {backbone_name}")

        # Modify input layers for CIFAR/STL if needed
        if image_class != 'ImageNet':
            self.f.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.f.maxpool = nn.Identity()

        # Remove final classification head (fc)
        self.feature_dim = self.f.fc.in_features
        self.f.fc = nn.Identity()

        self._load_backbone(state_dict) # updates self.f
        
        # 2. Mask layer
        if mask_blueprint:
            self.mask_fun = mask_blueprint(self.feature_dim)
        else:    
            self.mask_fun = MaskModule(Mask('ident'), self.feature_dim, trainable=False)
            
        # 3. Arms
        self.arms = nn.ModuleDict()
        self.out_transforms = {}
        
        self.add_arms(arms_blueprints=arms_blueprints, out_transforms=out_transforms, shortcuts=shortcuts)
        
    def _load_backbone(self, state_dict):
        # Load pretrained weights (if provided)
        if state_dict is not None:
            # Handle MoCo checkpoints (strip encoder_q prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module.encoder_q."):
                    k = k[len("module.encoder_q."):]
                new_state_dict[k] = v

            msg = self.f.load_state_dict(new_state_dict, strict=False)

            # Don't care about fc layer from pretrained
            print("\tMissing keys (ignoring fc):", [k for k in msg.missing_keys if not k.startswith("fc.")])
            print("\tUnexpected keys (ignoring fc):", [k for k in msg.unexpected_keys if not k.startswith("fc.")])

    def add_arms(self, arms_blueprints, out_transforms=None, shortcuts=None):
        """
        Takes names and partial() blueprints in a dict, transforms and shortcuts to them.
        Instantiates the arms using the system's known feature_dim.
        """
        common_keys = arms_blueprints.keys() & self.arms.keys()
        if common_keys:
            raise ValueError(f"The following arm names overlap: {common_keys}.")

        #  1. Register arms (modules)
        for name, module in arms_blueprints.items():
            # Check if 'input_dim' was already set in the partial blueprint
            if isinstance(module, partial) and 'input_dim' in module.keywords:
                # It's already set (e.g. predictor), so call it without arguments
                self.arms[name] = module()
            else:
                # It's missing (e.g. projector), so provide the default
                self.arms[name] = module(input_dim=self.feature_dim)

        # 2. Register post-processing (e.g., normalization)
        if out_transforms:
            self.out_transforms.update(out_transforms)

        # 3. Aliases
        if shortcuts:
            for alias, target_name in shortcuts.items():
                # Check for attribute collisions on 'self'
                if hasattr(self, alias):
                    raise AttributeError(f"Cannot create shortcut '{alias}'; attribute already exists.")
                    
                if target_name not in arms_blueprints:
                    raise AttributeError(f"Cannot alias '{alias}' to non-existent arm '{target_name}'")

                # Point the shortcut to a wrapper that calls the main arm() method
                setattr(self, alias, partial(self.arm, target_name))

    def forward(self, x):
        raise NotImplementedError

    def backbone(self, x):
        return self.f(self.in_transform(x)) if self.in_transform else self.f(x)
        
    def mask(self, x):
        return self.mask_fun(x)
        
    def arm(self, name, x, **kwargs):
        """The single entry point for all arm logic."""
        assert name in self.arms, f"arm {name} not in arm names {self.arms.keys()}"
        # 1. Apply the module (e.g., MLP)
        out = self.arms[name](x)

        # 2. Apply transform, if it exists, for this arm name
        if name in self.out_transforms:
            out = self.out_transforms[name](out, **kwargs)

        return out

# ================================================

class ModelResnet(nn.Module):
    def __init__(self, feature_dim=128, image_class='ImageNet', state_dict=None):
        super().__init__()

        # Backbone
        self.f = resnet50(weights=None)

        # Modify input layers for CIFAR/STL if needed
        if image_class != 'ImageNet':
            self.f.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.f.maxpool = nn.Identity()

        # Remove final classification head (fc)
        dim_mlp = self.f.fc.in_features
        self.f.fc = nn.Identity()

        # Projection head for SSL
        self.g = nn.Sequential(
            nn.Linear(dim_mlp, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )
        
        # Load pretrained weights (if provided)
        if state_dict is not None:
            # Handle MoCo checkpoints (strip encoder_q prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module.encoder_q."):
                    k = k[len("module.encoder_q."):]
                new_state_dict[k] = v

            msg = self.f.load_state_dict(new_state_dict, strict=False)

            # Don't care about fc layer from pretrained
            print("\tMissing keys (ignoring fc):", [k for k in msg.missing_keys if not k.startswith("fc.")])
            print("\tUnexpected keys (ignoring fc):", [k for k in msg.unexpected_keys if not k.startswith("fc.")])

    def forward(self, x):
        # Extract backbone features
        feature = self.f(x)                      # [N, 2048] after avgpool & flatten
        out = self.g(feature)                    # projection head

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Model(nn.Module):
    def __init__(self, feature_dim=128, image_class='ImageNet', state_dict=None):
        super(Model, self).__init__()

        self.f = []
        res50 = resnet50(weights=None) 
        if state_dict is not None:
            msg = res50.load_state_dict(state_dict, strict=False)
            print(msg)

        for name, module in res50.named_children():
            if image_class != 'ImageNet':  # STL, CIFAR
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            else:
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        """ page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        )
        self.num_layers = 3
    
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        """
        page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h's input and output (z and p) is d = 2048, 
        and h's hidden layer's dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x, normalize=True):
        x = self.layer1(x)
        x = self.layer2(x)
        if normalize:
            x = F.normalize(x, dim=-1)
        return x 

class SimSiam(nn.Module):
    def __init__(self, feature_dim=128, image_class='ImageNet', state_dict=None):
        super().__init__()

        # Backbone
        self.f = resnet50(weights=None)

        # Modify input layers for CIFAR/STL if needed
        if image_class != 'ImageNet':
            self.f.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.f.maxpool = nn.Identity()

        # Remove final classification head (fc)
        dim_mlp = self.f.fc.in_features
        self.f.fc = nn.Identity()

        # Load pretrained weights (if provided)
        if state_dict is not None:
            # Handle MoCo / SimSiam checkpoints (strip encoder_q prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                if k.startswith("encoder_q."):
                    k = k[len("encoder_q."):]
                if k.startswith("encoder."):
                    k = k[len("encoder."):]
                if k.startswith("predictor."):
                    continue
                if k.startswith("fc."):
                    continue
                new_state_dict[k] = v

            msg = self.f.load_state_dict(new_state_dict, strict=False)

            # Don't care about fc layer from pretrained
            print("\tMissing keys (ignoring fc):", [k for k in msg.missing_keys if not k.startswith("fc.")])
            print("\tUnexpected keys (ignoring fc):", [k for k in msg.unexpected_keys if not k.startswith("fc.")])


        self.projector = projection_MLP(dim_mlp, hidden_dim=512, out_dim=feature_dim)

        self.predictor = prediction_MLP(in_dim=feature_dim, hidden_dim=int(feature_dim/2), out_dim=feature_dim)
   
    def forward(self, x, normalize=True):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        
        f = self.projector
        z = f(feature)
        if normalize:
            feature = F.normalize(feature, dim=-1)
            z       = F.normalize(z, dim=-1)
        return feature, z
        
