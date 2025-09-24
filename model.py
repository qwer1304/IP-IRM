import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

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
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
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
        return F.normalize(x, dim=-1) 


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
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return F.normalize(x, dim=-1) 

class SimSiam(nn.Module):
    def __init__(self, feature_dim=128, image_class='ImageNet', state_dict=None):
        super(SimSiam, self).__init__()

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

        self.f = nn.Sequential(*self.f) # backbone

        self.projector = projection_MLP(2048, hidden_dim=512, out_dim=feature_dim)

        self.predictor = prediction_MLP(in_dim=feature_dim, hidden_dim=int(feature_dim/2), out_dim=feature_dim)
    
    def forward(self, x):

        f = self.projector

        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        
        z = f(feature)

        return F.normalize(feature, dim=-1), F.normalize(z, dim=-1)
        
