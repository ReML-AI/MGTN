import torchvision.models as models
from util import *
from gnn import *
import torch
import torch.nn as nn

class MGTNResnet(nn.Module):
    def __init__(self, model_name, num_classes, emb_features=300, t1=0.0, t2=0.0, adj_file=None, mod_file=None, ml_threshold=0.999):
        super(MGTNResnet, self).__init__()

        _mods = np.loadtxt(mod_file, dtype=int)
        
        self.backbones = nn.ModuleList()
        # Create multiple backbones
        for i in range(int(max(_mods)) - int(min(_mods)) + 1):
            model = load_model(model_name)
            backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.MaxPool2d(14, 14),
            nn.Flatten(1)
            )
            self.backbones.append(backbone)

        self.num_classes = num_classes

        # Graph Convolutions
        self.gc1 = GConv(emb_features, 2048)
        self.gc2 = GConv(2048, 4096)
        self.relu = nn.LeakyReLU(0.2)

        # Topology
        self.A = torch.stack([
            torch.eye(num_classes).type(torch.FloatTensor),
            torch.from_numpy(AdjacencyHelper.gen_A(num_classes, 1.0, t1, adj_file)).type(torch.FloatTensor),
            torch.from_numpy(AdjacencyHelper.gen_A(num_classes, t1, t2, adj_file)).type(torch.FloatTensor)
        ]).unsqueeze(0)

        self.gtn = GTLayer(self.A.shape[1], 1, first=True)
        self.mods = nn.Parameter(torch.from_numpy(AdjacencyHelper.gen_M(_mods, dims=2048, t=ml_threshold)).float())

    def forward(self, img, emb):
        fs = []
        for i in range(len(self.backbones)):
            fs.append(self.backbones[i](img))
        f = torch.cat(fs, 1)
        
        adj, _ = self.gtn.forward(self.A)
        adj = torch.squeeze(adj, 0) + torch.eye(self.num_classes).type(torch.FloatTensor).cuda()
        adj = AdjacencyHelper.gen_adj(adj)

        w = self.gc1(emb[0], adj)
        w = self.relu(w)
        w = self.gc2(w, adj)
        w = torch.mul(w, self.mods)

        w = w.transpose(0, 1)
        y = torch.matmul(f, w)
        return y

    def get_config_optim(self, lr, lrp):
        config_optim = []
        for backbone in self.backbones:
            config_optim.append({'params': backbone.parameters(), 'lr': lr * lrp})
        config_optim.append({'params': self.gc1.parameters(), 'lr': lr})
        config_optim.append({'params': self.gc2.parameters(), 'lr': lr})
        return config_optim

def mgtn_resnet(num_classes, t1, t2, pretrained=True, adj_file=None, mod_file=None, emb_features=300, ml_threshold=0.999):
    return MGTNResnet('resnext50_32x4d_swsl', num_classes, t1=t1, t2=t2, adj_file=adj_file, mod_file=mod_file, emb_features=emb_features, ml_threshold=ml_threshold)
