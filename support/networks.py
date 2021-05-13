import torch
import torch.nn as nn
import sys
sys.path.insert(1, '/root/sbmc/')
from sbmc import modules as ops

class PathNet(nn.Module):
    """Path embedding network
    """

    def __init__(self, ic, intermc=64, outc=3):
        super(PathNet, self).__init__()
        self.ic = ic
        self.intermc = intermc
        self.outc = outc
        self.final_ic = intermc + intermc

        self.embedding = ops.ConvChain(ic, intermc, width=intermc, depth=3,
                ksize=1, pad=False)
        self.propagation = ops.Autoencoder(intermc, intermc, num_levels=3, 
                increase_factor=2.0, num_convs=3, width=intermc, ksize=3,
                output_type="leaky_relu", pooling="max")
        self.final = ops.ConvChain(self.final_ic, outc, width=self.final_ic, 
                depth=2, ksize=1, pad=False, output_type="relu")

    def __str__(self):
        return "PathNet i{}in{}o{}".format(self.ic, self.intermc, self.outc)

    def forward(self, samples):
        paths = samples["paths"]
        bs, spp, nf, h, w = paths.shape

        flat = paths.contiguous().view([bs*spp, nf, h, w])
        flat = self.embedding(flat)
        flat = flat.view([bs, spp, self.intermc, h, w])
        reduced = flat.mean(1)

        propagated = self.propagation(reduced)
        flat = torch.cat([flat.view([bs*spp, self.intermc, h, w]), propagated.unsqueeze(1).repeat(
            [1, spp, 1, 1, 1]).view(bs*spp, self.intermc, h, w)], 1)
        out = self.final(flat).view([bs, spp, self.outc, h, w])
        return out
