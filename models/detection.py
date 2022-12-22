from torch import nn


def cnnLayer(in_filters, out_filters, kernel_size=3):
    padding = kernel_size//2
    return nn.Sequential(
      nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding),
      nn.BatchNorm2d(out_filters),
      nn.LeakyReLU()
    )

def create_backbone(C=1, classes=1, n_filters=32):
    backbone = nn.Sequential(
            cnnLayer(C, n_filters),
            cnnLayer(n_filters, n_filters),
            cnnLayer(n_filters, n_filters),
            nn.MaxPool2d((2,2)),
            cnnLayer(n_filters, 2*n_filters),
            cnnLayer(2*n_filters, 2*n_filters),
            cnnLayer(2*n_filters, 2*n_filters),
            nn.MaxPool2d((2,2)),
            cnnLayer(2*n_filters, 4*n_filters),
            cnnLayer(4*n_filters, 4*n_filters),
    )
    backbone.out_channels = 4*n_filters
    return backbone