#!/usr/bin/env python3


from torch import nn

class CBlock(nn.Module):
    """
        k: kernel
        s: stride
        p: padding
        kwargs: other parameters
    """

    def __init__(self, inc, out, k, s, p, **kwargs) -> None:
        super(CBlock, self).__init__()
        self.conv = nn.Conv2d(inc, out, k, s, p, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class CLRBlock(nn.Module):
    """
        k: kernel
        s: stride
        p: padding
        kwargs: other parameters
    """
    def __init__(self, inc, out, k, s, p, **kwargs) -> None:
        super().__init__(inc, out, k, s, p, **kwargs)
        self.conv = nn.Conv2d(inc, out, k, s, p, **kwargs)
        self.ln = nn.LayerNorm(out)
        self.relu = nn.ReLU(inplace=True)
        #self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        x = self.relu(x)
        return x
    
class CBR6Block(nn.Module):
    """
        k: kernel
        s: stride
        p: padding

        Compared with CBRBlock, this block use relu6 activation function.
    """
    def __init__(self, inc, out, k, s, p, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(inc, out, k, s, p, **kwargs)
        self.bn = nn.BatchNorm2d(out)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        return x

class CBR6DWBlock(nn.Module):
    """
        separable convolution block
        dw (depth-wise) conv  and pw (point-wise) conv
    """
    def __init__(self,inc, outc, kernel_size = 3, stride=1, padding=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dw = [
            nn.Conv2d(inc, inc, kernel_size, stride, padding, groups=inc),
            nn.BatchNorm2d(inc),
            nn.ReLU2d(inplace=True),
        ]

        
        # kernel: 1, 
        # stride: 1,
        # padding : 0
        self.pw = [
            nn.Conv2d(inc, outc, 1, 1, 0),
            nn.BatchNorm2d(outc),
            nn.ReLU6(inplace=True),
        ] 
    
    def forward(self, x):
        for layer in self.dw:
            x = layer(x)
        for layer in self.pw:
            x = layer(x)
        return x
    

class CBRBlock(nn.Module):
    """
        k: kernel
        s: stride
        p: padding
        kwargs: other parameters
    """
    def __init__(self, inc, out, k, s, p, **kwargs) -> None:
        super(CBRBlock, self).__init__()
        self.conv = nn.Conv2d(inc, out, k, s, p, **kwargs)
        self.bn = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class CRBlock(nn.Module):
    """
        k: kernel
        s: stride
        p: padding
        kwargs: other parameters
    """
    def __init__(self, inc, out, k, s, p, **kwargs) -> None:
        super(CRBlock, self).__init__()
        self.conv = nn.Conv2d(inc, out, k, s, p, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x