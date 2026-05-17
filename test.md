這就為您準備這三種常見且高效的注意力機制 PyTorch 程式碼。

為了讓您能順利將這些模組無縫接軌到 YOLO（如 Ultralytics 的 YOLOv8/v11）的 block.py 或 modules.py 中，我將輸入通道數的參數統一命名為 c1（這是 YOLO 源碼中代表 in_channels 的慣用命名方式），並確保它們都不會改變特徵圖的尺寸和通道數（即輸入等於輸出），實現真正的「隨插即用」。

1. ECA-Net (Efficient Channel Attention)
ECA-Net 的程式碼最為簡潔。它根據輸入通道數自適應地計算 1D 卷積的核大小（Kernel Size），無需像 SE Block 那樣進行全連接層的降維，非常輕量。

Python
import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    """Efficient Channel Attention module."""
    def __init__(self, c1, b=1, gamma=2):
        super(ECA, self).__init__()
        # 根據通道數自適應計算 1D 卷積的 kernel size
        t = int(abs((math.log(c1, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, H, W]
        y = self.avg_pool(x) # shape: [B, C, 1, 1]
        # 為了使用 Conv1d，需進行維度轉換: [B, C, 1, 1] -> [B, 1, C]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # 轉換回原維度: [B, 1, C] -> [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)
2. CBAM (Convolutional Block Attention Module)
CBAM 包含了通道注意力（Channel Attention）和空間注意力（Spatial Attention）兩個子模組。它會先在通道維度上計算注意力，接著在空間維度上計算，是一種雙重注意力機制。

Python
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, c1, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 使用 1x1 卷積代替全連接層以適應各種輸入尺寸
        self.fc1 = nn.Conv2d(c1, c1 // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(c1 // ratio, c1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道維度進行平均與最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(c1, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
3. CoordAtt (Coordinate Attention)
CoordAtt（第一篇論文使用的機制）將位置資訊嵌入到通道注意力中。它將 2D 的全局池化分解為兩個 1D 的方向池化（水平與垂直），對於捕捉微小物件的邊緣與特徵特別有效。

Python
import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    """Coordinate Attention Module."""
    def __init__(self, c1, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 獲取水平和垂直方向的特徵
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # 拼接並壓縮通道
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分割回原本的高與寬度
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 產生兩個方向的注意力權重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_w * a_h
放入 YOLO 源碼後的下一步：
將這些代碼貼入 block.py 後，記得去 tasks.py 裡的 parse_model 函數中，將 ECA, CBAM, CoordAtt 加進 eval() 可解析的模組列表中，這樣您就可以在 .yaml 檔案裡直接呼叫它們了。
