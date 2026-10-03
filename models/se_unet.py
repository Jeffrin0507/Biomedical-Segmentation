import torch
import torch.nn as nn


# =========================================================
# Double Convolution Block
# =========================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        return self.block(x)


# =========================================================
# Squeeze-and-Excitation Block
# =========================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(
                channels,
                channels // reduction,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                channels // reduction,
                channels,
                bias=False
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        batch_size, channels, _, _ = x.size()

        # -------------------------
        # Squeeze
        # -------------------------

        y = self.global_pool(x)

        y = y.view(
            batch_size,
            channels
        )


        # -------------------------
        # Excitation
        # -------------------------

        y = self.fc(y)

        y = y.view(
            batch_size,
            channels,
            1,
            1
        )


        # -------------------------
        # Recalibration
        # -------------------------

        return x * y


# =========================================================
# SE U-Net
# =========================================================

class SEUNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1
    ):

        super().__init__()


        # =================================================
        # Encoder
        # =================================================

        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.se1 = SEBlock(64)


        self.pool1 = nn.MaxPool2d(
            kernel_size=2
        )


        self.enc2 = DoubleConv(
            64,
            128
        )

        self.se2 = SEBlock(128)


        self.pool2 = nn.MaxPool2d(
            kernel_size=2
        )


        self.enc3 = DoubleConv(
            128,
            256
        )

        self.se3 = SEBlock(256)


        self.pool3 = nn.MaxPool2d(
            kernel_size=2
        )


        self.enc4 = DoubleConv(
            256,
            512
        )

        self.se4 = SEBlock(512)


        self.pool4 = nn.MaxPool2d(
            kernel_size=2
        )


        # =================================================
        # Bottleneck
        # =================================================

        self.bottleneck = DoubleConv(
            512,
            1024
        )


        # =================================================
        # Decoder
        # =================================================

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )


        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )


        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )


        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )


        # =================================================
        # Output
        # =================================================

        self.output = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )


    def forward(self, x):

        # -------------------------
        # Encoder
        # -------------------------

        e1 = self.enc1(x)
        e1 = self.se1(e1)

        p1 = self.pool1(e1)


        e2 = self.enc2(p1)
        e2 = self.se2(e2)

        p2 = self.pool2(e2)


        e3 = self.enc3(p2)
        e3 = self.se3(e3)

        p3 = self.pool3(e3)


        e4 = self.enc4(p3)
        e4 = self.se4(e4)

        p4 = self.pool4(e4)


        # -------------------------
        # Bottleneck
        # -------------------------

        b = self.bottleneck(p4)


        # -------------------------
        # Decoder
        # -------------------------

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)


        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)


        # -------------------------
        # Output
        # -------------------------

        return self.output(d1)


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    model = SEUNet()

    x = torch.randn(
        2,
        3,
        256,
        256
    )

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        parameters
    )