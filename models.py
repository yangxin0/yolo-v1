from config import YOLOv1ModelConfig
import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        self.S = YOLOv1ModelConfig.S
        self.B = YOLOv1ModelConfig.B
        self.C = YOLOv1ModelConfig.C
        self.depth = self.B * 2 + self.C

        layers = [
            # 1st part
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd part
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3rd part
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # 4th part
        for _ in range(0, 4):
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            ]
        layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # 5th part
        for _ in range(0, 2):
            layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        if pretrain:
            # pretrain model only keep first 20 layers
            layers += [
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(self.S * self.S * 1024, 4096),
            ]
        else:
            layers += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]

            # 6th part
            for _ in range(0, 2):
                layers += [
                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                ]

            # last 2 layers
            layers += [
                nn.Flatten(),
                nn.Linear(self.S * self.S * 1024, 4096),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(),
                nn.Linear(4096, self.S * self.S * self.depth)
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model.forward(x)
        shape = self.S * self.S * self.depth
        return torch.reshape(y, shape)

