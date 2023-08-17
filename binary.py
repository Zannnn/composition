import torch
import torch.nn as nn
import numpy as np

# 定义U-Net网络模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # 编码器
        x1 = self.encoder(x)
        # 解码器
        x = self.decoder(x1)

        return x

# 创建U-Net网络实例
in_channels = 3  # 输入图像通道数
out_channels = 1  # 输出图像通道数，这里是一个01矩阵，因此通道数为1
model = UNet(in_channels, out_channels)

# 输出模型结构
print(model)

# input1 = torch.rand([1,3,640,480])
# output1 = model(input1)
# output1 = torch.sigmoid(output1)  # 应用sigmoid函数将输出映射到0-1范围
# output1 = (output1 > 0.5).float()
# print(output1)

import torchvision.transforms as transforms
from PIL import Image

# # 加载预训练的U-Net模型
# model = UNet(in_channels, out_channels)
# model.load_state_dict(torch.load('unet_model.pth'))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((480, 640)),  # 调整图像大小为256x256
    transforms.ToTensor()  # 转换为张量
])

# # 读取测试图像
image_path = 'dataset\\train\\0.jpg'
image_source = Image.open(image_path).convert('RGB')
image_source_array = np.array(image_source)

# 图像预处理
image = transform(image_source)
image = image.unsqueeze(0)  # 添加一个维度作为批处理维度

# 模型推理
with torch.no_grad():
    output = model(image)

# 处理输出结果
output = torch.sigmoid(output)  # 应用sigmoid函数将输出映射到0-1范围
output = (output > 0.5).float()  # 将输出二值化为0或1

# 输出结果保存为图片
output = output.squeeze(0).squeeze(0)  # 移除批处理和通道维度
output = output.numpy()  # 转换为NumPy数组
mask_image = Image.fromarray((output * 255).astype('uint8'), mode='L')  # 转换为PIL图像
mask_image.save('mask.png')

output_image = np.zeros_like(image_source)
output_image[output>0] = image_source_array[output>0]
output_image_PIL = Image.fromarray(output_image)
output_image_PIL.save("output.png")