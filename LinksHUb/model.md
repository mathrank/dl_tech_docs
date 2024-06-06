# 模型设计与实现

## 1. 模型组件

| 类别                      | 组件                                                         |
| ------------------------- | ------------------------------------------------------------ |
| 注意力机制                | SE, Non-local, CcNet, GC-Net, Gate, CBAM, Dual Attention, Spatial Attention, Channel Attention |
| 卷积结构                  | Residual block, Bottle-neck block, Split-Attention block, Depthwise separable convolution, Recurrent convolution, Group convolution, Dilated convolution, Octave convolution, Ghost convolution |
| 多尺度模块                | ASPP, PPM, DCM, DenseASPP, FPA, OCNet, MPM                   |
| 损失函数                  | Focal loss, Dice loss, BCE loss, Wetight loss, Boundary loss, Lovász-Softmax loss, TopK loss, Hausdorff distance(HD) loss, Sensitivity-Specificity (SS) loss, Distance penalized CE loss, Colour-aware Loss |
| 池化结构                  | Max pooling, Average pooling, Random pooling, Strip Pooling, Mixed Pooling |
| 归一化模块                | Batch Normalization, Layer Normalization, Instance Normalization, Group Normalization, Switchable Normalization, Filter Response Normalization |
| 学习衰减策略              | StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau |
| 优化算法                  | BGD, SGD, Adam, RMSProp, Lookahead                           |
| 数据增强                  | 水平翻转, 垂直翻转, 旋转, 平移, 缩放, 裁剪, 擦除, 反射变换, 亮度, 对比度, 饱和度, 色彩抖动, 对比度变换, 锐化, 直方图均衡, Gamma增强, PCA白化, 高斯噪声, GAN, Mixup |
| [骨干网络](backbone.html) | LeNet, ResNet, U-Net, DenseNet, VGGNet, GoogLeNet, Res2Net, ResNeXt, InceptionNet, SqueezeNet, ShuffleNet, SENet, DPNet, MobileNet, NasNet, DetNet, EfficientNet |

## 2. 模型设计与测试





## 3. 迁移学习