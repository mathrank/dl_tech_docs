# 计算平台和开发环境配置

## 1. 硬件配置

- **GPU**: NVIDIA RTX 3080 Ti

## 2. 软件配置

- **操作系统**: Windows 10 x64
- **Visual Studio**: 2019 
- **CUDA**: 12.1.0
- **cuDNN**: 8.8.1
- **TensorRT**: 8.6.1.6
- **OpenCV**: 4.9.0
- **ONNX Runtime**: 1.17.3

## 3. 深度学习环境配置

| 环境名称 |  框架   | 版本  | Python | CUDA | cuDNN |              推理部署               |
| :------: | :-----: | :---: | :----: | :--: | :---: | :---------------------------------: |
| pytorch  | PyTorch | 2.3.0 |  3.11  | 12.1 | 8.8.1 |      ONNX Runtime 1.17.3 + C++      |
| paddlex  | PaddleX | 2.1.0 |  3.8   | 11.2 | 8.2.1 | FastDeploy + TensorRT 8.6.1.6 + C++ |

## 4. 应用场景

- **PyTorch 环境**: 主要用于语义分割和 YOLO 目标检测
- **PaddleX 环境**: 主要用于 PaddleOCR 光学字符识别

## 5. 虚拟环境配置步骤

### 安装 [Anaconda](https://www.anaconda.com/download/success)

下载慢可以去[镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)下载，安装完之后打开 Anaconda Prompt 。

### PyTorch 环境

1. 创建 conda 虚拟环境并激活:

   ```bash
   conda create -n pytorch python=3.11
   
   conda activate pytorch
   ```

2. 安装 [PyTorch](https://pytorch.org/get-started/locally/) :

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. 安装其他依赖（缺啥安啥）:

   ```bash
   pip install 依赖名 -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. 验证环境和版本:

   ```bash
   python
   
   import torch
   
   print("CUDA Version:", torch.version.cuda, "cuDNN Version:", torch.backends.cudnn.version())
   ```

### PaddleX 环境

1. 创建 conda 虚拟环境并激活:

   ```bash
   conda create -n paddlex python=3.8
   
   conda activate paddlex
   ```

2. 安装 [PaddleX](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/quick_start_API.md) 和相关依赖:

   ```bash
   conda install git
   
   pip install cython -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
   
   git clone https://github.com/PaddlePaddle/PaddleX.git
   
   cd PaddleX
   
   git checkout develop
   
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   python setup.py install
   
   pip install filelock -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   conda remove paddlepaddle-gpu
   
   conda install paddlepaddle-gpu==2.3.2 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
   ```

3. 验证环境和版本:

   ```bash
   python
   
   import paddle
   
   import paddlex
   
   print(f"CUDA Version: {paddle.version.cuda()},cuDNN Version: {paddle.version.cudnn()}")
   ```

> PaddleX 2.1.0的安装环境极为苛刻，经过不断尝试几乎是唯一解，期待六月份的 3.0 。
>
> `git clone https://github.com/PaddlePaddle/PaddleX.git`在不开魔法的情况下容易出错，可以替换成：
>
> 下载 [PaddleX压缩包](https://pan.baidu.com/s/1pd43o11xvKkNxucokBi42g?pwd=dba5)并解压到 Anaconda Prompt对应的环境路径下。
>
> 验证时会弹出一大堆警告，忽略即可，再次编译会消失。

### Jupyter Notebook

打开 Anaconda Navigator ，分别切换刚刚安装好的环境，再 Install 下载 Jupyter Notebook 。

修改工作目录：

1. 在 Windows 菜单中找到 jupyter notebook (pytorch/paddlex) ，点击鼠标右键，从更多中打开文件位置
2. 找到 jupyter 右键打开属性， 将目标中的 %USERPROFILE%/ 替换为工作路径，注意前面有个空格不能删

## 6. 标注平台

- **标注工具**:  Labelme

- **安装方法**:

  ```bash
  pip install labelme -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- **使用方法**:
  ```bash
  labelme
  ```
  或者在 jupyter notebook / python 环境中输入
  ```python
  !labelme
  ```

- **Labelme 标注工具输出的 JSON 文件格式如下**:

  ```json
  {
    "version": "5.4.1",
    "flags": {},
    "shapes": [
      {
        "label": "标签名称",
        "points": [[x1, y1], [x2, y2], ...],
        "group_id": null,
        "shape_type": "形状类型",
        "flags": {}
      }
    ],
    "imagePath": "图像文件名",
    "imageData": "图像数据（base64 编码）",
    "imageHeight": 图像高度,
    "imageWidth": 图像宽度
  }
  ```

## 7. 推理部署环境配置

### 软件环境

1. 下载安装 [CUDA 12.1.0](https://developer.nvidia.com/cuda-toolkit-archive)（安装在 C 盘，勾选 Visual Studio Integration）

2. 下载解压 [cuDNN 8.8.1](https://developer.nvidia.com/rdp/cudnn-archive)（for CUDA 12.x）

3. 复制 cuDNN 文件夹里所有东西到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

4. 下载安装 [OpenCV 4.9.0](https://opencv.org/releases/)，下载解压 [ONNX RunTime 1.17.3](https://github.com/microsoft/onnxruntime/tags) 、[TensorRT 8.6.1.6](https://developer.nvidia.com/tensorrt/download)

5. 将 TensorRT-8.6.1.6\include 中所有头文件 copy 到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include

   将 TensorRT-8.6.1.6\lib 中所有 lib 文件 copy 到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64

   将 TensorRT-8.6.1.6\lib 中所有 dll 文件 copy 到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin

6. 下载 [MSBuild](https://pan.baidu.com/s/17okJk17kkG0d8PHYX08-Tw?pwd=dba5) 解压到 C:\Users\用户名\AppData\Local\Microsoft ，可以先查看有没有 MSBuild

7. 打开 Visual Studio 2019 ，[配置 OpenCV 环境](vs2019_opencv4.9_setup.html)

> 另附：[安装包](https://pan.baidu.com/s/1fCaesikH2DqwzrJ6bvxqeQ?pwd=dba5)百度网盘链接

### Pytorch 部署环境

1. 配置 ONNX RunTime 环境，具体参考 OpenCV 环境，里面包含了 ONNX RunTime 路径截图
2. 运行时把 ...\onnxruntime-win-x64-gpu-1.17.3\lib 中四个 dll 复制到 exe 所在目录下

### PaddleX 部署环境

#### 1. FastDeploy C++ SDK 编译安装

- [预编译库安装](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)

  下载解压 [fastdeploy-win-x64-gpu-1.0.7.zip](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.1.zip) ，Github 进不去可以[网盘](https://pan.baidu.com/s/1SZgcovhxjYoUYVe0aoDRxA?pwd=dba5)下载

> 该版本由 Visual Studio 2019 , CUDA 11.2 , cuDNN 8.2 编译产出
>
> 版本信息 : Paddle Inference 2.4-dev5，ONNXRuntime 1.12.0，OpenVINO 2022.2.0.dev20220829，TensorRT 8.5.2.2

- [部署库编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)

  下载解压我编译好的 [fastdeploy_sdk.zip](https://pan.baidu.com/s/1p7D2oB101aelzQRliXa8ZQ?pwd=dba5) 

> 该版本由 Visual Studio 2019 , CUDA 12.1 , cuDNN 8.8.1 编译产出
>
> 版本信息 : ONNXRuntime 1.17.3 , TensorRT 8.6.1.6

或者根据需求自编译，需求编译选项表：

| 选项                    | 支持平台                                           | 说明                                                         |
| ----------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| WITH_GPU                | Linux(x64)/Windows(x64)                            | 默认OFF，当编译支持Nvidia-GPU时，需设置为ON                  |
| ENABLE_ORT_BACKEND      | Linux(x64/aarch64)/Windows(x64)/Mac OSX(arm64/x86) | 默认OFF, 是否编译集成ONNX Runtime后端                        |
| ENABLE_PADDLE_BACKEND   | Linux(x64)/Windows(x64)                            | 默认OFF，是否编译集成Paddle Inference后端                    |
| ENABLE_TRT_BACKEND      | Linux(x64)/Windows(x64)                            | 默认OFF，是否编译集成TensorRT后端                            |
| ENABLE_OPENVINO_BACKEND | Linux(x64)/Windows(x64)/Mac OSX(x86)               | 默认OFF，是否编译集成OpenVINO后端(仅支持CPU)                 |
| ENABLE_VISION           | Linux(x64)/Windows(x64)                            | 默认OFF，是否编译集成视觉模型的部署模块                      |
| ENABLE_TEXT             | Linux(x64)/Windows(x64)                            | 默认OFF，是否编译集成文本NLP模型的部署模块                   |
| CUDA_DIRECTORY          | Linux(x64)/Windows(x64)                            | 默认/usr/local/cuda，要求CUDA>=11.2                          |
| TRT_DIRECTORY           | Linux(x64)/Windows(x64)                            | 默认为空，要求TensorRT>=8.4， 指定路径如/Download/TensorRT-8.5 |
| WITH_CAPI               | Linux(x64)/Windows(x64)/Mac OSX(x86)               | 默认OFF，是否编译集成C API                                   |
| WITH_CSHARPAPI          | Windows(x64)                                       | 默认OFF，是否编译集成C# API                                  |

第三方库依赖指定（不设定如下参数，会自动下载预编译库）

| 选项               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| ORT_DIRECTORY      | 当开启ONNX Runtime后端时，用于指定用户本地的ONNX Runtime库路径；如果不指定，编译过程会自动下载ONNX Runtime库 |
| OPENCV_DIRECTORY   | 当ENABLE_VISION=ON时，用于指定用户本地的OpenCV库路径；如果不指定，编译过程会自动下载OpenCV库 |
| OPENVINO_DIRECTORY | 当开启OpenVINO后端时, 用于指定用户本地的OpenVINO库路径；如果不指定，编译过程会自动下载OpenVINO库 |

Windows 编译需要满足条件

- Windows 10/11 x64
- Visual Studio 2019
- cuda >= 11.2
- cudnn >= 8.2

> 安装 CUDA 时需要勾选`Visual Studio Integration` , 或者手动将`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\visual_studio_integration\MSBuildExtensions\`文件夹下的 4 个文件复制到`C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations\`文件夹。否则执行 cmake 命令时可能会遇到 `No CUDA toolset found` 报错。

在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，分别执行如下命令

> git clone 出错可以下载 [FastDeploy](https://pan.baidu.com/s/14DcON8bZpngjfzWdFGqBxg?pwd=dba5) 解压到工作目录
>
> 执行 cmake 前先修改 ...\FastDeploy\cmake\cuda.cmake
>
> 把 set(fd_known_gpu_archs "35 50 52 60 61 70 75 80 86")
>      set(fd_known_gpu_archs10 "35 50 52 60 61 70 75") 中的 35 都删掉（git clone的也要改）

```bash
cd 工作目录

git clone https://github.com/PaddlePaddle/FastDeploy.git

cd FastDeploy

mkdir build && cd build

cmake .. -G "Visual Studio 16 2019" -A x64 ^
         -DENABLE_ORT_BACKEND=ON ^
         -DENABLE_PADDLE_BACKEND=ON ^
         -DENABLE_OPENVINO_BACKEND=ON ^
         -DENABLE_TRT_BACKEND=ON ^
         -DENABLE_VISION=ON ^
         -DENABLE_TEXT=ON ^
         -DWITH_GPU=ON ^
         -DTRT_DIRECTORY="D:\TensorRT-8.6.1.6" ^
         -DCUDA_DIRECTORY="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" ^
         -DCMAKE_INSTALL_PREFIX="D:\fastdeploy_sdk"
         
% nuget restore  （请在设置 WITH_CSHARPAPI=ON 的情况下执行它，以准备C#的依赖项)

msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64

msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```

> 根据需要修改选项，替换为你的路径
>
> `msbuild fastdeploy.sln` 会产生一千多个警告，可能是版本太新

编译完成后，即在 `CMAKE_INSTALL_PREFIX` 指定的目录下生成 C++ 推理库

#### 2. [FastDeploy C++ 库在 Windows 上的多种使用方式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows_build.md)

推荐 Visual Studio 2019 创建 CMake 工程使用 C++ SDK ，进不去请[点击](use_sdk_on_windows_build.html)
