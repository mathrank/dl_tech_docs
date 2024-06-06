# PaddleOCR 微调实践：温控表数码管

## 1. 背景介绍

温控表数码管是一种在各类温控设备中广泛使用的显示组件，主要用于显示设备的当前温度设置或实际温度读数。这些数码管通常仅显示四位数字，足以提供从室温到高温处理所需的温度范围信息。在温控系统中，能够准确快速地读取这些数字是确保环境条件稳定和过程控制精准的关键。

![温控表](https://i.imgur.com/92DD2ik.jpg)

尽管温控表的功能重要，但传统上这些设备缺乏将显示数据自动化输入到计算系统中的能力。操作员通常需要手动读取并记录这些数据，这一过程不仅耗时，而且容易因人为错误而影响数据的准确性。如果可以使用机器替代人工，将节约大量成本。针对上述问题，希望通过摄像头拍照->智能读数的方式高效地完成此任务。

- 智能读数

智能读数是一种应用 OCR（光学字符识别）技术的实践，专门用于自动从图像中识别数字和文字。这种技术广泛应用于各种设备和系统，如温控表数码管，以自动捕捉并解析显示的温度读数，提高数据输入的速度和准确性。通过使用智能读数技术，可以减少人工干预，实现更高效和可靠的数据管理和监控流程。

在智能读数的实现中，主要存在两种OCR模型方案：二阶段模型和端到端模型。每种方案都有其独特的优势和应用场景：

| 特性/模型    | 二阶段模型                             | 端到端模型                           |
| ------------ | -------------------------------------- | ------------------------------------ |
| **描述**     | 分步处理：先文本检测，后文本识别       | 单步处理：直接从图像到文本           |
| **精度**     | 高，由于可以独立优化检测和识别步骤     | 可变，依赖于模型训练和数据的质量和量 |
| **灵活性**   | 高，可根据需求选择不同的检测和识别技术 | 低，一体化模型减少了灵活调整的可能   |
| **稳定性**   | 高，各步骤经过精细调整                 | 中到高，依赖于模型的整体训练效果     |
| **处理速度** | 一般，每个步骤需要单独处理             | 快，减少了处理步骤                   |
| **适用场景** | 高精度和复杂文本环境                   | 快速响应需求和较简单的文本环境       |

- 方案选择

**准确性需求**：温控表数码管通常显示关键的操作数据，如温度，错误的读数可能导致设备操作不当。二阶段模型在精确度和可靠性方面的表现更为优越。

**复杂环境适应性**：温控设备可能处于各种环境中，如不同的光照、角度和背景干扰。二阶段模型允许针对这些复杂因素进行更细致的调整和优化。

**技术成熟度**：二阶段模型在工业应用中已有广泛的实践和验证，使其更为可靠。

综上，虽然端到端模型在某些场景下可能提供更快的处理速度，但考虑到智能读数的高精度要求和环境的复杂性，二阶段模型是一个更合适的选择。这种选择不仅保证了读数的准确性，也确保了整个系统的稳定性和可靠性。

- 技术实现

本文采用 PaddleOCR 提供的检测与识别模型进行微调。PaddleOCR 是一个由百度开发的先进的光学字符识别系统，支持多种语言和多种文字检测与识别技术。选择 PaddleOCR 的原因在于其强大的功能和灵活性，能够高效地适应和处理温控表数码管的 OCR 任务。通过对这些预训练模型进行微调，可以进一步提升模型在特定应用场景下的表现和准确性。

## 2. 环境搭建

默认已经安装好了 [PaddleX 虚拟环境](dev_setup.html)，环境搭建推荐在 Anaconda Prompt 中运行：

```bash
conda activate paddlex

cd 工作目录

git clone https://github.com/PaddlePaddle/PaddleOCR

python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

python -m pip install paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple

cd PaddleOCR

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 可下载 [PaddleOCR压缩包](https://pan.baidu.com/s/1QWml-fAWs2Co4LL3b6fYmA?pwd=dba5)解压到工作目录下代替 git clone
>
> 使用过程中会出现许多 DeprecationWarning ，忽略即可，不影响结果，目前网上没有解决方案

简单测试一下原始 PP-OCRv4 识别模型的效果，准备工作目录下的

rec_test1.jpg ：

![Imgur](https://i.imgur.com/cYwNnWz.jpg)

```bash
paddleocr --lang=ch --det=False --image_dir="../rec_test1.jpg" --show_log=False
```

效果如下：

```bash
ppocr INFO: ('0298', 0.8551594018936157)
```

rec_test2.jpg ：

![Imgur](https://i.imgur.com/SsN6OQ9.jpg)

```bash
paddleocr --lang=ch --det=False --image_dir="../rec_test2.jpg" --show_log=False
```

效果如下：

```bash
ppocr INFO: ('D3CE', 0.24095825850963593)
```

由于 [paddleocr 检测模式从 2.7.0.3 版本开始不再起作用](https://github.com/PaddlePaddle/PaddleOCR/issues/10924)，所以

```bash
paddleocr --lang=ch --rec=False --image_dir="../rec_test1.jpg" --show_log=False
```

会报错 `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()` ，测试原始 PP-OCRv4 检测模型可以下载[超轻量中文检测模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)解压到 PaddleOCR/tools/infer_model/ 下，将背景介绍中的图片 det_test.jpg 保存在工作目录下，接着使用如下命令：

```bash
python tools/infer/predict_det.py --image_dir="../det_test.jpg" --det_model_dir="./tools/infer_model/ch_PP-OCRv4_det_infer/"
```

可视化文本检测结果默认保存到 PaddleOCR/inference_results/，结果文件的名称前缀为 'det_res'。结果如下：

![Imgur](https://i.imgur.com/Fj3iSkw.jpg)

可以发现在图片较为模糊的情况下识别不准确；而检测则完全不准确。这是因为 PP-OCRv4 的训练数据大多为通用场景数据，在特定的场景上效果可能不够好。因此需要基于场景数据进行微调。

以上测试代码更推荐在 Jupyter Notebook (paddlex) 中运行，可以保存运行日志：

```python
import os

os.chdir('/PaddleOCR')

!paddleocr --lang=ch --det=False --image_dir="../rec_test1.jpg" --show_log=False

!paddleocr --lang=ch --det=False --image_dir="../rec_test2.jpg" --show_log=False

!python tools/infer/predict_det.py --image_dir="../det_test.jpg" --det_model_dir="./tools/infer_model/ch_PP-OCRv4_det_infer/"
```

## 3. 数据准备

特定的工业场景往往很难获取带标注的真实数据集，温控表也是如此。在实际工业场景中，可以通过摄像头采集的方法收集大量真实特定场景数据，在模型微调时，加入真实通用场景数据，可以进一步提升模型精度与泛化性能。

### 3.1 检测任务

- 数据量：建议至少准备500张的文本检测数据集用于模型微调
- 数据标注：单行文本标注格式，建议标注的检测框与实际语义内容一致。如在火车票场景中，姓氏与名字可能离得较远，但是它们在语义上属于同一个检测字段，这里也需要将整个姓名标注为1个检测框。

真实特定场景数据最好大于500张，包括了各种数据增强，再加入一定的真实通用场景数据，公开数据集如下：

| 数据集名称 |                         直链下载地址                         |                    PaddleOCR 标注下载地址                    |
| :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ICDAR 2015 |      [下载](https://rrc.cvc.uab.es/?ch=4&com=downloads)      | [train](https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt) / [test](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt) |
|  ctw1500   | [下载](https://paddleocr.bj.bcebos.com/dataset/ctw1500.zip)  |                     图片下载地址中已包含                     |
| total text | [下载](https://paddleocr.bj.bcebos.com/dataset/total_text.tar) |                     图片下载地址中已包含                     |
|   td tr    |  [下载](https://paddleocr.bj.bcebos.com/dataset/TD_TR.tar)   |                     图片下载地址中已包含                     |

### 3.2 识别任务

- 数据量：不更换字典的情况下，建议至少准备5000张的文本识别数据集用于模型微调；如果更换了字典，需要的数量更多。

- 数据分布：建议分布与实测场景尽量一致。如果实测场景包含大量短文本，则训练数据中建议也包含较多短文本，如果实测场景对于空格识别效果要求较高，则训练数据中建议也包含较多带空格的文本内容。

- 数据合成：针对部分字符识别有误的情况，建议获取一批特定字符数据，加入到原数据中使用小学习率微调。其中原始数据与新增数据比例可尝试 10:1 ～ 5:1， 避免单一场景数据过多导致模型过拟合，同时尽量平衡语料词频，确保常用字的出现频率不会过低。

- 特定字符生成可以使用 TextRenderer 工具，可参考 [text_renderer 批量生成数据集](synthetic_data.html)，合成数据语料尽量来自真实使用场景，在贴近真实场景的基础上保持字体、背景的丰富性，有助于提升模型效果。

- 通用中英文数据：在训练的时候，可以在训练集中添加通用真实数据（如在不更换字典的微调场景中，建议添加LSVT、RCTW、MTWI等真实数据），进一步提升模型的泛化性能。

可以在训练初期使用较高比例的合成数据可以帮助模型快速学习基本的文本特征，然后逐渐增加真实特定场景数据的比例，以细化和优化模型的表现。公开数据集如下：

|                          数据集名称                          |                         直链下载地址                         |                    PaddleOCR 标注下载地址                    |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| en benchmark(MJ, SJ, IIIT, SVT, IC03, IC15, SVTP, and CUTE.) | [DTRB](https://githubfast.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | LMDB格式，可直接用[`lmdb_dataset.py`](https://githubfast.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppocr/data/lmdb_dataset.py)加载 |
|                          ICDAR 2015                          |      [下载](http://rrc.cvc.uab.es/?ch=4&com=downloads)       | [train](https://paddleocr.bj.bcebos.com/dataset/rec_gt_train.txt)/ [test](https://paddleocr.bj.bcebos.com/dataset/rec_gt_test.txt) |
|                         多语言数据集                         | [下载](https://pan.baidu.com/s/1mq1A77HZZ2l9qye0dWY3_Q?pwd=dba5) |                     图片下载地址中已包含                     |

### 3.3 数据标注

使用 [PPOCRLabel](dev_setup.html) 进行半自动化标注，标注完有如下文件结构：

```tex
|-train_data
  |-crop_img
    |- word_001_crop_0.png
    |- word_002_crop_0.jpg
    |- word_003_crop_0.jpg
    | ...
  | Label.txt
  | rec_gt.txt
  |- word_001.png
  |- word_002.jpg
  |- word_003.jpg
  | ...
```

下面是我收集并制作的带批注的数据集：

- [检测数据集](https://pan.baidu.com/s/1rmeBOaKVjbWd1_1T56thAA?pwd=dba5)：包含训练数据集3653张图片，验证数据集680张，图片都统一调整至规定的大小：190×80像素，数据集都进行了数据增强处理，包括颜色变换、添加滤波干扰等方法。
- [识别数据集](https://pan.baidu.com/s/1pbIENYoI1zeqMIczXsOhFg?pwd=dba5)：包含训练数据集1717张图片，测试数据集341张，数据集都进行了数据增强处理，包括反转、颜色变换等方法。

文件结构如下：

```tex
|-LED_Thermostat_Digits_det
  |-train
    |- train_0001.jpg
    |- train_0002.jpg
    | ...
  |-val
    |- val_0001.jpg
    |- val_0002.jpg
    | ...
  |-train.txt
  |-val.txt

|-LED_Thermostat_Digits_rec
  |-train
    |- train_0001.jpg
    |- train_0002.jpg
    | ...
  |-val
    |- val_0001.jpg
    |- val_0002.jpg
    | ...
  |-train.txt
  |-val.txt
```

将数据集解压到 PaddleOCR/data/ 目录下（没有的目录需要自己创建）

## 4. 文本识别模型微调

### 4.1 模型选择

建议选择 PP-OCRv3 模型进行微调，其精度与泛化性能是 PP-OCRv4 模型出来之前的最优预训练模型。PP-OCRv4 模型微调我还在研究中。

#### 4.1.1 PP-OCRv3 识别模型训练

1. **下载预训练模型**

下载 [PP-OCRv3 中文预训练识别模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar)解压到 PaddleOCR/pretrain_models/ 下

2. **自定义字典文件**

接下来需要提供一个字典（{word_dict_name}.txt），使模型在训练时，可以将所有出现的字符映射为字典的索引。

因此字典需要包含所有希望被正确识别的字符，{word_dict_name}.txt需要写成如下格式，并以 `utf-8` 编码格式保存：

```tex
0
1
2
3
4
5
6
7
8
9
-
.
```

word_dict.txt 每行有一个单字，将字符与数字索引映射在一起，例如“-3.14” 将被映射成 [ - , 3 , . , 1 , 4 ]

PaddleOCR内置了一部分字典：

`ppocr/utils/ppocr_keys_v1.txt` 是一个包含6623个字符的中文字典，是预训练模型的字典；

`ppocr/utils/ic15_dict.txt` 是一个包含36个字符的英文字典；

其余字典在 ppocr/utils/ 或 ppocr/utils/dict/ 下。

内置字典面向通用场景，而在具体的工业场景中，可能需要识别特殊字符，或者只需识别某几个字符，此时自定义字典会更提升模型精度。例如在温控表场景中，只需要识别数字。（其实还有错误代码或状态指示，例如 Err ，不过我没有收集到数据，本文中暂不考虑）

遍历真实数据标签中的字符，制作 ppocr/utils/dict/ 下的字典 `digital_dict.txt` 如下所示：

```tex
0
1
2
3
4
5
6
7
8
9
```

3. **修改配置文件**

为了更好的使用预训练模型，训练推荐使用[ch_PP-OCRv3_rec_distillation.yml](ppocr_yml.html) 配置文件，并参考下列说明修改配置文件：

（位于 PaddleOCR/configs/rec/PP-OCRv3/）

```yaml
Global:
  ...
  # 添加自定义字典，如修改字典请将路径指向新字典
  character_dict_path: ppocr/utils/dict/digital_dict.txt
  ...

Optimizer:
  ...
  # 添加学习率衰减策略
  lr:
    name: Cosine
    learning_rate: 0.0001
    warmup_epoch: 2
  ...

...

Train:
  dataset:
    # 数据集格式，支持LMDBDataSet以及SimpleDataSet
    name: SimpleDataSet
    # 数据集路径
    data_dir: ./data/LED_Thermostat_Digits_rec/
    # 训练集标签文件
    label_file_list:
    - ./data/LED_Thermostat_Digits_rec/train.txt
    ratio_list:
    - 1.0
    transforms:
      ...
      - RecResizeImg:
          # 修改 image_shape 以适应长文本
          image_shape: [3, 48, 320]
      ...
  loader:
    ...
    # 单卡训练的batch_size
    batch_size_per_card: 128
    ...

Eval:
  dataset:
    # 数据集格式，支持LMDBDataSet以及SimpleDataSet
    name: SimpleDataSet
    # 数据集路径
    data_dir: ./data/LED_Thermostat_Digits_rec/
    # 验证集标签文件
    label_file_list: 
    - ./data/LED_Thermostat_Digits_rec/val.txt
    transforms:
      ...
      - RecResizeImg:
          # 修改 image_shape 以适应长文本
          image_shape: [3, 48, 320]
      ...
  loader:
    # 单卡验证的batch_size
    batch_size_per_card: 128
    ...
```

> **训练/预测/评估时的配置文件请务必与训练一致**
>
> 主要修改的地方有四点：
>
> - 字典路径
> - 学习率调整
> - 训练集路径
> - 验证集路径

4. **启动训练**

如果安装的是 cpu 版本，需要将配置文件中的 `use_gpu` 字段修改为 false

```bash
# GPU训练 支持单卡，多卡训练
# 训练数码管数据 训练日志会自动保存为 "{save_model_dir}" 下的train.log

#单卡训练
python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.pretrained_model="./pretrain_models/ch_PP-OCRv3_rec_train/best_accuracy"

#多卡训练，通过--gpus参数指定卡号
python -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.pretrained_model="./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy"
```

PaddleOCR支持训练和评估交替进行, 可以在 `configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml` 中修改 `eval_batch_step` 设置评估频率，默认每500个 iter 评估一次。评估过程中默认将最佳 acc 模型，保存为 `output/ch_PP-OCRv3_rec_distill/best_accuracy` 。如果验证集很大，测试将会比较耗时，建议减少评估次数，或训练完再进行评估。如果有以下错误，说明 GPU 显存不足，需要适当减少 `batch_size_per_card` ，或者使用多卡训练

```bash
SystemError: (Fatal) Operator conv2d raises an struct paddle::memory::allocation::BadAlloc exception.
The exception content is
:ResourceExhaustedError:

Out of memory error on GPU 0. Cannot allocate 240.250244MB memory on GPU 0, 0.000000B memory has been allocated and available memory is only 0.000000B.

Please check whether there is any other process using GPU 0.
1. If yes, please stop them, or start PaddlePaddle on another GPU.
2. If no, please decrease the batch size of your model.

 (at ..\paddle\fluid\memory\allocation\cuda_allocator.cc:87)
. (at ..\paddle\fluid\imperative\tracer.cc:307)
```

> 注意：`batch_size_per_card` 应与学习率保持线性关系
>
> 例：八卡训练 `batch_size` 为 128 时学习率为 1e-3，则单卡训练 `batch_size` 为 64 时学习率为 1e-3/16 左右

先按默认字典进行训练，发现训练完 800 个训练周期需要一天15小时，而且训练到第 21 个周期 acc 就达到 0.98 了，按这个周期数肯定会过拟合。原因是数据量太少，800 个训练周期大约需要 6.5 万张训练集图片；以及 PP-OCRv3 模型使用了 GTC 策略，其中 SAR 分支参数量大，当训练数据为简单场景时模型容易过拟合，导致微调效果不佳，可以选择去除 GTC 策略，模型结构部分配置文件修改如下：

```tex
Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 64
    depth: 2
    hidden_dims: 120
    use_guide: False
  Head:
    name: CTCHead
    fc_decay: 0.00001
Loss:
  name: CTCLoss

Train:
  dataset:
  ......
    transforms:
    # 去除 RecConAug 增广
    # - RecConAug:
    #     prob: 0.5
    #     ext_data_num: 2
    #     image_shape: [48, 320, 3]
    #     max_text_length: *max_text_length
    - RecAug:
    # 修改 Encode 方式
    - CTCLabelEncode:
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
...

Eval:
  dataset:
  ...
    transforms:
    ...
    - CTCLabelEncode:
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
...
```







为了简单测试效果可以将 `epoch_num` 设置为 50 ，同时 `eval_batch_step` 改为 [0, 400] 

修改为自定义字典，会弹出以下警告：

```bash
ppocr WARNING: The shape of model params Teacher.head.ctc_head.fc.bias [12] not matched with loaded params Teacher.head.ctc_head.fc.bias [6625] !
ppocr WARNING: The shape of model params Teacher.head.sar_head.decoder.embedding.weight [14, 512] not matched with loaded params Teacher.head.sar_head.decoder.embedding.weight [6627, 512] !
ppocr WARNING: The shape of model params Teacher.head.sar_head.decoder.prediction.weight [1536, 13] not matched with loaded params Teacher.head.sar_head.decoder.prediction.weight [1536, 6626] !
ppocr WARNING: The shape of model params Teacher.head.sar_head.decoder.prediction.bias [13] not matched with loaded params Teacher.head.sar_head.decoder.prediction.bias [6626] !
ppocr WARNING: The shape of model params Student.head.ctc_head.fc.weight [64, 12] not matched with loaded params Student.head.ctc_head.fc.weight [64, 6625] !
ppocr WARNING: The shape of model params Student.head.ctc_head.fc.bias [12] not matched with loaded params Student.head.ctc_head.fc.bias [6625] !
ppocr WARNING: The shape of model params Student.head.sar_head.decoder.embedding.weight [14, 512] not matched with loaded params Student.head.sar_head.decoder.embedding.weight [6627, 512] !
ppocr WARNING: The shape of model params Student.head.sar_head.decoder.prediction.weight [1536, 13] not matched with loaded params Student.head.sar_head.decoder.prediction.weight [1536, 6626] !
ppocr WARNING: The shape of model params Student.head.sar_head.decoder.prediction.bias [13] not matched with loaded params Student.head.sar_head.decoder.prediction.bias [6626] !
```

对结果没有影响，原因是输出层不匹配。减少参数缩短了训练时间，训练完 800 个训练周期仅需要18小时。自定义字典训练时前面的周期 acc 一直为 0 ，训练到一定周期数就好了，官方回答里也有提到：

>  **Q3.3.25: 识别模型训练时，loss能正常下降，但 acc一直为 0**
>
> **A**：识别模型训练初期 acc为 0是正常的，多训一段时间指标就上来了。

使用自定义字典训练 50 个 epoch 共计花费一小时二十分钟，结果如下：

```bash
ppocr INFO: best metric, acc: 0.9921874844970706, is_float16: False, norm_edit_dis: 0.9976562500366211, Teacher_acc: 0.9921874844970706, Teacher_norm_edit_dis: 0.9976562500366211, fps: 572.9446532584591, best_epoch: 45
```


#### 4.1.2 验证效果

- 指标评估

训练中模型参数默认保存在`Global.save_model_dir`目录下。在评估指标时，需要设置 `Global.checkpoints` 指向保存的参数文件。评估数据集可以通过 `configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml` 修改 Eval 中的 `label_file_path` 设置。

```bash
# GPU 评估， Global.checkpoints 为待测权重
python -m paddle.distributed.launch --gpus 0 tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.checkpoints="./output/rec_ppocr_v3_distillation/best_accuracy"
```

输出：

```bash
ppocr INFO: metric eval ***************
ppocr INFO: acc:0.9921874844970706
ppocr INFO: norm_edit_dis:0.9976562500366211
ppocr INFO: Teacher_acc:0.9921874844970706
ppocr INFO: Teacher_norm_edit_dis:0.9976562500366211
ppocr INFO: fps:132.1708639159654
```


- 测试识别效果

使用 PaddleOCR 训练好的模型，可以通过以下脚本进行快速预测。

默认预测图片存储在 `infer_img` 里，通过 `-o Global.checkpoints` 加载训练好的参数文件：

根据配置文件中设置的 `save_model_dir` 和 `save_epoch_step` 字段，会有以下几种参数被保存下来：

```yaml
output/rec_ppocr_v3_distillation/
├── best_accuracy.pdopt  
├── best_accuracy.pdparams  
├── best_accuracy.states  
├── config.yml  
├── iter_epoch_3.pdopt  
├── iter_epoch_3.pdparams  
├── iter_epoch_3.states  
├── latest.pdopt  
├── latest.pdparams  
├── latest.states  
└── train.log
```

其中 *best_accuracy.* 是评估集上的最优模型；*iter_epoch_x.* 是以 `save_epoch_step` 为间隔保存下来的模型；*latest.* 是最后一个 epoch 的模型。

```bash
# 重新测试 rec_test2.jpg
python tools/infer_rec.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.checkpoints="./output/rec_ppocr_v3_distillation/best_accuracy" Global.infer_img="../rec_test2.jpg"
```

预测图片：

![Imgur](https://i.imgur.com/SsN6OQ9.jpg)

得到输入图像的预测结果：

```bash
ppocr INFO: load pretrain successful from ./output/rec_ppocr_v3_distillation/best_accuracy
ppocr INFO: infer_img: rec_test2.jpg
ppocr INFO: 	 result: {"Student": {"label": "0397", "score": 0.9985830783843994}, "Teacher": {"label": "0397", "score": 0.997998058795929}}
ppocr INFO: success!
```

为测试模型的泛化能力，截取背景介绍中温控表的图片：

![Imgur](https://i.imgur.com/qkC7joA.jpg)

预测结果：

```bash
result: {"Student": {"label": "0360", "score": 0.9610638618469238}, "Teacher": {"label": "0360", "score": 0.8638999104499817}}
```

可以发现，微调后的模型在特定场景上的精度有了大幅提升。

### 4.2 模型导出和推理

Inference 模型（通过 `paddle.jit.save` 保存）通常是训练完成后的固化模型，包含了模型结构和参数，主要用于预测和部署场景。与此相对的是训练过程中保存的checkpoint模型，这类模型主要保存参数信息，用于训练中断后的恢复等场合。相比于仅含参数的checkpoints，inference 模型还包括了完整的模型结构信息，这在预测部署和推理加速中提供了性能上的优势，使其更加灵活且适合实际系统的集成。

- 识别模型转inference模型：

```bash
# -c 后面设置训练算法的yml配置文件
# -o 配置可选参数
# Global.pretrained_model 参数设置待转换的训练模型地址，不用添加文件后缀 .pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python tools/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.pretrained_model="./output/rec_ppocr_v3_distillation/best_accuracy" Global.save_inference_dir="./inference/ch_PP-OCRv3_rec/"
```

转换成功后，在目录下有 Student 和 Teacher 两个文件夹，每个文件夹各有三个文件：

```
inference/ch_PP-OCRv3_rec/Student/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```

这是因为使用了知识蒸馏方法来进行训练，最终部署更倾向于使用 Student 模型。


- 自定义模型推理

  如果训练时修改了文本的字典，在使用 inference 模型预测时，需要通过`--rec_char_dict_path`指定使用的字典路径

```bash
python tools/infer/predict_rec.py --image_dir="../rec_test2.jpg" --rec_model_dir="./inference/ch_PP-OCRv3_rec/Student/" --rec_char_dict_path="./ppocr/utils/dict/digital_dict.txt"
```

否则会出现乱码：

```bash
ppocr INFO: Predicts of rec_test2.jpg:('绚溜娇绚', 0.9985830783843994)
```

`image_dir` 也可以是文件夹，例如：

```bash
--image_dir="./data/val/"
```

- 测试推理速度

  推理耗时是部署时另一个重要的评价指标，测试时可以通过打开 --benchamark 开关统计模型实际使用的推理速度。

```bash
python tools/infer/predict_rec.py --image_dir="./data/val/" --rec_model_dir="./inference/ch_PP-OCRv3_rec/Student/" --rec_char_dict_path="./ppocr/utils/dict/digital_dict.txt" --benchmark=True
```

> 报错：
>
> ```bash
> ModuleNotFoundError: No module named 'auto_log'
> ```
> 解决：
>
> ```bash
> conda activate paddlex
> 
> cd 工作目录/PaddleOCR
> 
> git clone https://github.com/LDOUBLEV/AutoLog
> 
> cd AutoLog
> 
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> 
> python setup.py bdist_wheel
> 
> pip install ./dist/auto_log-1.2.0-py3-none-any.whl
> ```
>
> 注意：自行前往....../PaddleOCR/AutoLog/dist/查看并修改最后一句的 auto_log版本

控制台会打印出如下预测耗时信息：

```bash
[2024/05/31 16:50:25] ppocr INFO: ----------------------- Model info ----------------------
[2024/05/31 16:50:25] ppocr INFO:  model_name: rec
[2024/05/31 16:50:25] ppocr INFO:  precision: fp32
[2024/05/31 16:50:25] ppocr INFO: ----------------------- Data info -----------------------
[2024/05/31 16:50:25] ppocr INFO:  batch_size: 6
[2024/05/31 16:50:25] ppocr INFO:  input_shape: dynamic
[2024/05/31 16:50:25] ppocr INFO:  data_num: 114
[2024/05/31 16:50:25] ppocr INFO: ----------------------- Perf info -----------------------
[2024/05/31 16:50:25] ppocr INFO:  cpu_rss(MB): 3688.8906, gpu_rss(MB): 1467.0, gpu_util: 47.0%
[2024/05/31 16:50:25] ppocr INFO:  total time spent(s): 2.6224
[2024/05/31 16:50:25] ppocr INFO:  preprocess_time(ms): 0.0, inference_time(ms): 22.5543, postprocess_time(ms): 0.4489
```

### 4.3 转ONNX模型

Paddle2ONNX 支持将 **PaddlePaddle** 模型格式转化到 **ONNX** 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括 TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。

- 安装paddle2onnx

```bash
python -m pip install paddle2onnx -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- inference模型转onnx模型

```bash
paddle2onnx --model_dir="./inference/ch_PP-OCRv3_rec/Student/" --model_filename="./inference.pdmodel" --params_file="./inference.pdiparams" --save_file="./inference/ch_PP-OCRv3_rec/Student/inference.onnx" --opset_version=11 --enable_onnx_checker=True
```

可调整的转换参数如下表:

| 参数                       | 参数说明                                                     |
| -------------------------- | ------------------------------------------------------------ |
| --model_dir                | 配置包含 Paddle 模型的目录路径                               |
| --model_filename           | **[可选]** 配置位于 `--model_dir` 下存储网络结构的文件名     |
| --params_filename          | **[可选]** 配置位于 `--model_dir` 下存储模型参数的文件名称   |
| --save_file                | 指定转换后的模型保存目录路径                                 |
| --opset_version            | **[可选]** 配置转换为 ONNX 的 OpSet 版本，目前支持 7~16 等多个版本，默认为 9 |
| --enable_onnx_checker      | **[可选]** 配置是否检查导出为 ONNX 模型的正确性, 建议打开此开关， 默认为 False |
| --enable_auto_update_opset | **[可选]** 是否开启 opset version 自动升级功能，当低版本 opset 无法转换时，自动选择更高版本的 opset进行转换， 默认为 True |
| --deploy_backend           | **[可选]** 量化模型部署的推理引擎，支持 onnxruntime、tensorrt 或 others，当选择 others 时，所有的量化信息存储于 max_range.txt 文件中，默认为 onnxruntime |
| --save_calibration_file    | **[可选]** TensorRT 8.X版本部署量化模型需要读取的 cache 文件的保存路径，默认为 calibration.cache |
| --version                  | **[可选]** 查看 paddle2onnx 版本                             |
| --external_filename        | **[可选]** 当导出的 ONNX 模型大于 2G 时，需要设置 external data 的存储路径，推荐设置为：external_data |
| --export_fp16_model        | **[可选]** 是否将导出的 ONNX 的模型转换为 FP16 格式，并用 ONNXRuntime-GPU 加速推理，默认为 False |
| --custom_ops               | **[可选]** 将 Paddle OP 导出为 ONNX 的 Custom OP，例如：--custom_ops '{"paddle_op":"onnx_op"}，默认为 {} |

## 5. 文本检测模型微调

### 5.1 模型选择

选择了 PP-OCRv4 模型进行尝试

#### 5.1.1 PP-OCRv4 检测模型训练

1. **下载预训练模型**

下载 [PP-OCRv4 中文预训练检测模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar)解压到 PaddleOCR/pretrain_models/ 下

2. **修改配置文件**

配置文件： PaddleOCR/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml

```yaml
Global:
  debug: false
  use_gpu: true
  epoch_num: &epoch_num 500
  log_smooth_window: 20
  print_batch_step: 100
  save_model_dir: ./output/ch_PP-OCRv4_det
  save_epoch_step: 10
  eval_batch_step:
  - 0
  - 1500
  cal_metric_during_train: false
  checkpoints:
  pretrained_model:
  save_inference_dir: null
  use_visualdl: false
  infer_img: doc/imgs_en/img_10.jpg
  save_res_path: ./checkpoints/det_db/predicts_db.txt
  distributed: true

Architecture:
  model_type: det
  algorithm: DB
  Transform: null
  Backbone:
    name: PPLCNetV3
    scale: 0.75
    det: True
  Neck:
    name: RSEFPN
    out_channels: 96
    shortcut: True
  Head:
    name: DBHead
    k: 50

Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0001 #(8*8c)
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 5.0e-05

PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5

Metric:
  name: DetMetric
  main_indicator: hmean

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/LED_Thermostat_Digits_det/
    label_file_list:
      - ./data/LED_Thermostat_Digits_det/train.txt
    ratio_list: [1.0]
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - CopyPaste: null
    - IaaAugment:
        augmenter_args:
        - type: Fliplr
          args:
            p: 0.5
        - type: Affine
          args:
            rotate:
            - -10
            - 10
        - type: Resize
          args:
            size:
            - 0.5
            - 3
    - EastRandomCropData:
        size:
        - 640
        - 640
        max_tries: 50
        keep_ratio: true
    - MakeBorderMap:
        shrink_ratio: 0.4
        thresh_min: 0.3
        thresh_max: 0.7
        total_epoch: *epoch_num
    - MakeShrinkMap:
        shrink_ratio: 0.4
        min_text_size: 8
        total_epoch: *epoch_num
    - NormalizeImage:
        scale: 1./255.
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - threshold_map
        - threshold_mask
        - shrink_map
        - shrink_mask
  loader:
    shuffle: true
    drop_last: false
    batch_size_per_card: 16
    num_workers: 8

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/LED_Thermostat_Digits_det/
    label_file_list:
      - ./data/LED_Thermostat_Digits_det/val.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - DetResizeForTest:
    - NormalizeImage:
        scale: 1./255.
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - shape
        - polys
        - ignore_tags
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 1
    num_workers: 2
profiler_options: null

```



尝试中......



3. **启动训练**

```bash
python tools/train.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml -o Global.pretrained_model=./pretrain_models/ch_PP-OCRv4_det_train/best_accuracy
```



#### 5.1.2 验证效果



正在训练中......

![Imgur](https://i.imgur.com/nX87TVD.jpg)









### 5.2 模型导出和推理

















## 6. 总结与展望

- 模型微调

在深度学习项目中，模型微调是一个关键步骤，它允许我们根据特定需求调整模型性能。在PaddleOCR中，通过编辑YML配置文件，用户可以灵活地定制训练参数和模型结构。这包括但不限于调整词典、学习率策略、数据集选择。此外，还可以修改网络的骨干结构或者设计自己的网络架构，并将其集成到PaddleOCR框架中进行实验和训练。这种高度的可配置性使得PaddleOCR不仅适用于通用OCR任务，也能满足更为特定的应用需求。

- 部署

部署阶段是将训练好的模型应用于实际场景中的关键步骤。在PaddleOCR中，生成的识别inference模型可以与检测inference模型结合使用，从而创建一个完整的OCR系统。为了简化部署过程和提高模型在生产环境中的表现，推荐使用Fastdeploy工具。Fastdeploy不仅支持多种平台和设备，还提供了优化算法以保证高效运行，使得模型部署变得快速且高效。

- 展望

展望未来，PaddleOCR将继续扩展其模型库，提供满足各种OCR需求的丰富模型选择。包括SVTR_Tiny和最新的PP-OCRv4等模型。这些模型在提高精确度和效率方面具有很大的潜力，适用于资源丰富或资源受限的环境。各种模型的微调过程大体一致，[YML配置的细微差别](ppocr_yml.html)还需要进一步探索。未来的工作不仅包括试验这些新模型，还包括持续完善训练和部署过程，以在不同的应用场景中达到最优性能。

通过不断探索和试验这些先进模型及其调整方法，我们可以更好地理解和利用OCR技术的潜力，适应技术需求和业务场景的不断变化。
