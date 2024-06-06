# PaddleOCR模型配置文件YML分析

## ch_PP-OCRv3_rec_distillation.yml

```yaml
Global:
  debug: false                       # 是否开启调试模式。当设为true时，可能会打印更多的调试信息，有助于开发或排错。
  use_gpu: true                      # 是否使用GPU进行训练。GPU可以显著加速模型的训练。
  epoch_num: 800                     # 训练的总轮次。每个epoch都会遍历一次全部数据。
  log_smooth_window: 20              # 日志平滑窗口大小，用于计算平滑过的指标，如平均损失等。
  print_batch_step: 10               # 每训练多少个batch打印一次日志信息。
  save_model_dir: ./output/rec_ppocr_v3_distillation  # 模型保存路径。
  save_epoch_step: 3                 # 每训练多少epoch保存一次模型。
  eval_batch_step: [0, 2000]         # 在第0个epoch后，每2000个batch进行一次评估。
  cal_metric_during_train: true      # 是否在训练过程中计算评估指标。
  pretrained_model:                  # 预训练模型的路径，如果提供，训练将从这个模型开始。
  checkpoints:                       # 用于保存训练中间状态的检查点路径。
  save_inference_dir:                # 推断模型的保存路径。
  use_visualdl: false                # 是否使用VisualDL进行训练过程的可视化。
  infer_img: doc/imgs_words/ch/word_1.jpg  # 推断时使用的图片路径。
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt  # 字符字典路径，定义了模型输出层的字符集。
  max_text_length: &max_text_length 25  # 最大文本长度，模型预测结果的最大字符数。
  infer_mode: false                  # 是否为推断模式，设置为true时，模型将只进行推断，不训练。
  use_space_char: true               # 是否在字符字典中包含空格字符。
  distributed: true                  # 是否使用分布式训练，适用于多GPU环境。
  save_res_path: ./output/rec/predicts_ppocrv3_distillation.txt  # 保存推断结果的路径。
  d2s_train_image_shape: [3, 48, -1] # 训练时图像的形状，3代表颜色通道数，48是高度，-1代表宽度由模型根据实际情况确定。


### 优化器
Optimizer:
  name: Adam                     # 使用的优化器名称，这里是Adam优化器，广泛用于深度学习训练中。
  beta1: 0.9                     # Adam优化器的beta1参数，用于计算梯度的一阶矩估计的指数衰减率。
  beta2: 0.999                   # Adam优化器的beta2参数，用于计算梯度的二阶矩估计的指数衰减率。
  lr:
    name: Piecewise              # 学习率调整策略，Piecewise表示分段常数学习率。
    decay_epochs: [700]          # 学习率衰减的时刻，这里在第700个epoch时降低学习率。
    values: [0.0005, 0.00005]    # 对应衰减时刻的学习率值，训练开始时是0.0005，第700个epoch后变为0.00005。
    warmup_epoch: 5              # 学习率预热的epoch数，训练初期学习率会逐渐增加到初始设定值。
  regularizer:
    name: L2                     # 使用的正则化类型，L2正则化有助于防止模型过拟合。
    factor: 3.0e-05              # L2正则化的系数，控制正则化项的影响程度。


### 模型架构
Architecture:
  model_type: &model_type "rec"           # 模型类型，这里是"rec"，PaddelOCR识别模型。
  name: DistillationModel                 # 模型名称。
  algorithm: Distillation                 # 使用的算法，这里是知识蒸馏。
  Models:
    Teacher:
      pretrained:                         # 教师模型的预训练模型路径。
      freeze_params: false                # 是否冻结教师模型的参数，false表示训练过程中参数可以更新。
      return_all_feats: true              # 是否返回教师模型的所有特征，用于蒸馏过程。
      model_type: *model_type             # 指向上文定义的模型类型"rec"。
      algorithm: SVTR_LCNet               # 教师模型使用的算法。
      Transform:                          # 转换操作，具体操作未定义。
      Backbone:
        name: MobileNetV1Enhance          # 使用的骨干网络名称，加强版的MobileNetV1。
        scale: 0.5                        # 骨干网络的规模缩放因子。
        last_conv_stride: [1, 2]          # 最后一个卷积层的步长。
        last_pool_type: avg               # 最后一个池化层的类型，这里是平均池化。
        last_pool_kernel_size: [2, 2]     # 最后一个池化层的核大小。
      Head:
        name: MultiHead                   # 多头部，可以包含多种不同的输出层。
        head_list:                        # 头部列表。
          - CTCHead:                      # CTC头部，用于字符级别的识别。
              Neck:
                name: svtr                # Neck部分的名称，可能是一种特定的特征转换模块。
                dims: 64                  # 特征维度。
                depth: 2                  # 模块深度。
                hidden_dims: 120          # 隐藏层维度。
                use_guide: True           # 是否使用引导。
              Head:
                fc_decay: 0.00001         # 全连接层的衰减因子。
          - SARHead:                      # SAR头部，用于序列识别。
              enc_dim: 512                # 编码器维度。
              max_text_length: *max_text_length  # 最大文本长度，引用上文定义的25。

    Student:
      pretrained:                         # 学生模型的预训练模型路径。
      freeze_params: false                # 是否冻结学生模型的参数，false表示训练过程中参数可以更新。
      return_all_feats: true              # 是否返回学生模型的所有特征，用于蒸馏过程中与教师模型的特征对比。
      model_type: *model_type             # 指向上文定义的模型类型"rec"。
      algorithm: SVTR_LCNet               # 学生模型使用的算法。
      Transform:                          # 转换操作，具体操作未定义。
      Backbone:
        name: MobileNetV1Enhance          # 使用的骨干网络名称，加强版的MobileNetV1。
        scale: 0.5                        # 骨干网络的规模缩放因子。
        last_conv_stride: [1, 2]          # 最后一个卷积层的步长。
        last_pool_type: avg               # 最后一个池化层的类型，这里是平均池化。
        last_pool_kernel_size: [2, 2]     # 最后一个池化层的核大小。
      Head:
        name: MultiHead                   # 多头部，可以包含多种不同的输出层。
        head_list:                        # 头部列表。
          - CTCHead:                      # CTC头部，用于字符级别的识别。
              Neck:
                name: svtr                # Neck部分的名称，可能是一种特定的特征转换模块。
                dims: 64                  # 特征维度。
                depth: 2                  # 模块深度。
                hidden_dims: 120          # 隐藏层维度。
                use_guide: True           # 是否使用引导。
              Head:
                fc_decay: 0.00001         # 全连接层的衰减因子。
          - SARHead:                      # SAR头部，用于序列识别。
              enc_dim: 512                # 编码器维度。
              max_text_length: *max_text_length  # 最大文本长度，引用上文定义的25。


### 损失函数
Loss:
  name: CombinedLoss           # 使用的总损失函数的名称，这里是组合损失。
  loss_config_list:            # 损失函数配置列表。
  - DistillationDMLLoss:       # 蒸馏互信息损失，用于教师和学生模型的输出对齐。
      weight: 1.0              # 该损失的权重。
      act: "softmax"           # 激活函数，用于在计算损失前处理输出。
      use_log: true            # 是否在损失计算前对输出应用对数变换。
      model_name_pairs:        # 模型对，指明哪些模型的输出需要对比。
      - ["Student", "Teacher"]
      key: head_out            # 指定哪个输出（head_out表示来自模型头部的输出）。
      multi_head: True         # 是否有多个头部输出。
      dis_head: ctc            # 指定具体哪个头部，这里是CTC头。
      name: dml_ctc            # 损失函数的名称。
  - DistillationDMLLoss:       # 另一个蒸馏互信息损失，针对不同的模型头。
      weight: 0.5
      act: "softmax"
      use_log: true
      model_name_pairs:
      - ["Student", "Teacher"]
      key: head_out
      multi_head: True
      dis_head: sar            # 这里针对的是SAR头部。
      name: dml_sar
  - DistillationDistanceLoss:  # 蒸馏距离损失，通常用于最小化教师和学生模型特征之间的距离。
      weight: 1.0
      mode: "l2"               # 使用L2范数作为距离度量。
      model_name_pairs:
      - ["Student", "Teacher"]
      key: backbone_out        # 指定来自模型骨干网络的输出。
  - DistillationCTCLoss:       # 蒸馏CTC损失，专门用于处理CTC输出的对齐。
      weight: 1.0
      model_name_list: ["Student", "Teacher"]
      key: head_out
      multi_head: True
  - DistillationSARLoss:       # 蒸馏SAR损失，专门用于处理SAR输出的对齐。
      weight: 1.0
      model_name_list: ["Student", "Teacher"]
      key: head_out
      multi_head: True


### 后处理
PostProcess:
  name: DistillationCTCLabelDecode    # 后处理步骤的名称，用于解码CTC标签。
  model_name: ["Student", "Teacher"]  # 指定这个后处理步骤应用于哪些模型。
  key: head_out                       # 指明从模型的哪个输出进行处理，这里是头部输出。
  multi_head: True                    # 表明模型包含多个头部。


### 评估指标
Metric:
  name: DistillationMetric            # 评估指标的名称，这里特别用于蒸馏训练。
  base_metric_name: RecMetric         # 基础评估指标的名称，RecMetric可能是一个专门针对识别任务的评估指标。
  main_indicator: acc                 # 主要评估指标，这里是准确率（accuracy）。
  key: "Student"                      # 指定主要关注哪个模型的评估结果，这里关注学生模型。
  ignore_space: False                 # 是否在评估时忽略空格字符，这里选择不忽略。


### 训练数据集
Train:
  dataset:
    name: SimpleDataSet                  # 数据集的名称。
    data_dir: ./train_data/              # 数据集所在的目录。
    ext_op_transform_idx: 1              # 扩展操作转换索引，可能用于指定特定转换操作的顺序或位置。
    label_file_list:                     # 包含标签的文件列表。
    - ./train_data/train_list.txt
    transforms:                          # 应用于数据的转换列表。
    - DecodeImage:                       # 解码图像文件。
        img_mode: BGR                    # 图像模式，BGR表示使用BGR颜色空间。
        channel_first: false             # 是否将通道置于维度的首位，这里为false，因此数据格式为HWC（高、宽、通道）。
    - RecConAug:                         # 识别内容增强，用于数据增强。
        prob: 0.5                        # 应用此增强的概率。
        ext_data_num: 2                  # 额外数据数量，可能用于生成更多训练样本。
        image_shape: [48, 320, 3]        # 图像的尺寸。
        max_text_length: *max_text_length  # 最大文本长度，引用上文定义的25。
    - RecAug:                            # 通用的识别增强。
    - MultiLabelEncode:                  # 多标签编码。
    - RecResizeImg:                      # 调整图像大小。
        image_shape: [3, 48, 320]        # 调整后的图像尺寸。
    - KeepKeys:                          # 指定保留的关键字段。
        keep_keys:
        - image                          # 保留图像数据。
        - label_ctc                      # 保留CTC标签。
        - label_sar                      # 保留SAR标签。
        - length                         # 保留文本长度。
        - valid_ratio                    # 保留有效比率。

# 数据加载器（Loader）配置
  loader:
    shuffle: true                        # 是否在每个epoch开始时打乱数据。
    batch_size_per_card: 128             # 每个GPU的批量大小。
    drop_last: true                      # 是否丢弃最后一个不完整的batch，通常用于确保所有的batch大小一致。
    num_workers: 4                       # 用于数据加载的工作线程数量。


### 评估数据集
Eval:
  dataset:
    name: SimpleDataSet                  # 数据集的名称，这里使用与训练相同的数据集类型。
    data_dir: ./train_data               # 数据集所在目录。
    label_file_list:                     # 包含标签的文件列表，这里指向验证集。
    - ./train_data/val_list.txt
    transforms:                          # 应用于数据的转换列表。
    - DecodeImage:                       # 解码图像文件。
        img_mode: BGR                    # 图像模式，BGR表示使用BGR颜色空间。
        channel_first: false             # 是否将通道置于维度的首位，这里为false，因此数据格式为HWC（高、宽、通道）。
    - MultiLabelEncode:                  # 多标签编码，为不同的输出标签格式编码。
    - RecResizeImg:                      # 调整图像大小。
        image_shape: [3, 48, 320]        # 调整后的图像尺寸。
    - KeepKeys:                          # 指定保留的关键字段。
        keep_keys:
        - image                          # 保留图像数据。
        - label_ctc                      # 保留CTC标签。
        - label_sar                      # 保留SAR标签。
        - length                         # 保留文本长度。
        - valid_ratio                    # 保留有效比率。

# 数据加载器（Loader）配置
  loader:
    shuffle: false                       # 是否在每个epoch开始时打乱数据，评估时通常不打乱。
    drop_last: false                     # 是否丢弃最后一个不完整的batch，评估时通常保留，以评估所有数据。
    batch_size_per_card: 128             # 每个GPU的批量大小。
    num_workers: 4                       # 用于数据加载的工作线程数量。
```

## ch_PP-OCRv4_rec_distill.yml

```yaml
Global:
  debug: false                            # 是否开启调试模式，false表示不开启。
  use_gpu: true                           # 是否使用GPU进行训练，true表示使用。
  epoch_num: 200                          # 训练的总轮次。
  log_smooth_window: 20                   # 日志平滑窗口大小，用于调整日志输出的平滑度。
  print_batch_step: 10                    # 每隔多少批次输出一次训练日志。
  save_model_dir: ./output/rec_dkd_400w_svtr_ctc_lcnet_blank_dkd0.1/  # 模型保存目录。
  save_epoch_step: 40                     # 每隔多少轮次保存一次模型。
  eval_batch_step: [0, 2000]              # 在哪些批次上进行评估。
  cal_metric_during_train: true           # 是否在训练过程中计算评估指标。
  pretrained_model: null                  # 预训练模型的路径，null表示不使用预训练模型。
  checkpoints: ./output/rec_dkd_400w_svtr_ctc_lcnet_blank_dkd0.1/latest  # 检查点路径。
  save_inference_dir: null                # 推理模型的保存路径，null表示不保存。
  use_visualdl: false                     # 是否使用VisualDL进行可视化，false表示不使用。
  infer_img: doc/imgs_words/ch/word_1.jpg # 推理时使用的图像文件路径。
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt  # 字符字典路径。
  max_text_length: 25                     # 最大文本长度。
  infer_mode: false                       # 是否为推理模式，false表示不是。
  use_space_char: true                    # 是否使用空格字符。
  distributed: true                       # 是否使用分布式训练，true表示使用。
  save_res_path: ./output/rec/predicts_ppocrv3.txt   # 保存结果的路径。


### 优化器
Optimizer:
  name: Adam               # 使用的优化器类型，这里是Adam。
  beta1: 0.9               # 优化器的beta1参数，用于计算梯度的指数移动平均，通常用来控制梯度的平滑程度。
  beta2: 0.999             # 优化器的beta2参数，用于计算梯度平方的指数移动平均，帮助调整学习步长。
  lr:
    name: Cosine           # 学习率调整策略，这里使用的是余弦退火策略。
    learning_rate: 0.001   # 初始学习率。
    warmup_epoch: 2        # 热身周期数，在此期间学习率会逐渐增加到初始设定值。
  regularizer:
    name: L2               # 正则化类型，这里是L2正则化，常用于防止模型过拟合。
    factor: 3.0e-05        # 正则化因子，决定了正则化的强度。


### 模型架构
Architecture:
  model_type: rec                         # 模型类型，这里是recognition。
  name: DistillationModel                 # 模型名称。
  algorithm: Distillation                 # 使用的算法，这里是蒸馏学习。
  Models:
    Teacher:
      pretrained: null                    # 教师模型的预训练模型路径，null表示未使用。
      freeze_params: true                 # 是否冻结教师模型的参数。
      return_all_feats: true              # 是否返回所有特征。
      model_type: rec                     # 教师模型类型。
      algorithm: SVTR                     # 教师模型使用的算法。
      Transform: null                     # 转换层，null表示未使用。
      Backbone:
        name: SVTRNet                     # 主干网络名称。
        img_size: [48, 320]               # 输入图像尺寸。
        out_char_num: 40                  # 输出字符数量。
        out_channels: 192                 # 输出通道数。
        patch_merging: Conv               # 合并补丁的方式。
        embed_dim: [64, 128, 256]         # 嵌入维度。
        depth: [3, 6, 3]                  # 各层深度。
        num_heads: [2, 4, 8]              # 头的数量。
        mixer: [Conv, Conv, Conv, Conv, Conv, Conv, Global, Global, Global, Global, Global, Global]  # 混合方式。
        local_mixer: [[5, 5], [5, 5], [5, 5]]  # 本地混合尺寸。
        last_stage: false                 # 是否为最后阶段。
        prenorm: true                     # 是否使用预归一化。
      Head:
        name: MultiHead                   # 头部名称。
        head_list:                        # 头部列表。
          - CTCHead:                      # CTC头部。
              Neck:
                name: svtr                # 颈部名称。
                dims: 120                 # 维度。
                depth: 2                  # 深度。
                hidden_dims: 120          # 隐藏层维度。
                kernel_size: [1, 3]       # 核大小。
                use_guide: True           # 是否使用引导。
              Head:
                fc_decay: 0.00001         # 全连接层衰减。
          - NRTRHead:                     # NRTR头部。
              nrtr_dim: 384               # NRTR维度。
              max_text_length: 25         # 最大文本长度，从全局配置引用。

    Student:
      pretrained: null                    # 学生模型的预训练模型路径，null表示未使用。
      freeze_params: false                # 是否冻结学生模型的参数，false表示不冻结，模型参数会在训练中更新。
      return_all_feats: true              # 是否返回所有特征。
      model_type: rec                     # 学生模型类型。
      algorithm: SVTR                     # 学生模型使用的算法。
      Transform: null                     # 转换层，null表示未使用。
      Backbone:
        name: PPLCNetV3                   # 主干网络名称。
        scale: 0.95                       # 网络缩放因子，影响模型大小和计算复杂度。
      Head:
        name: MultiHead                   # 头部名称。
        head_list:                        # 头部列表。
          - CTCHead:                      # CTC头部。
              Neck:
                name: svtr                # 颈部名称。
                dims: 120                 # 维度。
                depth: 2                  # 深度。
                hidden_dims: 120          # 隐藏层维度。
                kernel_size: [1, 3]       # 核大小。
                use_guide: True           # 是否使用引导。
              Head:
                fc_decay: 0.00001         # 全连接层衰减。
          - NRTRHead:                     # NRTR头部。
              nrtr_dim: 384               # NRTR维度。
              max_text_length: 25         # 最大文本长度，从全局配置引用。


### 损失函数
Loss:
  name: CombinedLoss                      # 使用组合损失。
  loss_config_list:
  - DistillationDKDLoss:                  # 蒸馏距离知识损失。
      weight: 0.1                         # 该损失的权重。
      model_name_pairs:                   # 模型对，指定损失计算涉及的模型。
      - - Student
        - Teacher
      key: head_out                       # 指定损失计算的模型输出键名。
      multi_head: true                    # 表示模型具有多个输出头。
      alpha: 1.0                          # 损失计算的alpha参数。
      beta: 2.0                           # 损失计算的beta参数。
      dis_head: gtc                       # 指定在损失计算中使用的输出头类型。
      name: dkd                           # 损失函数的名称。
  - DistillationCTCLoss:                  # 蒸馏CTC损失。
      weight: 1.0                         # 损失权重。
      model_name_list:                    # 涉及的模型。
      - Student
      key: head_out
      multi_head: true
  - DistillationNRTRLoss:                 # 蒸馏NRTR损失。
      weight: 1.0
      smoothing: false                    # 是否应用标签平滑。
      model_name_list:
      - Student
      key: head_out
      multi_head: true
  - DistillCTCLogits:                     # CTC逻辑损失蒸馏。
      weight: 1.0
      reduction: mean                     # 损失的缩减方法。
      model_name_pairs:
      - - Student
        - Teacher
      key: head_out


### 后处理
PostProcess:
  name: DistillationCTCLabelDecode     # 后处理步骤的名称。
  model_name:
  - Student                            # 指定这个后处理步骤应用于哪些模型，这里是学生模型。
  key: head_out                        # 指明从模型的哪个输出进行处理，这里是头部输出。
  multi_head: true                     # 表明模型包含多个头部。



### 评估指标
Metric:
  name: DistillationMetric             # 评估指标的名称，专为蒸馏训练设计。
  base_metric_name: RecMetric          # 基础评估指标的名称，RecMetric通常是一个专门针对识别任务的评估指标。
  main_indicator: acc                  # 主要评估指标，这里是准确率（accuracy）。
  key: Student                         # 指定主要关注哪个模型的评估结果，这里关注学生模型。
  ignore_space: false                  # 是否在评估时忽略空格字符，这里选择不忽略。



### 训练数据集
Train:
  dataset:
    name: SimpleDataSet                # 数据集的名称。
    data_dir: ./train_data/            # 数据集所在的目录。
    label_file_list:                   # 包含标签的文件列表。
    - ./train_data/train_list.txt
    ratio_list:                        # 数据集中每个部分的比例，这里设置为1.0表示全部使用。
    - 1.0
    transforms:                        # 应用于数据的转换列表。
    - DecodeImage:                     # 解码图像文件。
        img_mode: BGR                  # 图像模式，BGR表示使用BGR颜色空间。
        channel_first: false           # 是否将通道置于维度的首位，这里为false，因此数据格式为HWC（高、宽、通道）。
    - RecAug:                          # 识别增强，用于数据增强。
    - MultiLabelEncode:                # 多标签编码，为不同的输出标签格式编码。
        gtc_encode: NRTRLabelEncode    # 使用NRTR标签编码方式。
    - KeepKeys:                        # 指定保留的关键字段。
        keep_keys:
        - image                        # 保留图像数据。
        - label_ctc                    # 保留CTC标签。
        - label_gtc                    # 保留GTC标签，可能用于其他任务或验证。
        - length                       # 保留文本长度。
        - valid_ratio                  # 保留有效比率。

# 数据加载器（Loader）配置
  loader:
    shuffle: true                      # 是否在每个epoch开始时打乱数据。
    batch_size_per_card: 128           # 每个GPU的批量大小。
    drop_last: true                    # 是否丢弃最后一个不完整的batch，通常用于确保所有的batch大小一致。
    num_workers: 8                     # 用于数据加载的工作线程数量。
    use_shared_memory: true            # 是否使用共享内存加速数据加载。


### 评估数据集
Eval:
  dataset:
    name: SimpleDataSet                # 数据集的名称。
    data_dir: ./train_data             # 数据集所在目录。
    label_file_list:                   # 包含标签的文件列表，这里指向验证集。
    - ./train_data/val_list.txt
    transforms:                        # 应用于数据的转换列表。
    - DecodeImage:                     # 解码图像文件。
        img_mode: BGR                  # 图像模式，BGR表示使用BGR颜色空间。
        channel_first: false           # 是否将通道置于维度的首位，这里为false，因此数据格式为HWC（高、宽、通道）。
    - MultiLabelEncode:                # 多标签编码，为不同的输出标签格式编码。
        gtc_encode: NRTRLabelEncode    # 使用NRTR标签编码方式。
    - RecResizeImg:                    # 调整图像大小。
        image_shape: [3, 48, 320]      # 调整后的图像尺寸。
    - KeepKeys:                        # 指定保留的关键字段。
        keep_keys:
        - image                        # 保留图像数据。
        - label_ctc                    # 保留CTC标签。
        - label_gtc                    # 保留GTC标签，可能用于其他任务或验证。
        - length                       # 保留文本长度。
        - valid_ratio                  # 保留有效比率。

# 数据加载器（Loader）配置
  loader:
    shuffle: false                     # 是否在每个epoch开始时打乱数据，评估时通常不打乱。
    drop_last: false                   # 是否丢弃最后一个不完整的batch，评估时通常保留，以评估所有数据。
    batch_size_per_card: 128           # 每个GPU的批量大小。
    num_workers: 4                     # 用于数据加载的工作线程数量。



profiler_options: null               # 配置文件中可能包含对性能分析工具的设置，这里为null表示未使用。
```

## PP-OCRv_rec 与 PP-OCRv4_rec 对比

### Global 配置对比

| **配置项**                  | **v3版本详细配置**                             | **v4版本详细配置**                                           |
| --------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| **debug**                   | false                                          | false                                                        |
| **use_gpu**                 | true                                           | true                                                         |
| **epoch_num**               | 800                                            | <span style="color:red">200</span>                           |
| **log_smooth_window**       | 20                                             | 20                                                           |
| **print_batch_step**        | 10                                             | 10                                                           |
| **save_model_dir**          | ./output/rec_ppocr_v3_distillation             | <span style="color:red">./output/rec_dkd_400w_svtr_ctc_lcnet_blank_dkd0.1/</span> |
| **save_epoch_step**         | 3                                              | <span style="color:red">40</span>                            |
| **eval_batch_step**         | [0, 2000]                                      | [0, 2000]                                                    |
| **cal_metric_during_train** | true                                           | true                                                         |
| **pretrained_model**        | (未设置)                                       | <span style="color:red">null</span>                          |
| **checkpoints**             | (未设置)                                       | <span style="color:red">./output/rec_dkd_400w_svtr_ctc_lcnet_blank_dkd0.1/latest</span> |
| **save_inference_dir**      | (未设置)                                       | <span style="color:red">null</span>                          |
| **use_visualdl**            | false                                          | false                                                        |
| **infer_img**               | doc/imgs_words/ch/word_1.jpg                   | doc/imgs_words/ch/word_1.jpg                                 |
| **character_dict_path**     | ppocr/utils/ppocr_keys_v1.txt                  | ppocr/utils/ppocr_keys_v1.txt                                |
| **max_text_length**         | 25 (通过 &max_text_length 引用)                | 25                                                           |
| **infer_mode**              | false                                          | false                                                        |
| **use_space_char**          | true                                           | true                                                         |
| **distributed**             | true                                           | true                                                         |
| **save_res_path**           | ./output/rec/predicts_ppocrv3_distillation.txt | <span style="color:red">./output/rec/predicts_ppocrv3.txt</span> |
| **d2s_train_image_shape**   | [3, 48, -1]                                    | <span style="color:red">(未设置)</span>                      |

### Optimizer 配置对比

| **配置项**               | **v3版本详细配置** | **v4版本详细配置**                            |
| ------------------------ | ------------------ | --------------------------------------------- |
| **name**                 | Adam               | Adam                                          |
| **beta1**                | 0.9                | 0.9                                           |
| **beta2**                | 0.999              | 0.999                                         |
| **lr → name**            | Piecewise          | <span style="color:red">Cosine</span>         |
| **lr → values**          | [0.0005, 0.00005]  | <span style="color:red">0.001</span>          |
| **lr → decay_epochs**    | [700]              | <span style="color:red">(未使用此设置)</span> |
| **lr → warmup_epoch**    | 5                  | <span style="color:red">2</span>              |
| **regularizer → name**   | L2                 | L2                                            |
| **regularizer → factor** | 3.0e-05            | 3.0e-05                                       |

### Architecture 配置对比

| 配置项                  | v3版本                                                       | v4版本                                                       |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **模型类型**            | rec                                                          | rec                                                          |
| **模型名称**            | DistillationModel                                            | DistillationModel                                            |
| **算法**                | Distillation                                                 | Distillation                                                 |
| **Teacher模型**         | - MobileNetV1Enhance (Backbone)<br>- MultiHead (Head)        | <span style="color:red">- SVTRNet (Backbone)</span><br>- MultiHead (Head) |
| **Teacher模型配置细节** | - scale: 0.5<br>- last_conv_stride: [1, 2]<br>- last_pool_type: avg<br>- last_pool_kernel_size: [2, 2] | <span style="color:red">- img_size: [48, 320]<br>- out_char_num: 40<br>- out_channels: 192<br>- patch_merging: Conv<br>- embed_dim: [64, 128, 256]<br>- depth: [3, 6, 3]<br>- num_heads: [2, 4, 8]<br>- mixer: Various Conv and Global<br>- local_mixer: [5, 5] across stages<br>- prenorm: true</span> |
| **Student模型**         | - MobileNetV1Enhance (Backbone)<br>- MultiHead (Head)        | <span style="color:red">- PPLCNetV3 (Backbone)</span><br>- MultiHead (Head) |
| **Student模型配置细节** | - scale: 0.5<br>- last_conv_stride: [1, 2]<br>- last_pool_type: avg<br>- last_pool_kernel_size: [2, 2] | <span style="color:red">- scale: 0.95</span>                 |
| **Head配置**            | - CTCHead (MultiHead的一部分)                                | - CTCHead (MultiHead的一部分)<br><span style="color:red">- NRTRHead </span> |

#### 关键变化点

1. **Teacher模型Backbone**：
   - **v3**：使用了MobileNetV1Enhance，参数较简单，主要关注scale和pooling操作。
   - **v4**：改为使用SVTRNet，这是一个基于Transformer的复杂网络，具有多种深度和维度的嵌入层，显示了对模型结构的显著扩展和深化。
2. **Student模型Backbone**：
   - **v3**：与Teacher相同，使用MobileNetV1Enhance。
   - **v4**：使用PPLCNetV3，不仅模型类型不同，还提高了scale参数，表明v4在学生模型的计算效率和容量上进行了优化。
3. **Head配置**：
   - **v3**：仅使用了CTCHead作为模型的输出部分。
   - **v4**：除了保留CTCHead，还新增了NRTRHead，这表明v4版本增加了模型处理复杂文本任务的能力。

### Loss 配置对比

| **配置项**                   | **v3版本详细配置**                                           | **v4版本详细配置**                                           |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **name**                     | CombinedLoss                                                 | CombinedLoss                                                 |
| **DistillationDMLLoss**      | - weight: 1.0, act: "softmax", use_log: true, model_name_pairs: [["Student", "Teacher"]], key: head_out, multi_head: True, dis_head: ctc, name: dml_ctc | <span style="color:red">(未使用)</span>                      |
|                              | - weight: 0.5, act: "softmax", use_log: true, model_name_pairs: [["Student", "Teacher"]], key: head_out, multi_head: True, dis_head: sar, name: dml_sar | <span style="color:red">(未使用)</span>                      |
| **DistillationDistanceLoss** | - weight: 1.0, mode: "l2", model_name_pairs: [["Student", "Teacher"]], key: backbone_out | <span style="color:red">(未使用)</span>                      |
| **DistillationCTCLoss**      | - weight: 1.0, model_name_list: ["Student", "Teacher"], key: head_out, multi_head: True | - weight: 1.0, model_name_list: ["Student"], key: head_out, multi_head: True |
| **DistillationSARLoss**      | - weight: 1.0, model_name_list: ["Student", "Teacher"], key: head_out, multi_head: True | <span style="color:red">(未使用)</span>                      |
| **DistillCTCLogits**         | <span style="color:red">(未使用)</span>                      | - weight: 1.0, reduction: mean, model_name_pairs: [["Student", "Teacher"]], key: head_out |
| **DistillationDKDLoss**      | <span style="color:red">(未使用)</span>                      | - weight: 0.1, model_name_pairs: [["Student", "Teacher"]], key: head_out, multi_head: true, alpha: 1.0, beta: 2.0, dis_head: gtc, name: dkd |
| **DistillationNRTRLoss**     | <span style="color:red">(未使用)</span>                      | - weight: 1.0, model_name_list: ["Student"], key: head_out, multi_head: true, smoothing: false |

### PostProcess 配置对比

| **配置项**     | **v3版本详细配置**         | **v4版本详细配置**                         |
| -------------- | -------------------------- | ------------------------------------------ |
| **name**       | DistillationCTCLabelDecode | DistillationCTCLabelDecode                 |
| **model_name** | ["Student", "Teacher"]     | <span style="color:red">["Student"]</span> |
| **key**        | head_out                   | head_out                                   |
| **multi_head** | true                       | true                                       |

### Metric 配置对比

| **配置项**           | **v3版本详细配置** | **v4版本详细配置** |
| -------------------- | ------------------ | ------------------ |
| **name**             | DistillationMetric | DistillationMetric |
| **base_metric_name** | RecMetric          | RecMetric          |
| **main_indicator**   | acc                | acc                |
| **key**              | "Student"          | "Student"          |
| **ignore_space**     | false              | false              |

### Train 配置对比

| **配置项**                         | **v3版本详细配置**                                           | **v4版本详细配置**                                           |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **dataset → name**                 | SimpleDataSet                                                | SimpleDataSet                                                |
| **dataset → data_dir**             | ./train_data/                                                | ./train_data/                                                |
| **dataset → ext_op_transform_idx** | 1                                                            | <span style="color:red">(未设置)</span>                      |
| **dataset → label_file_list**      | - ./train_data/train_list.txt                                | - ./train_data/train_list.txt                                |
| **dataset → ratio_list**           | <span style="color:red">(未设置)</span>                      | - 1.0                                                        |
| **dataset → transforms**           | - DecodeImage: { img_mode: BGR, channel_first: false }<br> - RecConAug: { prob: 0.5, ext_data_num: 2, image_shape: [48, 320, 3], max_text_length: 25 }<br> - RecAug<br> - MultiLabelEncode<br> - RecResizeImg: { image_shape: [3, 48, 320] }<br> - KeepKeys: { keep_keys: [image, label_ctc, label_sar, length, valid_ratio] } | - DecodeImage: { img_mode: BGR, channel_first: false }<br> - RecAug<br><span style="color:red">- MultiLabelEncode: { gtc_encode: NRTRLabelEncode }</span>  <br> - RecResizeImg: { image_shape: [3, 48, 320] }<br> - KeepKeys: { keep_keys: [image, label_ctc, label_gtc, length, valid_ratio] } |
| **loader → shuffle**               | true                                                         | true                                                         |
| **loader → batch_size_per_card**   | 128                                                          | 128                                                          |
| **loader → drop_last**             | true                                                         | true                                                         |
| **loader → num_workers**           | 4                                                            | <span style="color:red">8</span>                             |
| **loader → use_shared_memory**     | <span style="color:red">(未设置)</span>                      | <span style="color:red">true</span>                          |

### Eval 配置对比

| **配置项**                       | **v3版本详细配置**                                           | **v4版本详细配置**                                           |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **dataset → name**               | SimpleDataSet                                                | SimpleDataSet                                                |
| **dataset → data_dir**           | ./train_data                                                 | ./train_data                                                 |
| **dataset → label_file_list**    | - ./train_data/val_list.txt                                  | - ./train_data/val_list.txt                                  |
| **dataset → transforms**         | - DecodeImage: { img_mode: BGR, channel_first: false }<br> - MultiLabelEncode<br> - RecResizeImg: { image_shape: [3, 48, 320] }<br> - KeepKeys: { keep_keys: [image, label_ctc, label_sar, length, valid_ratio] } | - DecodeImage: { img_mode: BGR, channel_first: false }<br><span style="color:red">- MultiLabelEncode: { gtc_encode: NRTRLabelEncode }</span>  <br/> - RecResizeImg: { image_shape: [3, 48, 320] }<br> - KeepKeys: { keep_keys: [image, label_ctc, label_gtc, length, valid_ratio] } |
| **loader → shuffle**             | false                                                        | false                                                        |
| **loader → drop_last**           | false                                                        | false                                                        |
| **loader → batch_size_per_card** | 128                                                          | 128                                                          |
| **loader → num_workers**         | 4                                                            | 4                                                            |
| **profiler_options**             | <span style="color:red">(未设置)</span>                      | <span style="color:red">null</span>                          |
