# text_renderer 批量生成数据集

## 环境搭建

text_renderer 是一个最后更新于两年前的开源工具，有一些库的版本比较老，以及会出现一些依赖问题，所以有必要专门搭建一个环境来进行使用。

- 创建虚拟环境：

```bash
conda create --name text python=3.8
```

- 激活并进入工作目录

```bash
conda activate text

cd 工作目录
```

- 安装 text_renderer

```bash
conda install git

git clone https://github.com/oh-my-ocr/text_renderer.git
```

>  git clone 失败请[下载](https://pan.baidu.com/s/1YKwpNdUd2vJvms7I4Eiphw?pwd=dba5)解压到工作目录下，记得删掉 master 后缀
>
> text_renderer/docker/requirements.txt 中 opencv-python==3.4.5.20 改为 3.4.8.29 ，否则接下来会报错：
>
> ```bash
> ERROR: Ignored the following yanked versions: 3.4.9.31, 3.4.10.35, 3.4.11.39
> ERROR: Could not find a version that satisfies the requirement opencv-python==3.4.5.20 (from versions: 3.4.0.14, 3.4.8.29, 3.4.9.33, 3.4.10.37, 3.4.11.41, 3.4.11.43, 3.4.11.45, 3.4.13.47, 3.4.14.51, 3.4.14.53, 3.4.15.55, 3.4.16.57, 3.4.16.59, 3.4.17.61, 3.4.17.63, 3.4.18.65, 4.1.2.30, 4.2.0.32, 4.2.0.34, 4.3.0.36, 4.3.0.38, 4.4.0.40, 4.4.0.42, 4.4.0.44, 4.4.0.46, 4.5.1.48, 4.5.2.52, 4.5.2.54, 4.5.3.56, 4.5.4.58, 4.5.4.60, 4.5.5.62, 4.5.5.64, 4.6.0.66, 4.7.0.68, 4.7.0.72, 4.8.0.74, 4.8.0.76, 4.8.1.78, 4.9.0.80)
> ERROR: No matching distribution found for opencv-python==3.4.5.20
> ```

```bash
cd text_renderer

python setup.py develop

python -m pip install -r docker/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 接下来降级 numpy和 pandas，否则会出现以下错误：
>
> ```bash
> AttributeError: module 'numpy' has no attribute 'int'.
> `np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
> The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
>     https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
> ```

```bash
pip install numpy==1.19.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install pandas==1.2.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 验证环境

```bash
python main.py --config example_data/example.py --dataset img --num_processes 2 --log_period 10
```

> 没有报错就说明环境搭建成功

- 安装 Jupyter Notebook

> 参考[计算平台和环境配置](dev_setup.html)

## 使用教程

- 字体

网站如 [DaFont](https://www.dafont.com/) , [1001 Fonts](https://www.1001fonts.com/) 或 [Urban Fonts](https://www.urbanfonts.com/) 等提供了多种可用于个人和商业用途的字体

下载后将 ttf 文件复制到 text_renderer/example_data/font/ 下

在 text_renderer/example_data/font_list/ font_list.txt 中修改字体

例如：数码管字体 [digital display tfb.ttf](https://dl.dafont.com/dl/?f=digital_display_tfb)

- 字典

将自定义字典复制到 text_renderer/example_data/char/ 下

例如：温控表自定义字典 digital_dict.txt：

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

- 语料

语料是一个或多个文档的集合，可以包括书籍、文章、对话记录等，旨在提供有关特定语言或语言现象的全面数据

语料 txt 存放于 text_renderer/example_data/text/

例如温控表数码管语料，应该是一系列数字的组合，这些组合用于生成训练数据，帮助模型学习如何识别类似的数码管显示内容，可以使用 python 快速生成从 0000 到 9999 的所有数字。

使用 Jupyter Notebook ，工作路径中新建 TextCreator.ipynb ：

```python
with open("./text_renderer/example_data/text/digital_text.txt", "w") as f:
    for num in range(10000):
        f.write(f"{num:04d}")   #不添加换行，digital display tfb.ttf不支持换行符\n
```

语料 digital_text.txt 如下：

```tex
00000001000200030004000500060007000800090010001100120013001400150016001700180019002000210022002300240025002600270028002900300031003200330034003500360037003800390040004100420043004400450046004700480049005000510052005300540055005600570058005900600061006200630064006500660067006800690070007100720073007400750076007700780079008000810082008300840085008600870088008900900091009200930094009500960097009800990100...9987998899899990999199929993999499959996999799989999
```

- 主程序

参考示例程序 text_renderer/example_data/example.py：

```python
# 导入必要的库
import inspect
import os
from pathlib import Path
import imgaug.augmenters as iaa  # 图像增强库

# 从text_renderer包导入各种效果和配置模块
from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,  # 渲染配置类
    NormPerspectiveTransformCfg,  # 正常视角变换配置
    GeneratorCfg,  # 生成器配置
    FixedTextColorCfg,  # 固定文本颜色配置
)
from text_renderer.layout.same_line import SameLineLayout  # 同行布局
from text_renderer.layout.extra_text_line import ExtraTextLineLayout  # 额外文本行布局

# 定义当前文件所在目录
CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
# 定义输出目录
OUT_DIR = CURRENT_DIR / "output"
# 数据目录
DATA_DIR = CURRENT_DIR
# 背景图片目录
BG_DIR = DATA_DIR / "bg"
# 字典目录
CHAR_DIR = DATA_DIR / "char"
# 字体目录
FONT_DIR = DATA_DIR / "font"
# 字体列表目录
FONT_LIST_DIR = DATA_DIR / "font_list"
# 文本目录
TEXT_DIR = DATA_DIR / "text"

# 字体配置字典
font_cfg = dict(
    font_dir=FONT_DIR,  # 字体文件夹路径
    font_list_file=FONT_LIST_DIR / "font_list.txt",  # 字体列表文件
    font_size=(30, 31),  # 字体大小范围
)

# 创建一个正常视角变换的配置实例
perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)
```

```python
def get_char_corpus():
    """
    创建并返回一个字符语料库对象。这个语料库从指定的文本文件中读取字符。
    """
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],  # 指定中英文本的路径
            filter_by_chars=True,                         # 指定是否按字符过滤
            chars_file=CHAR_DIR / "chn.txt",              # 指定字符文件
            length=(5, 10),                               # 生成文本的长度范围（相同会报错）
            char_spacing=(-0.3, 1.3),                     # 字符间距范围
            **font_cfg                                    # 使用先前定义的字体配置
        ),
    )

def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True
):
    """
    根据提供的参数创建并返回一个生成器配置对象，用于配置图像渲染的各种属性。
    """
    return GeneratorCfg(
        num_image=50,                                     # 生成图像的数量
        save_dir=OUT_DIR / name,                          # 指定保存目录
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,                                # 背景图片目录
            perspective_transform=perspective_transform,  # 使用先前定义的视角变换配置
            gray=gray,                                    # 是否生成灰度图像
            layout_effects=layout_effects,                # 布局效果
            layout=layout,                                # 使用的布局配置
            corpus=corpus,                                # 使用的语料库
            corpus_effects=corpus_effects,                # 语料库效果
        ),
    )
```

```python
# fmt: off
# 配置文件必须包含一个名为 configs 的变量
configs = [
    xxx_data()  # 调用 xxx_data 函数，并将结果添加到配置列表中
]
# fmt: on
```

其中给出了七个生成例子：

```python
def chn_data():
    """
    生成中文文本渲染配置，包括一系列特定的文本效果。
    """
    return base_cfg(
        inspect.currentframe().f_code.co_name,             # 获取当前函数名
        corpus=get_char_corpus(),                          # 获取字符语料库
        corpus_effects=Effects(                            # 定义文本效果
            [
                Line(0.5, color_cfg=FixedTextColorCfg()),  # 线条效果，搭配固定的文本颜色配置
                OneOf([DropoutRand(), DropoutVertical()]), # 随机选择一个效果：随机点滴或垂直点滴
            ]
        ),
    )

def enum_data():
    """
    生成枚举类型文本的渲染配置。
    """
    return base_cfg(
        inspect.currentframe().f_code.co_name,            # 获取当前函数名
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "enum_text.txt"],  # 枚举文本的路径
                filter_by_chars=True,                     # 是否按字符过滤
                chars_file=CHAR_DIR / "chn.txt",          # 字典文件
                **font_cfg                                # 使用先前定义的字体配置
            ),
        ),
    )

def rand_data():
    """
    生成随机文本的渲染配置。
    """
    return base_cfg(
        inspect.currentframe().f_code.co_name,            # 获取当前函数名
        corpus=RandCorpus(
            RandCorpusCfg(chars_file=CHAR_DIR / "chn.txt", **font_cfg),  # 随机语料库配置，指定字符文件和字体配置
        ),
    )

def eng_word_data():
    """
    生成英文单词文本渲染配置。
    """
    return base_cfg(
        inspect.currentframe().f_code.co_name,            # 获取当前函数名
        corpus=WordCorpus(
            WordCorpusCfg(
                text_paths=[TEXT_DIR / "eng_text.txt"],   # 英文文本的路径
                filter_by_chars=True,                     # 是否按字符过滤
                chars_file=CHAR_DIR / "eng.txt",          # 字符文件
                **font_cfg                                # 使用先前定义的字体配置
            ),
        ),
    )

def same_line_data():
    """
    生成同行布局的文本渲染配置。
    """
    return base_cfg(
        inspect.currentframe().f_code.co_name,            # 获取当前函数名
        layout=SameLineLayout(),                          # 使用同行布局
        gray=False,                                       # 设置为彩色图像
        corpus=[
            EnumCorpus(
                EnumCorpusCfg(
                    text_paths=[TEXT_DIR / "enum_text.txt"],  # 枚举文本的路径
                    filter_by_chars=True,                 # 是否按字符过滤
                    chars_file=CHAR_DIR / "chn.txt",      # 字符文件
                    **font_cfg                            # 使用先前定义的字体配置
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "eng_text.txt",
                    ],                                    # 中英文文本的路径
                    filter_by_chars=True,                 # 是否按字符过滤
                    chars_file=CHAR_DIR / "chn.txt",      # 字符文件
                    length=(5, 10),                       # 文本长度范围
                    font_dir=font_cfg["font_dir"],        # 字体目录
                    font_list_file=font_cfg["font_list_file"],  # 字体列表文件
                    font_size=(30, 35),                   # 字体大小范围
                ),
            ),
        ],
        corpus_effects=[Effects([Padding(), DropoutRand()]), NoEffects()],  # 文本效果，包括填充和随机点滴
        layout_effects=Effects(Line(p=1)),                # 布局效果，包括一条概率为1的线
    )

def extra_text_line_data():
    """
    生成具有额外文本行布局的文本渲染配置。
    """
    return base_cfg(
        inspect.currentframe().f_code.co_name,            # 获取当前函数名
        layout=ExtraTextLineLayout(),                     # 使用额外文本行布局
        corpus=[
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "eng_text.txt",
                    ],                                    # 中英文文本的路径
                    filter_by_chars=True,                 # 是否按字符过滤
                    chars_file=CHAR_DIR / "chn.txt",      # 字符文件
                    length=(9, 10),                       # 文本长度范围
                    font_dir=font_cfg["font_dir"],        # 字体目录
                    font_list_file=font_cfg["font_list_file"],  # 字体列表文件
                    font_size=(30, 35),                   # 字体大小范围
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "eng_text.txt",
                    ],                                    # 中英文文本的路径
                    filter_by_chars=True,                 # 是否按字符过滤
                    chars_file=CHAR_DIR / "chn.txt",      # 字典文件
                    length=(9, 10),                       # 文本长度范围
                    font_dir=font_cfg["font_dir"],        # 字体目录
                    font_list_file=font_cfg["font_list_file"],  # 字体列表文件
                    font_size=(30, 35),                   # 字体大小范围
                ),
            ),
        ],
        corpus_effects=[Effects([Padding()]), NoEffects()],  # 文本效果，包括填充
        layout_effects=Effects(Line(p=1)),                # 布局效果，包括一条概率为1的线
    )

def imgaug_emboss_example():
    """
    生成使用 ImgAug 进行浮雕效果处理的文本渲染配置。
    """
    return base_cfg(
        inspect.currentframe().f_code.co_name,            # 获取当前函数名
        corpus=get_char_corpus(),                         # 获取字符语料库
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),  # 应用填充效果，指定宽高比
                ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),  # 应用 ImgAug 的浮雕效果
            ]
        ),
    )
```

可以简单配置一个温控表数据生成函数（仅示例）：

```python
def digital_display_data():
    """
    生成用于数码管显示效果的文本渲染配置。
    """
    return base_cfg(
        name=inspect.currentframe().f_code.co_name,           # 获取当前函数名，用作生成配置的名称
        corpus=CharCorpus(
            CharCorpusCfg(
                text_paths=[TEXT_DIR / "digital_text.txt"],   # 指定数码管文本的路径
                chars_file=CHAR_DIR / "digital_dict.txt",     # 指定数码管使用的字典
                length=(4, 5),                                # 文本长度范围（相同会报错）
                char_spacing=(-0.3, 1.3),                     # 字符间距
                **font_cfg                                    # 使用先前定义的字体配置
            ),
        ),
        corpus_effects=Effects([]),                           # 不使用任何特效
        layout=SameLineLayout(),                              # 使用单行布局
        gray=False                                            # 设置为彩色显示
    )

```

复制 example.py ，删除七个 data 函数，加入 `digital_display_data` ，修改 configs ，命名为 digital_example.py

在 `base_cfg` 函数中，修改 `num_image` 来设置生成图片的数量，默认为 50

- 生成

```bash
python main.py --config example_data/digital_example.py --dataset img --num_processes 2 --log_period 10
```

在 text_renderer/example_data/output/digital_display_data/images/ 中可以看到刚刚生成的合成数据：

<p align="center">
  <img src="https://i.imgur.com/3cQ99ol.jpg" width="30%" />
  <img src="https://i.imgur.com/uuLVVT5.jpg" width="30%" />
  <img src="https://i.imgur.com/FShn8wT.jpg" width="30%" />
</p>
可以看到，合成图像与真实图像还是有不小差异，为了使合成图像更接近真实数据，后续可以考虑以下几个修改方向：

- **字符材质和颜色调整**：根据真实数码管显示的材质和颜色进行调整。真实数码管使用的是LED发光，可以调整光源效果，使字符呈现类似LED的亮度和饱和度。

- **字符形状与大小**：确保字符的形状、大小和字体与真实数码管一致。可以调整字体样式或者自定义字体来匹配真实的显示效果。

- **背景和周边环境**：调整合成图像的背景，使之与真实数码管使用环境的背景一致。包括背景颜色、材质以及可能的反光或阴影效果。

- **光照和阴影效果**：真实环境中的数码管会受到环境光照影响，显示不同的光照和阴影效果。可以在合成图像中添加相似的光照效果，比如通过调整亮度、对比度或是使用图像处理软件中的光照效果工具。

- **噪声和损耗效果**：真实的数码管图像可能会有细微的噪声或损耗，可以对图像添加滤波，增加真实感。