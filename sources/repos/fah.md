# FAH 官方源码：noahcao/FAH

> 一手开源代码核对与复现入口（repo）。

- **项目：** FAH — Fast and Accurate Hand Estimation in Pixels, Interaction and World Space
- **类型：** repo
- **仓库：** <https://github.com/noahcao/FAH>
- **论文：** <https://arxiv.org/abs/2611.07711>
- **项目页：** <https://www.jinkuncao.com/FAH/>
- **权重：** <https://huggingface.co/noahcao/FAH>
- **核对日期：** 2026-11-10
- **源码核对范围：** README.md、pyproject.toml、demo.py、scripts/infer.py、fah/__init__.py、fah/cli.py、fah/api.py 和 bench/methods/fah.py 的推理入口、参数传递与模块创建，及 README 给出的加速、视频和世界系配置；不是逐行审计整个仓库。
- **开源状态：** 有可运行推理 CLI/Python API，且包含 train/、bench/ 等训练和评测入口；权重研究/非商业许可，第三方资产单独授权。

## 官方快速路径

1. 安装：Python 3.10+、PyTorch 2.1+；官方示例在克隆后执行 pip install -e ".[trt]"。
2. 权重：HF 下载 noahcao/FAH 到 ~/.cache/fah/weights（亦可设置 FAH_WEIGHTS），按需取得 MANO 与 SAM 3D Body/MHR 相关许可资产。
3. 配置：复制 configs/local.example.json 为 configs/local.json，填写路径与可选输出参数。
4. 图像：python scripts/infer.py hand.jpg；视频：python scripts/infer.py clip.mp4。世界系使用视频输入及 mode=world，另需相机相关依赖。
5. 也可以直接 import fah 后调用 fah.load("fah", mode="image")，输入 BGR 图像和可选内参 K。
6. 推理默认输出 joints2d 像素坐标与 camera-frame joints3d 米坐标；根据 --mesh、--parts、--contact、--dense 和 world 模式额外返回网格、分割、接触和世界系输出。

## 已核验的运行时模块

- [scripts/infer.py](https://github.com/noahcao/FAH/blob/main/scripts/infer.py)：读取本地 config，判断图像、视频或世界模式，生成 python -m fah.cli 命令。
- [fah/cli.py](https://github.com/noahcao/FAH/blob/main/fah/cli.py)：解析模型尺寸/网格/视频/接触选项，读取帧、计算内参、调用模型，输出 JSON 与可视化叠图。
- [fah/api.py](https://github.com/noahcao/FAH/blob/main/fah/api.py)：fah.load / FAH 和 method_args，组织权重、可选 TensorRT 引擎、视频 refiner、接触头、网格拟合与相机设置；__call__ 返回 hand 字典。
- [bench/methods/fah.py](https://github.com/noahcao/FAH/blob/main/bench/methods/fah.py)：FAHMethod 初始化检测器与手部模型，进行点预测、拟合及可选处理。
- [fah/world.py](https://github.com/noahcao/FAH/blob/main/fah/world.py)：世界模式相机估计与坐标放置接口；离线相机路线见 docs/world.md。

## 复现与授权边界

- README 标注 RTX 4070 Ti SUPER、TensorRT、WSL2、含网格拟合、未开启 parts/contact 的 **65 FPS**；这不是我们本地独立基准。
- 视频需在每条 clip 前调用 model.reset()，逐帧时序执行。
- 仓库代码 Apache-2.0；WEIGHTS_LICENSE 中的所有 FAH 权重限研究和非商业用途；MANO、SAM/DINOv3 等依上游条款取得。
- 未在本次 ingest 中运行实际权重、下载完整数据集或测量帧率，部署前需自行核对显存/模型版本/本机设备与许可。

## 映射

- [论文一手资料](../papers/fah_arxiv_2611_07711.md)
- [项目页](../sites/fah.md)
- [FAH 知识实体](../../wiki/entities/paper-fah.md)
