# Blueprint3D — 2D 户型图转 3D 建筑模型

- **类型：** repo / 官方源码与技术说明
- **标题：** Floor Plan to 3D Pipeline（Blueprint3D）
- **作者：** ashnomad
- **代码：** https://github.com/ashnomad/Blueprint3D
- **核对版本：** [bb4a4fd81f0d29904c522dac710617ce9eb11b1e](https://github.com/ashnomad/Blueprint3D/tree/bb4a4fd81f0d29904c522dac710617ce9eb11b1e)
- **核对日期：** 2026-11-08
- **核对范围：** 已通读官方根 README、ForgeFloor/README.md、SplitFloor/README.md、SemaFloor/README.md；核对 ForgeFloor/extruder.py 的 run_both、ForgePlanPipeline.run、CLI 入口，以及 Shape.py / morpho.py 关键处理段和 SplitFloor/deploy.py、SemaFloor/deploy.py 的推理调用。未声称逐行审计全部训练源码，未运行模型。
- **通读：** 已完成
- **开放状态：** 三个模块及 Python 训练/推理/网格脚本公开；仓库目录未见 SplitFloor/checkpoints、SemaFloor/checkpoints 及 LICENSE，自动推理所需模型权重并未随所见版本一同分发，不能据此宣称开箱运行或获宽松商业许可。
- **归入 wiki：** [Blueprint3D](../../wiki/entities/blueprint3d-ashnomad.md)

## 原文中可核验的结构

| 模块 | 官方路径 | 职责 |
|---|---|---|
| SplitFloor | `SplitFloor/deploy.py`, `net.py` | ResNet-50 编码器、U-Net 类解码器；背景/门/墙/窗四类结构分割，导出 CSV 与 PNG |
| SemaFloor | `SemaFloor/deploy.py`, `deploy_sema.py`, `net.py` | ResNet-50 + ASPP + U-Net 类解码器；房间掩码及边界辅助头 |
| ForgeFloor | `ForgeFloor/extruder.py` | CLI/GUI；调起两套预测、连接几何清洗与拉伸 |
| ShapeProcessor | `ForgeFloor/Shape.py` | 连通域、形态学、结构细化及 CSV/overlay |
| MorphoExtruder | `ForgeFloor/morpho.py` | 使用 PyVista 构造楼板、墙和窗洞几何，导出 OBJ |

## 源码入口与复现边界

```bash
python ForgeFloor/extruder.py --gui
python ForgeFloor/extruder.py --input path/to/floorplan.jpg --auto --output_dir outputs/
# 已有分割 CSV 时可走手动路径，无须自动推理：
python ForgeFloor/extruder.py --sema sema.csv --split split.csv --output_dir outputs/
```

`NeuralInferenceOrchestrator.check_dependencies()` 逐项检查脚本与两个 checkpoints 目录；自动模式缺少目录即返回错误。不能把 README 的示例命令等同于已提供可立即复现的权重。几何输出仍需独立检查单位、尺寸、门窗间隙、网格拓扑。

## 对机器人研究的映射

- 建筑平面图转粗三维场景可作为导航地图/仿真场景初稿，**不是** 带物理参数的完整机器人仿真环境。
- 与 [Pascal Editor](../../wiki/entities/pascal-editor.md)（直接编辑建筑语义节点）及 [Infinigen](../../wiki/entities/infinigen.md)（程序化场景生成）对照。

## 一手链接

- [官方 README](https://github.com/ashnomad/Blueprint3D/blob/bb4a4fd81f0d29904c522dac710617ce9eb11b1e/README.md)
- [ForgeFloor README](https://github.com/ashnomad/Blueprint3D/blob/bb4a4fd81f0d29904c522dac710617ce9eb11b1e/ForgeFloor/README.md)
- [主入口 extruder.py](https://github.com/ashnomad/Blueprint3D/blob/bb4a4fd81f0d29904c522dac710617ce9eb11b1e/ForgeFloor/extruder.py)
- [SplitFloor 推理](https://github.com/ashnomad/Blueprint3D/blob/bb4a4fd81f0d29904c522dac710617ce9eb11b1e/SplitFloor/deploy.py)
- [SemaFloor 推理](https://github.com/ashnomad/Blueprint3D/blob/bb4a4fd81f0d29904c522dac710617ce9eb11b1e/SemaFloor/deploy.py)
