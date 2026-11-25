# Pi4 官方开源推理仓库

> 类型：repo；来源归档

- **仓库：** <https://github.com/yyfz/Pi4>
- **论文：** <https://arxiv.org/abs/2611.23144>（v1）
- **项目页：** <https://yyfz.github.io/pi4/>
- **权重：** <https://huggingface.co/yyfz233/Pi4>
- **许可：** 原创代码与 Pi4 模型权重均为 **CC BY-NC 4.0**（README 的 License 表）；不允许商业使用，第三方代码保留其各自许可
- **入库日期：** 2026-11-21
- **核对范围：** README Overview / Quickstart / Python API / License、`docs/inference.md` 的安装说明、`example.py` 完整 CLI、`pi4/io.py` 图像/视频装载、`pi4/models/pi4.py` 模型 forward / lift_tracks、`pi4/visualization.py` 输出采样与可视化入口；仅静态阅读，未做 GPU 执行。
- **通读：** 已完成
- **开源状态：** **已开源推理实现 + 推理模型权重**。仓库文档将其描述为固定推理模型，未见完整训练框架入口；不要把论文训练配置冒充可一键复现。
- **论文归档：** [π⁴ 原文](../papers/pi4_arxiv_2611_23144.md)
- **项目页归档：** [Pi4 主页](../sites/pi4-4d-scene-geometry-motion.md)
- **沉淀到 wiki：** [π⁴ 独立论文详情](../../wiki/entities/paper-pi4-4d-scene-geometry-motion.md)

## 官方运行路径（从源码核对）

1. `example.py` 解析输入路径、`--weights`、`--output`、`--query-frame`、`--stride`、`--max-frames` 与 `--viewer`。
2. `pi4.io.load_images` 读取单图、目录（排序）或视频，缩放到 16 的倍数并返回 RGB 帧和抽帧信息。
3. `Pi4.from_pretrained` 从本地 safetensors 或默认 `yyfz233/Pi4` 下载权重；随后 `Pi4.forward` 用图像编码器和几何解码器，输出 `local_points`、`camera_poses`、`metric_scale`、`confidence`。
4. 若指定合法查询帧且至少两张图，`forward` 再经 `motion_decoder`、`tracker`、`motion_mask` 和 `lift_tracks` 输出 `tracks_2d`、`tracks_3d`、`visibility`、`tracking_confidence`、`dynamic_mask`。
5. `save_visualization` 生成输出与可选 viewer 数据；`python viewer/serve.py <output> --port 8080` 提供浏览器预览。源码输出与流程图详见对应 wiki 论文节点。

## 复现条件与限制

- 官方推荐 Linux x86-64、Python 3.11、CUDA 12.6 相容驱动、BF16 NVIDIA GPU；`DepthRefiner` 需要 CUDA。
- README 测量 A800 上 150 帧 512×288 的峰值显存为 12.03 GiB；更小显存应减少帧数，并以实际运行验证为准。
- 单图输入支持几何重建；`query_frame` 跟踪要求至少两张图，输出几何和轨迹均受预测误差影响。
- 代码的点云与轨迹输出不等于具备接触、质量、摩擦、关节等物理属性的 Real2Sim 资产。
