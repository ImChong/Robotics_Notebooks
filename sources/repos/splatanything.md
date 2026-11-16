# zju3dv/SplatAnything：官方推理源码

> 来源归档（ingest · 源码）

- **标题：** SplatAnything (InfiniSplat v3)
- **类型：** repo
- **代码：** <https://github.com/zju3dv/SplatAnything>
- **项目页：** <https://zju3dv.github.io/SplatAnything/>，归档见 [项目页](../sites/splatanything.md)
- **论文：** arXiv:2611.08264，归档见 [论文](../papers/splatanything_arxiv_2611_08264.md)
- **模型权重：** <https://huggingface.co/ritianyu/SplatAnything>
- **入库日期：** 2026-11-10
- **核对范围：** README.md、docs/inference.md、prepare_data.md、inference.py、src/models/models/{geometry_worldmirror,gaussian_worldmirror,rasterization}.py、src/utils/scene_io.py、License.txt；核查推理入口和关键数据流，未运行 CUDA 复现。
- **通读：** 已完成
- **开源状态：** 推理代码可获取，论文训练代码不在本仓；README 给出 HF 权重下载地址；许可不是可直接假定 Apache/MIT 的宽松许可证。
- **许可注意：** 仓库 `License.txt` 为 Tencent HunyuanWorld-Mirror Community License Agreement，含地域/用途相关限制，商用或再分发前须核对条款与第三方 Notice。
- **一句话说明：** 输入多张重叠 RGB 图与可选相机位姿，先预测几何再输出高斯 PLY、点云、RGB/深度和相机元数据。

## 源码入口与真实调用关系

1. `inference.py` 解析 `--input_path / --mode / --resolution`，从 `configs/inference.yaml` 解析模型和 checkpoint；默认分辨率 756×756（宽高需为 14 的倍数）。
2. `src.utils.scene_io.build_views` 读取场景 `images/`；只有 `--mode pose` 才要求 `prior_camera.json` 中完整相机内外参；`--mode rgb` 不读取相机 JSON。
3. `src.models.models.geometry_worldmirror.GeometryWorldMirror` 作为冻结几何模型前向得到 depth、camera poses/intrinsics、confidence 等。
4. `src.models.models.gaussian_worldmirror.GaussianWorldMirror` 提取 Gaussian 特征，经 `GaussianSplatRenderer.normalize_gaussian_spacing` 做尺度规范化、`DPTDecoder` 预测高斯属性、过滤异常尺度/天空后 `merge_voxels`。
5. `scene_io.export_gaussians / export_points / export_depth_rgb` 写出 `gaussians.ply`、`points.ply`、RGB/深度；入口还写 `cameras.json`、`config.yaml`、`run.json`。

## 最短复现

```bash
python -m pip install -r requirements.txt
# 将官方 Hugging Face 权重按 README 和 configs/inference.yaml 放到 checkpoints/，或用 CLI 显式传入 --ckpt_path
python inference.py --input_path examples/object --mode rgb --output_path outputs/object/rgb
python inference.py --input_path examples/object --mode pose --output_path outputs/object/pose
```

- RGB-only 不要求已知相机；pose 模式必须有内参与外参；`--sky_seg` 额外需要 ONNX 天空分割权重。
- 默认使用 CUDA；`--device cpu` 仅用于较慢的 smoke run，实际吞吐取决于显卡/视图数/分辨率。
- 源码中 Gaussian spacing target 默认约 `0.0013227513`，并非输出真实米；pose 模式仍对几何尺度做归一化，别直接把 `gaussians.ply` 的位置理解为原始世界系米制。
- 官方 README 提供 Rig 多相机示例（150 张视图）和 Panorama 转 24 个透视视图的脚本 `scripts/convert_panorama.py`。

## 对 wiki 的映射

- [SplatAnything 项目统一实体](../../wiki/entities/paper-splatanything.md)
- [项目主页](../sites/splatanything.md)
- [论文原文](../papers/splatanything_arxiv_2611_08264.md)
