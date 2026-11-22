# bear-ty/BeyondRetarget

- **名称：** BeyondRetarget
- **类型：** repo / humanoid / monocular-video / motion-estimation
- **URL：** <https://github.com/bear-ty/BeyondRetarget>
- **License：** 见仓库（另页说明扩展机器人需联系作者）
- **配套论文：** [BeyondRetarget（arXiv:2609.29850）](../papers/beyondretarget_arxiv_2609_29850.md)
- **项目页：** [`sources/sites/beyondretarget-github-io.md`](../sites/beyondretarget-github-io.md)
- **入库日期：** 2026-09-26
- **开源结论（2026-09-26）：** **已开源** — `bash scripts/setup_rgb2robo.sh`（Conda Python 3.10、PyTorch 2.3、CUDA 12.1）；checkpoint/机器人资产经 **Google Drive**；含 HMR2 与 YOLOv8x 权重下载逻辑。

## 一句话说明

单目 RGB → **共享 robot-oriented motion 表征** → 八款人形 root+关节轨迹；`app/infer/infer_video.py` 一条龙或 `scripts/preprocess_videos.py` 批处理；真机视觉遥操作需 **另配 SONIC** 并接本项目 **GR00T ZMQ** 接口。

## 入口与目录（README 摘要）

| 路径 | 作用 |
|------|------|
| `scripts/setup_rgb2robo.sh` | 环境、权重、机器人 mesh 安装与 `--check-only` |
| `scripts/preprocess_videos.py` | 批量抽 HMR2 特征 |
| `app/infer/infer_video.py` | 单视频推理：检测 → 特征 → motion + foot contact → 后处理 |
| `assets/README.md` | checkpoint 与八机描述文件落盘说明 |

## 支持机器人（截至 README）

Unitree **G1、R1、H1**；Booster **T1**；**Tienkung**；Fourier **GR1-T1、GR2-V3**；**Atlas**。

## 版本边界

- **本仓库**为 **base version**：实时 teleop 取向，不包含移动相机/大尺度轨迹的 Plus 前端。
- **2026-11-22 版本更新：** [Plus 独立仓库](../repos/beyondretarget-plus.md) 已开源移动相机与大范围全局轨迹离线生成管线；不能将 Plus 与本仓库的实时遥操作性能等同。

## 关联 wiki

- [paper-beyondretarget-monocular-humanoid.md](../../wiki/entities/paper-beyondretarget-monocular-humanoid.md)
