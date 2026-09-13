# Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction（ABot-Recon，arXiv:2608.27529）

> 来源归档（ingest）

- **标题：** Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction
- **缩写 / 框架：** **ABot-Recon**
- **类型：** paper / streaming-3d / monocular / slam / reconstruction
- **arXiv：** <https://arxiv.org/abs/2608.27529>（Submitted 2026-08-27）
- **项目页：** <https://amap-cvlab.github.io/ABot-Recon-html/>
- **代码：** <https://github.com/amap-cvlab/ABot-Recon>
- **权重：** <https://huggingface.co/acvlab/ABot-Recon>（ModelScope 同步）
- **Demo：** <https://huggingface.co/spaces/acvlab/abot-recon-streaming-3d>
- **机构：** 阿里巴巴（高德 / AMap CV Lab，amap-cvlab）
- **入库日期：** 2026-09-13
- **一句话说明：** 长视频流式 3D 重建：固定 **12 帧**（前 11 帧 KV + 当前帧）局部上下文，每步预测当前相机系点图与相邻相对位姿，**序贯组合** 得全局轨迹与几何；单目 RGB、有界内存与算力。

## 开源状态（步骤 2.5）

- **项目页核查（2026-09-13）：** 页头链 GitHub、HF 模型与 **HF Space** 在线 demo；技术报告 PDF 在仓库内。
- **仓库核查：** `pip install -e .`、`demo.py`、`abot_recon.ABotRecon` Python API；checkpoint 自动从 HF 下载；可选 FlashInfer + cuRoPE 加速；Apache-2.0。
- **结论：** **已开源**（推理 / demo / 权重齐全；训练脚本以 README 发布范围为准）。

## 摘录 1：方法与局部上下文（Abstract / README）

- **设定：** 极长视频在线估计相机运动与场景几何，内存与每帧算力 **不随序列长度增长**。
- **与长记忆路线对照：** 不依赖持久化或多级长程学习状态；每步解 **同一有界问题**：
  - 缓存 **前 11 帧** KV 特征（共 12 帧窗口）；
  - 预测当前相机坐标系 **点图** \(P_i\)；
  - 估计相邻 **相对位姿** \(T_{i-1\leftarrow i}\)；
  - 通过位姿 **序贯组合** 恢复全局轨迹与点云。
- **抗漂移：** 轻量运动–视觉旋转 refiner + **composition-aware pose loss** 监督多步组合。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-abot-recon.md`](../../wiki/entities/paper-abot-recon.md)；升级原 [`cn-os-abot-recon`](../../wiki/entities/cn-os-abot-recon.md) 策展页。

## 摘录 2：评测亮点（README / 论文）

| 基准 | 结果 | 备注 |
|------|------|------|
| Oxford Spires 相机位姿 | ATE **4.35 m**，RPE-R **0.12°** | 纯流式、无 loop closure；较先前最佳约 **−40%** 误差 |
| Oxford Spires 稠密重建 | CD **1.37 m**，F1 **91.81%**（τ=4 m） | |
| KITTI-02 效率 | **24.45 FPS**，**6.71 GiB** | 504×280，H100；不含输入缓存 |

**对 wiki 的映射：** 结论写清：长程稳定性来自 **局部等变预测 + 组合监督**，而非更大记忆库。

## 摘录 3：复现入口（README）

```bash
conda create -n abot-recon python=3.11 -y && conda activate abot-recon
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
# 可选：flashinfer-python + cuRoPE 编译

python demo.py --image-dir examples/images --output-dir outputs/demo --no-loop-closure
```

- 输入：按字典序排序的单目 RGB 帧（`000001.jpg` …）；**无需深度**。
- 输出：轨迹、相邻相对位姿、局部点图、置信度；`--save-world-points` 组合为世界系点云。
- 默认可选 **loop closure** 精化（长序列重访区域）；最小因果推理可 `--no-loop-closure`。

**对 wiki 的映射：** 运行时序图对齐 `demo.py` → `ABotRecon` → 流式前向 → 写盘。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-abot-recon.md`**（canonical 论文页）。
- 深化 **`sources/repos/abot-recon.md`**、新建 **`sources/sites/abot-recon.md`**。
- 交叉具身导航 / 语义建图相关页（如 [2D→3D 提升 Gap](../../wiki/concepts/2d-to-3d-semantic-lifting-gap.md)）。
