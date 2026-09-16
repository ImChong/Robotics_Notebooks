# amap-cvlab/ABot-Recon

> 来源归档

- **标题：** ABot-Recon
- **类型：** repo / streaming-3d / monocular-reconstruction
- **机构：** 阿里巴巴（AMap CV Lab）
- **链接：** https://github.com/amap-cvlab/ABot-Recon
- **项目页：** https://amap-cvlab.github.io/ABot-Recon-html/
- **论文：** https://arxiv.org/abs/2608.27529
- **权重：** https://huggingface.co/acvlab/ABot-Recon
- **Demo：** https://huggingface.co/spaces/acvlab/abot-recon-streaming-3d
- **许可：** Apache-2.0
- **语言：** Python 3.10+
- **入库日期：** 2026-09-06（策展快照）；**2026-09-13** 深化论文 ingest
- **一句话说明：** 固定 **12 帧**局部上下文的单目 RGB 长视频流式 3D 重建：每步预测点图 + 相邻相对位姿，序贯组合全局几何；单目即可、有界内存。
- **开源状态：** **已开源**（`demo.py`、`abot_recon.ABotRecon` API、HF 自动拉权重、可选 FlashInfer/cuRoPE）
- **沉淀到 wiki：** [paper-abot-recon](../../wiki/entities/paper-abot-recon.md)（原 `cn-os-abot-recon` 策展页已并入此页）

---

## 定位

极长视频 **在线** 相机跟踪 + 局部稠密点图；学习态严格 **局部**（前 11 帧 KV），全局由位姿组合得到。与依赖持久长程记忆库的流式 3D 方法路线不同。

## 可运行入口

```bash
conda create -n abot-recon python=3.11 -y && conda activate abot-recon
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
# 可选加速
pip install flashinfer-python
cd abot_recon/modeling/pi3/models/curope && python setup.py build_ext --inplace

python demo.py --image-dir examples/images --output-dir outputs/demo --no-loop-closure
```

- 帧名须字典序零填充（`000001.jpg` …）。
- 默认 **启用** loop closure；论文流式核心数字多用 `--no-loop-closure`。
- Checkpoint 默认从 Hugging Face 下载；离线可放 `checkpoints/abot_recon.safetensors`。

## 关键设计（README）

| 模块 | 说明 |
|------|------|
| 12 帧 KV | 仅缓存前 11 帧 + 当前帧 |
| 点图 \(P_i\) | 当前相机坐标系稠密几何 |
| 相对位姿 | \(T_{i-1\leftarrow i}\) 相邻帧 |
| Refiner | 轻量运动–视觉旋转修正 |
| Loss | composition-aware pose 监督多步组合 |

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 论文实体 | `wiki/entities/paper-abot-recon.md` |
| 国内开源策展 | `wiki/entities/paper-abot-recon.md`（原 `cn-os-abot-recon`，已合并） |
| 项目页 | `sources/sites/abot-recon.md` |
| 论文摘录 | `sources/papers/abot_recon_arxiv_2608_27529.md` |
