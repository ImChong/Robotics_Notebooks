# ViDiHand 官方共享实现 | ViDiHand and ViDiHand2

- **仓库：** [NTUYWANG103/ViDiHand](https://github.com/NTUYWANG103/ViDiHand) · [核验提交 23cf80c](https://github.com/NTUYWANG103/ViDiHand/tree/23cf80c292f2fb2db20caa1dee96516a63ee30a7)
- **核验日期：** 2026-11-26；已替换 8 月的占位仓状态。
- **论文与项目：** [初代论文](../papers/vidihand_arxiv_2606_30308.md) / [初代项目](../sites/vidihand-github-io.md)；[二代论文](../papers/vidihand2_arxiv_2611_30597.md) / [二代项目](../sites/vidihand2-github-io.md)。
- **核心入口：** `vidihand_v1/inference.py` / `vidihand_v2/inference.py` 的 `load_model()`；根目录 `demo.ipynb`、`demo_gradio/app.py`。
- **模型文件：** [HF v2 文件目录](https://huggingface.co/WANG-XIXI/ViDiHand/tree/main/vidihand_v2) 显示 `vdm.safetensors`（约 6.07 GB）和 `projector.safetensors`（约 41.3 MB）。MANO 文件通过独立 EgoHandBench 数据仓路径获取；未下载完整权重。
- **Space：** [HF 源码提交 567da96](https://huggingface.co/spaces/WANG-XIXI/ViDiHand2/tree/567da963c6621176826b823f391194a061aae58c) 的 `app.py` 包装 Gradio，实现 ZeroGPU 调用和 CPU 渲染；未跑托管服务。
- **开放边界：** 核验到推理实现和公开权重清单；未在共享仓发现训练入口或独立许可文件。第三方组件许可应分别复核。
- **核对范围：** README、两代推理入口、v2 几何求解、样例 I/O、Space 入口和权重清单；仅静态核验。
- **通读：** 已完成

## 对 Wiki 的映射

- [初代 ViDiHand](../../wiki/entities/paper-vidihand.md)
- [ViDiHand2](../../wiki/entities/paper-vidihand2.md)
