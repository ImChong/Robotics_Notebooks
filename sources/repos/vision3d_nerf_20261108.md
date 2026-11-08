# NeRF — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** NeRF
- **类型：** repo
- **链接：** https://github.com/bmild/nerf
- **入库日期：** 2026-11-08
- **一句话说明：** 神经辐射场的原始研究代码。
- **核对范围：** 已完整阅读原作者 README 的模型、训练、数据与坐标约定，核对 TensorFlow1.15 源码 run_nerf.py 的 run_network、render_rays、raw2outputs、create_nerf、render_path 和 run_nerf_helpers.py 的 get_embedder/init_nerf_model/sample_pdf；未运行训练或渲染。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [NeRF 官方实现](https://github.com/bmild/nerf) **（本归档主类型）**

## 作者源码运行结构

- `run_nerf.py --config config_fern.txt` 或 `config_lego.txt` 加载训练图像及已知 camera-to-world 位姿；真实场景通过 LLFF/COLMAP 预先估计位姿。
- `create_nerf` 配置位置频率编码器、view direction 编码器、粗和细两套 MLP；`render_rays` 构造视线 sample，`run_network` 批处理采样点，`raw2outputs` 做 alpha compositing。
- `sample_pdf` 据粗 MLP 的光线权重重要性采样，训练时 fine MLP 用更密集采样；`render_path` 从给定轨迹合成画面，`img2mse` 是训练重建目标。
- **速度边界**：README 明确原始 TF1.15 在单 GPU 每个场景训练数小时至一天；V100 渲染高分辨率约 30 秒/帧。NeRF 不是本身实时跟踪模块；需要已知相机位姿，不能直接当 3DGS SLAM。

## 方法与边界

神经辐射场的原始研究代码。已读原作者 README/训练入口及渲染采样模块，未在当前硬件独立复现性能。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 原始 ECCV 论文全文和附录另见 paper 归档。
- [x] 核对作者真实 train/render 入口和主要函数，详见独立 wiki 的源码时序。
- [x] NeRF 独立实体含两类 Mermaid、实验/速度/源码风险。
