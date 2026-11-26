# Doppel 官方训练推理与标定源码

- **类型：** repo / video world model
- **代码：** https://github.com/DoppelWM/Doppel
- **论文：** https://arxiv.org/abs/2611.24925
- **项目：** https://doppelwm.github.io/
- **权重：** https://huggingface.co/sourlreapwr/doppel-emb8
- **相机参数：** https://huggingface.co/datasets/sourlreapwr/doppel-camera-extrinsics
- **核查日期：** 2026-11-26
- **固定版本：** [源码 306fc88](https://github.com/DoppelWM/Doppel/tree/306fc883dff36069c11d796b680f3f2998e45401) · [论文归档](../papers/doppel_arxiv_2611_24925.md) · [项目页归档](../sites/doppel.md)
- **核对范围：** 官方 README、推理与串窗源码、标定入口与 ABC 导出门槛、权重索引、公开外参数据卡/许可和三张 CSV。只做源码静态核验与 CSV 数值检查，未运行 GPU。
- **通读：** 已完成

## 原文记录

代码可训练、推理、评测和标定；`python -m doppel.inference --video clip.mp4 --render render.mp4 --actions actions.npz --robot_type panda --out out/pred.mp4`。入口通过 `DoppelWorldModel.from_pretrained` 加载 Wan2.1 I2V、VAE、CLIP 和 render-conditioned transformer，并使用 `predict` 输出 17 帧视频。作者设置 bf16、80 GB GPU、50 Euler steps、分辨率 240x432。README 另有 `python -m doppel.rollout` 与 `doppel.train`，`calibration/` 可估计缺失外参。**根目录代码 MIT**；部分标定文件依 Gaussian-Splatting / PyTorch3D 许可，权重 CC BY-NC 4.0，基础模型另受许可证约束。

- [Doppel-8 权重复核](https://huggingface.co/sourlreapwr/doppel-emb8/tree/8fe4a7001966cf48fef69b1c58a21fb67f0a3c81)：公开配置、各本体 action stats 与 7 个 transformer overlay 分片。不是独立的全量 Wan 基座。未下载或执行权重。
- [相机外参复核](https://huggingface.co/datasets/sourlreapwr/doppel-camera-extrinsics/tree/519cba5b6b58da3253bfe86634f8090bf4444fb6)：读取 Bridge、RoboMIND、ABC 三张 CSV，合计 45,345 行，矩阵与相机 K 可匿名获取。数据目录按各来源许可：Bridge CC BY 4.0、RoboMIND/ABC 参数 Apache-2.0，以随库条款为准。
- [Dr.Robot 资产链接](https://huggingface.co/sourlreapwr/doppel-drrobot) 在此次匿名复核时返回 HTTP 401。不要把可读取的标定源码和外参表等同于文档中四套 Dr.Robot 资产可匿名下载。有已有同视角渲染时，视频推理 CLI 本身并不需要先下载这套资产。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-doppel.md)
