# Embody 3D（arXiv:2510.16258）

> 来源归档（ingest）

- **标题：** Embody 3D: A Large-scale Multimodal Motion and Behavior Dataset
- **类型：** paper / dataset / motion-capture / multimodal / social-behavior
- **机构：** Meta（Codec Avatars Lab）
- **作者：** Claire McLean, Makenzie Meendering, Tristan Swartz, Orri Gabbay, Alexandra Olsen, Rachel Jacobs, Nicholas Rosen, Philippe de Bree, Tony Garcia, Gadsden Merrill, Jake Sandakly, Julia Buffalini, Neham Jain, Steven Krenn, Moneish Kumar, Dejan Markovic, Evonne Ng, Fabian Prada, Andrew Saba, Siwei Zhang, Vasu Agrawal, Tim Godisart, Alexander Richard, Michael Zollhoefer
- **发布时间：** 2025-10-17（v1）/ 日期标注 October 21, 2025
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2510.16258>
  - 数据集主页：<https://www.meta.com/emerging-tech/codec-avatars/embody-3d>
  - 官方代码：<https://github.com/facebookresearch/embody-3d>
- **入库日期：** 2026-11-14
- **核对范围：** 2026-11-14 通读 arXiv:2510.16258v1 全文文本（摘要、§1–§6 及表格/图注）以及官方代码仓 `facebookresearch/embody-3d` 的 `README.md` 与 `src/{download.py,run.py,dataloader.py,enums.py,visualize.py,load_obj.py}`。未实际申请 XRCIA 许可下载原始数据集。
- **通读：** 已完成
- **一句话说明：** Meta Codec Avatars Lab 在 80 相机 640 通道消声多相机舞台采集的 500 个人小时（439 名参与者、超 5,400 万帧 30fps）三维 SMPL-X 动作、分离语音与多粒度文本行为数据集。

## 开源与获取状态

- **工具箱代码：** MIT/官方仓 LICENSE 开源于 `facebookresearch/embody-3d`（提供按类别与模态过滤下载脚本、PyTorch `BaseDataset` 与 Pyrender SMPL-X 渲染脚本）。
- **数据集本体：** 需在 Meta 官网填写 release form 后获取 21 条下载链接，受 **XRCIA** 协议约束。

## 核心技术要素摘录

1. **硬件采集舞台：**
   - 6m × 6m × 3.6m 消声处理采集室（有效追踪空间 3.6m × 3.6m），14 块 LED 面板（约 650 lux）。
   - **80 台全局快门 24.47MP（5320×4600）相机**，30 fps 同步采集：64 台身体相机（8–15mm F4，4ms 曝光）+ 16 台面部相机（35mm F1.4，2ms 曝光）。
   - **5 个定制 128 通道 MEMS 球形麦克风阵列**，合计 **640 通道**（支持 10 阶球谐波声场 / ambisonics 与波束成形说话人分离，另保存中心原始单声道音频 48 kHz）。
2. **标定与追踪重建流程：**
   - 定制基准标定车（p50 重投影误差 <0.2 px，p99 约 0.8 px）；每位参与者先录 A/T/C/T-Rex 校准姿态求解个体体型。
   - 多人追踪：检测 top-N 人物框 + Sapiens-1B 提取 308 个人脸/身体关键点 → 射线束时空聚类 + 人脸嵌入匈牙利匹配（Hungarian matching）→ RANSAC 三角化 + 能量最小化 → 训练后的姿态编码器（Procrustes + 残差细化）输出 **SMPL-X**（`betas` 300 维、`body_pose`、`global_orient`、`left_hand_pose`、`right_hand_pose`、`transl`）及 `missing/` 质量二值掩码。
   - 人工质检（Likert 1–5 分制），保留 ≥2.5 分片段。
3. **七大子集规模（合计 500 个人小时）：**
   - Charades（88.9h，221 人，片段/提示词文本）
   - Hand Interactions（111.3h，137 人，手-身与自接触）
   - Locomotion（21.0h，46 人，提示词文本）
   - Dyadic Conversations（59.4h，86 人，分离音频，情绪/高层文本）
   - Multi-Person Conversations（125.2h，210 人，分离音频，含座椅/沙发/高桌）
   - Scenarios（49.2h，77 人，音频 + 细粒度文本，含物品与家具交互）
   - Day in the Life（46.4h，77 人，音频 + 整段主题文本 + 每 10 秒文本，3–4 人公寓合居场景）
4. **官方代码接口：**
   - `src/download.py`：`--src`、`--feat`（`smplx/audio/text/videos`）、`--category`。
   - `src/dataloader.py`：`BaseDataset` 读取各子目录 `dataset.json`，滑窗切分 `seq_len`（默认 600 帧=20s），支持 `--load_multiperson` 同步加载交互同伴序列。
   - `src/run.py`：构建 `smplx.create(..., gender="neutral", num_betas=300, use_pca=False)` + `PyrenderRenderer` 渲染顶视图/网格并可选通过 `ffmpeg` 混入 48kHz 音频。

## 对应 wiki 节点

- [Embody 3D](../../wiki/entities/dataset-embody-3d.md)
