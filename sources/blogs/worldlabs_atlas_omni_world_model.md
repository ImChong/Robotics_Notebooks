# World Labs Blog：Atlas — A World Model for Spatial Intelligence

> 官方技术博客来源归档（ingest）

- **标题：** Atlas: A World Model for Spatial Intelligence
- **类型：** blog / official
- **URL：** <https://www.worldlabs.ai/blog/atlas>
- **作者/机构：** World Labs Team
- **日期：** 2026-09-01（文内标注）
- **入库日期：** 2026-09-02
- **一句话说明：** World Labs 下一代 **omni 世界模型**：从零预训练、原生多模态（文本/图像/视频/3D），**多模态自回归扩散 Transformer**；统一支撑相机可控生成、稀疏视角 3D 重建、时空仿真（含 Real-to-Sim 机器人）与文生图/全景。

## 核心定位

- **世界模型** 在此指：生成、重建与仿真任意可能世界；理解世界如何出现、演化，以支持创意渲染、高保真真实世界仿真与机器人动作规划。
- **Atlas** 是 **omni model**：原生操作 text、images、video、3D；所有输入汇入 **共享 spatial context**（每张图锚定在 3D 位姿），再条件生成下一步输出，并在 3D 上保持一致。
- **可扩展**：官方称随训练算力增加性能提升，预期继续 scaling。

## 能力矩阵（博客归纳）

| 任务族 | 要点 |
|--------|------|
| **Camera-Controlled Generation** | 1–6 张参考图 + **像素级相机几何**（非仅靠文本运镜词）；最长约 **1 分钟 1440p** 视频；可在 spatial context 中放置多张无关参考图并 3D 插值缝合 |
| **Spatial Reconstruction** | 1 张到 **百余张** 输入视角；少视角时靠世界知识 **想象** 盲区，多视角时趋向 **忠实重建**；可输出 2D 新视角、点云、**3D Gaussian splats**（与 Marble 同表征） |
| **Space-Time Simulation** | 理解空间结构 + 时间演化；**子弹时间式** 多机位 reframing（3–5 部手机即可）；机器人 **Real-to-Sim**：重建空间后随机器人路径生成 body-mounted RGB+depth |
| **Image Generation** | 文生图、360 全景；复杂 prompt、文字渲染、多风格（非主焦点） |

## 模型架构（博客归纳）

**Multimodal autoregressive diffusion transformer**（rectified flow 扩散 + Transformer 骨干）：

| 属性 | 说明 |
|------|------|
| **Multimodal** | 文本、图像、相机位姿、3D depth map；视频 = 图像序列 |
| **Autoregressive** | 多模态序列逐元素生成；不同任务 = 不同序列模式（输入→输出） |
| **Diffusion** | rectified flow 去噪；可调步数权衡速度/质量 |
| **Transformer** | 大矩阵乘、适配现代硬件；可借鉴 LLM 的 KV-cache、disaggregated serving 与视频模型的蒸馏、CFG、VAE 等 |

**Spatial context**：每张图/深度图显式绑定 camera pose，形成 3D  grounded 上下文（区别于纯 token 序列的 LLM）。

## 评测（博客披露）

- **相机可控生成**：相对 SOTA 视频模型，第三方人工评分 **更跟相机路径**；轨迹越复杂优势越大（Atlas 用原生相机输入，基线用文本 cinematic 描述）。
- **稀疏视角 3D 重建**：在多个 benchmark 上 **低于** 专用开源重建模型误差（博客列 Pi3X、π³、VGGT-Ω、Depth Anything 3、MapAnything 等基线）；强调公平复现协议。

## 产品与开源（步骤 2.5 核查，2026-09-02）

- **Early access**：面向选定合作伙伴申请访问（博客页 request access）；**将驱动未来 Marble 等产品**。
- **代码/权重**：截至入库日博客与 [worldlabs.ai](https://www.worldlabs.ai/) **未列** GitHub / Hugging Face / 权重下载 → **未开源（早期访问）**。
- **Spark**：重建输出使用与 Marble 相同的 **3D Gaussian splat** 表征，与 [Spark](../../wiki/entities/spark-3dgs-renderer.md) 渲染栈天然衔接（Spark 本身已开源）。

## 引用（博客提供）

```bibtex
@article{worldlabs2026atlas,
    author = {World Labs Team},
    title = {Atlas: A World Model for Spatial Intelligence},
    journal = {World Labs Blog},
    year = {2026},
    note = {https://www.worldlabs.ai/blog/atlas},
}
```

## 对 wiki 的映射

- [`wiki/entities/atlas-world-model.md`](../../wiki/entities/atlas-world-model.md) — Atlas omni 世界模型实体页
- [`wiki/entities/world-labs.md`](../../wiki/entities/world-labs.md) — 公司与 Marble / Spark 上下文
- [`wiki/methods/generative-world-models.md`](../../wiki/methods/generative-world-models.md) — 生成式世界模型方法族中的 3D / Real-to-Sim 分支

## 当前提炼状态

- [x] 能力、架构与评测要点
- [x] 开源状态核查（早期访问，未公开仓库）
- [x] 与 wiki 实体/方法页交叉索引
