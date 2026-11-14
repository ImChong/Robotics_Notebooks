# RoboMP-DINOv2（arXiv:2609.25506）

> 来源归档（paper）

- **标题：** RoboMP-DINOv2: Prompts, Not Filters for Robust Robot Manipulation
- **作者：** Han Qi、Heng Yang
- **机构：** 哈佛大学 SEAS（Harvard University）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.25506>
- **PDF：** <https://arxiv.org/pdf/2609.25506>
- **代码：** <https://github.com/han20192019/RoboMP_DINOv2>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **核对范围：** 全文通读 arXiv `2609.25506v1`（§1–§6，含 Figs. 1–4 与 Tables 1–4）并对照官方仓库 `han20192019/RoboMP_DINOv2` 全部源码与 README/MODEL_CARD
- **通读：** 已完成
- **一句话说明：** 以“前景掩码空间提示（mask prompts）注入完整图像特征”替代背景过滤（hard masking），并用仅随机化被掩码物体像素颜色的 MCR 训练增强，让基于 DINOv2 的扩散策略同时具备障碍物意识、空间抗杂乱鲁棒性与背景颜色线索保留能力（全仿真评测）。

## 开源状态

- **部分开源（仅发布可复用视觉编码器包 `robomp_dinov2`）**（2026-11-14 源码核查）：公开 PyTorch 视觉表征模块 `RoboMPDINOv2` / `RoboMPDINOv2MCR` 与 `MaskedRegionColorRandomization`、最小推理脚本及旧 `PFSE` checkpoint 转换脚本；**下游 Diffusion Policy、CLIP 文本编码管线、Grounding DINO + SAM 在线掩码生成、7 个仿真任务环境、演示数据及完整预训练策略权重均未随仓库发布**（`hubconf.py` 调用 `pretrained=True` 会抛出未内置权重异常）。

## 核心摘录

### 1) Prompts, Not Filters：掩码提示融合完整场景特征（§3）

- **摘录要点：** 传统以物体为中心的 2D/3D 方法（如 DOCIR、GROOT）将语义掩码用作**硬过滤器**裁剪或屏蔽背景，丢弃了障碍物、支撑面与环境状态（例如炉灶指示灯颜色）。RoboMP-DINOv2 则将完整 RGB 图像送入冻结 `DINOv2 ViT-B/14`（112×112，patch 14 → 8×8=64 空间 token）保持全场景信息，将多槽位掩码（`object`、`target` 及隐含 `background`）映射并按权重嵌入到对应 patch token 上，再过 3 层 Transformer 编码器聚合出紧凑状态表征，最后配 `CLIP` 文本特征（经 MLP）、本体状态与观测视野 `T_o` 输入 `Diffusion Policy`（仅用策略去噪损失训练，无额外辅助损失）。

### 2) Masked-Region Color Randomization（MCR）与上下文颜色保留

- **摘录要点：** 常规全图颜色抖动（full-image randomization）会同时破坏任务相关背景颜色信号；**MCR** 在训练阶段**只把前景掩码槽位并集（默认 `"object"`）内的像素替换为均匀随机 RGB 颜色**（`[0, 1]^3`），推理阶段完全关闭，保持背景与提示信号颜色不变。实验中模拟“绿灯开、红灯关”炉灶状态任务上，RoboMP-DINOv2-MCR 分别达到 `88%` 与 `71%`，全图随机化 DP 基线仅 `25%` 与 `19%`。

### 3) 仿真定量评估结果（§4–§5, Tables 1–4 & Figs. 2–3）

- **摘录要点：** 在 7 个仿真操作任务（每任务 100 条示范，训练用仿真真值掩码、评测用 Grounding DINO + SAM 在线分割掩码）上评测：
  - **Clean Spatial OOD 7 任务均值（Table 1）：** 2D 物体中心（DOCIR）`8.1%`、3D 物体中心（GROOT）`28.6%`、DP-DINOv2 `50.7%`、RoboGround-DINOv2 `55.9%`、**RoboMP-DINOv2 `60.7%`**（在 6/7 任务居首；`Place cube` 一项 3D GROOT 为 `85%` 高于 `54%`）。
  - **Clutter OOD 7 任务均值（Table 1）：** 2D `6.3%`、3D `16.4%`、DP-DINOv2 `41.0%`、RoboGround-DINOv2 `36.8%`、**RoboMP-DINOv2 `59.7%`**（如 `Put can w/ obstacle` 达 `80.0%` vs DP `52.0%` / 3D `3.0%`；`Two arm handover` 达 `81.0%` vs DP `41.0%`）。
  - **Appearance OOD 7 任务均值（Table 3，全部基线均配颜色随机化）：** 2D `12.0%`、3D `14.2%`、DP + full rand `19.1%`、RoboGround-DINOv2 `35.1%`、**RoboMP-DINOv2-MCR `72.5%`**（全 7 任务领先）。
  - **Combined Appearance + Clutter（Table 4，3 项高难任务）：** `Two arm / Place cube / Pull tool` 下 RoboMP-DINOv2-MCR 为 **`0.26 / 0.24 / 0.38`**，而 DP + full rand 仅 `0.00 / 0.02 / 0.00`。
  - **匹配容量消融（Fig. 2，3 种子 × 50 trials）与扰动稳定性（Table 2）：** 性能排序为 RoboMP-DINOv2 > All-one masks > No mask prompting > DINOv2 CLS+mean，排除纯架构容量因素；表征余弦/相对 L2 偏移在 4/5 杂乱场景与全部 5 个外观变更场景中低于 DP-DINOv2。**局限：全篇实验均为仿真，不含真机验证。**

## 对 wiki 的映射

- [paper-robomp-dinov2](../../wiki/entities/paper-robomp-dinov2.md)
- [RoboMP-DINOv2 仓库归档](../repos/robomp_dinov2.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
