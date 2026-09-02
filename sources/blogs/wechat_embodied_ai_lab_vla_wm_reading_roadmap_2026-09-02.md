# VLA 和世界模型的宝藏教程：入门到进阶阅读路线图

> 来源归档（blog / 微信公众号）

- **标题：** VLA和世界模型的宝藏教程！入门到进阶的阅读路线图，附必读论文
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/fNAyDttYIs5kzTQHwxc5Pw
- **发表日期：** 2026-09-02
- **入库日期：** 2026-09-02
- **抓取方式：** Agent Reach + `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_ai_lab_vla_wm_reading_roadmap_2026-09-02.md`](../raw/wechat_embodied_ai_lab_vla_wm_reading_roadmap_2026-09-02.md)
- **一句话说明：** 10 篇 VLA 及相关基础论文 + 4 篇世界模型补充的阅读路线；**14/14 均有独立 `paper-*` 详情节点**（本 ingest **新建 10**、**复用 4**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：先把视觉–语言对齐与动作生成基座读通，再进入 RT 系列 → 开源 VLA → 流匹配动作头；世界模型四篇回答「预测如何服务策略」而不是再堆一条纯 VLA。

### 14 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | RT-1 | [2212.06817](https://arxiv.org/abs/2212.06817) | **已开源** `google-research/robotics_transformer` | [paper-rt-1](../../wiki/entities/paper-rt-1.md) |
| 02 | RT-2 | [2307.15818](https://arxiv.org/abs/2307.15818) | **官方训练未开源**；推理可参考社区实现 | [paper-rt-2](../../wiki/entities/paper-rt-2.md) |
| 03 | OpenVLA | [2406.09246](https://arxiv.org/abs/2406.09246) | **已开源** `openvla/openvla` | [paper-openvla](../../wiki/entities/paper-openvla.md) |
| 04 | π₀ | [2410.24164](https://arxiv.org/abs/2410.24164) | **已开源** `Physical-Intelligence/openpi`（非 `pi0` 仓名） | [paper-pi0](../../wiki/entities/paper-pi0.md) |
| 05 | Octo | [2405.12213](https://arxiv.org/abs/2405.12213) | **已开源** `octo-models/octo` | [paper-octo](../../wiki/entities/paper-octo.md) |
| 06 | Diffusion Policy | [2303.04137](https://arxiv.org/abs/2303.04137) | **已开源** `real-stanford/diffusion_policy` | [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md) |
| 07 | CLIP | [2103.00020](https://arxiv.org/abs/2103.00020) | **已开源** `openai/CLIP` | [paper-clip](../../wiki/entities/paper-clip.md) |
| 08 | ACT | [2304.13705](https://arxiv.org/abs/2304.13705) | **已开源** `tonyzhaozh/act` | [paper-act](../../wiki/entities/paper-act.md) |
| 09 | SPOC | [2312.02976](https://arxiv.org/abs/2312.02976) | **已开源** `allenai/spoc-robot-training` | [paper-spoc](../../wiki/entities/paper-spoc.md) |
| 10 | DINOv2 | [2304.07193](https://arxiv.org/abs/2304.07193) | **已开源** `facebookresearch/dinov2` | [paper-dinov2](../../wiki/entities/paper-dinov2.md) |
| 11 | LaDi-WM | [2505.11528](https://arxiv.org/abs/2505.11528) | **已开源** `GuHuangAI/LaDiWM` | [paper-sa-2505-11528-ladi-wm-a-latent-diffusion-based-world-model-for](../../wiki/entities/paper-sa-2505-11528-ladi-wm-a-latent-diffusion-based-world-model-for.md) |
| 12 | DreamDojo | [2602.06949](https://arxiv.org/abs/2602.06949) | **已开源** `NVIDIA/DreamDojo` | [paper-hrl-stack-35-dreamdojo](../../wiki/entities/paper-hrl-stack-35-dreamdojo.md) |
| 13 | RISE | [2602.11075](https://arxiv.org/abs/2602.11075) | **已开源** `OpenDriveLab/RISE` | [paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio](../../wiki/entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md) |
| 14 | PointWorld | [2601.03782](https://arxiv.org/abs/2601.03782) | **已开源** `NVlabs/PointWorld` | [paper-sa-2601-03782-pointworld](../../wiki/entities/paper-sa-2601-03782-pointworld.md) |

### 文内阅读路线

- **入门：** CLIP → RT-1 → RT-2 → OpenVLA；旁路 Diffusion Policy → π₀；动手 ACT。
- **进阶：** OpenVLA 源码 → Octo 架构 → π₀ Flow Matching。
- **世界模型补充：** LaDi-WM（隐空间预测操作）→ DreamDojo（人类视频预训练）→ RISE（想象 RL）→ PointWorld（3D 点流跨本体）。

## 对 wiki 的映射

- **14/14 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=wiki-entities-paper-…` 均可直达。
- **本 ingest 新建 10** 个实体；**复用 4**（LaDi-WM / DreamDojo / RISE / PointWorld 已有 arXiv 节点）；**0 重复 arXiv 节点**。
- Octo 的 canonical 论文节点从方法页 [octo-model](../../wiki/methods/octo-model.md) **迁到** [paper-octo](../../wiki/entities/paper-octo.md)（方法页保留，去掉 `arxiv` 以免双节点）。
- 阅读坐标：[VLA / 世界模型 14 篇阅读路线](../../wiki/overview/vla-wm-reading-roadmap-14-papers-technology-map.md)（**非**论文详情替代）。
- 交叉：[VLA](../../wiki/methods/vla.md)、[hub-vla](../../wiki/overview/hub-vla.md)、[VLA 纵深](../../roadmap/depth-vla.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 14 篇独立节点核查（10 新建 / 4 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
