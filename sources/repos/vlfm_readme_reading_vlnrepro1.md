# VLFM 官方仓库 README 通读（VLN 复现范式页支撑）

> 来源归档（原文通读 · 支撑 `wiki/overview/vln-open-source-repro-paradigms.md`）

- **标题：** VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation（官方 GitHub README）
- **作者：** Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang Wang, Bernadette Bucher（Boston Dynamics AI Institute 等）
- **类型：** repo / navigation / zero-shot
- **原文：** <https://github.com/bdaiinstitute/vlfm>（2026-11 经代理获取 raw README 全文）；论文 <https://arxiv.org/abs/2312.03275>
- **一句话说明：** 零样本语义导航：深度图建 occupancy map 找 frontier，RGB + 预训练 VLM 生成语言对齐的 value map，选最有希望的 frontier 去找目标类别物体。
- **核对范围：** 通读 README 全文（Overview、Installation、HM3D 数据下载、权重下载、Habitat 评测、License、Citation）。未读论文正文与源码，`reset`/`step` 接口说明仅来自博客归档，未对照源码核对。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **方法（Overview）：** 深度观测建 occupancy map 以识别 frontier；RGB + 预训练 VLM 生成 language-grounded value map；据此选最有希望的 frontier。在 Gibson、HM3D、MP3D（Habitat）上 ObjectNav 的 SPL 报告为 SOTA；可部署到 Boston Dynamics Spot。
- **环境（Installation）：** conda Python 3.9；torch 1.12.1+cu113；安装 GroundingDINO（固定 commit）与 salesforce-lavis 1.0.2；仿真用 `pip install -e .[habitat]`，Spot 用 `.[reality]`；需在仓库内 clone yolov7。
- **数据：** 通过 Matterport 账号 token 下载 HM3D v0.2 train/val 场景与 HM3D ObjectNav v1 episodes。
- **权重：** `mobile_sam.pt`、`groundingdino_swint_ogc.pth`、`yolov7-e6e.pt` 需自行下载放入 `data/`；`pointnav_weights.pth` 随仓库提供。
- **评测：** `./scripts/launch_vlm_servers.sh` 起 tmux 会话，以 flask 服务加载各模型（用完需 kill 释放 GPU）；`python -m vlfm.run` 评测 HM3D，覆盖 `habitat.dataset.data_path` 评 MP3D。
- **维护状态（License）：** MIT；代码为 2023 夏季实习产物，「as is」提供，无主动维护。

## 对 wiki 的映射

- [`wiki/overview/vln-open-source-repro-paradigms.md`](../../wiki/overview/vln-open-source-repro-paradigms.md)
