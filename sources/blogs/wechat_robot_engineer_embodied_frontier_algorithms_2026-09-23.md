# 具身智能前沿算法（2026-2027最新，按技术路线分类，附核心算法、开源项目、适用场景）

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能前沿算法（2026-2027最新，按技术路线分类，附核心算法、开源项目、适用场景）
- **类型：** blog
- **作者：** 机器人研发工程师（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/JtoOU_ncZz5SEikmsZB3Xg
- **发表日期：** 2026-09-23（估）
- **入库日期：** 2026-09-23
- **抓取方式：** WebFetch（Jina/Camoufox 不可用；agent-reach 本环境未预装 wechat 通道）
- **原始抓取落盘：** [`sources/raw/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md`](../raw/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md)
- **一句话说明：** 六大技术路线（VLA / WAM / 腿式 RL / 扩散操作 / 规划 Agent / 触觉力觉）+ 2026 新开源清单 + Awesome 索引；**36/36 独立详情节点**（**新建 8**、**复用 28**）。

## 节点映射

| # | 项目 | 路线 | arXiv | 开源 | wiki |
|---|------|------|-------|------|------|
| 01 | OpenVLA | VLA | [2406.09246](https://arxiv.org/abs/2406.09246) | **已开源** | [openvla](../../wiki/entities/openvla.md)（**复用**） |
| 02 | Octo | VLA | [2405.11172](https://arxiv.org/abs/2405.11172) | **已开源** | [paper-octo](../../wiki/entities/paper-octo.md)（**复用**） |
| 03 | π-0 | VLA | [2503.06669](https://arxiv.org/abs/2503.06669) | **已开源** | [paper-pi0](../../wiki/entities/paper-pi0.md)（**复用**） |
| 04 | π-0.5 | VLA | [2503.06669](https://arxiv.org/abs/2503.06669) | **已开源** | [paper-pi05-open-world-vla](../../wiki/entities/paper-pi05-open-world-vla.md)（**复用**） |
| 05 | ViLLA | VLA | — | **部分开源** | [paper-shenlan-wm-05-villa-x](../../wiki/entities/paper-shenlan-wm-05-villa-x.md)（**复用**） |
| 06 | GR00T N1 | VLA | — | **已开源** | [isaac-gr00t](../../wiki/entities/isaac-gr00t.md)（**复用**） |
| 07 | Green-VLA | VLA | — | **部分开源** | [paper-greenvla-staged-vla-humanoid](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md)（**复用**） |
| 08 | Gemini Robotics 2 | VLA | — | **未开源** | [gemini-robotics](../../wiki/entities/gemini-robotics.md)（**复用**） |
| 09 | DreamZero | WAM | [2602.15922](https://arxiv.org/abs/2602.15922) | **已开源** | [paper-notebook-dreamzero-world-action-models-are-zero-shot-poli](../../wiki/entities/paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md)（**复用**） |
| 10 | JEPA / V-JEPA 2 | WAM | [2506.09985](https://arxiv.org/abs/2506.09985) | **已开源** | [paper-vjepa2](../../wiki/entities/paper-vjepa2.md)（**复用**） |
| 11 | AMP | Legged | [2109.05498](https://arxiv.org/abs/2109.05498) | **已开源** | [amp-for-hardware](../../wiki/entities/amp-for-hardware.md)（**复用**） |
| 12 | OpenWBT | Legged | — | **已开源** | [cn-os-openwbt](../../wiki/entities/cn-os-openwbt.md)（**复用**） |
| 13 | RMA | Legged | [2104.08776](https://arxiv.org/abs/2104.08776) | **已开源** | [paper-rma-rapid-motor-adaptation](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)（**复用**） |
| 14 | Diffusion Policy | Manipulation | [2303.04137](https://arxiv.org/abs/2303.04137) | **已开源** | [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md)（**复用**） |
| 15 | RDT-1B | Manipulation | [2410.07835](https://arxiv.org/abs/2410.07835) | **已开源** | [paper-rdt-1b](../../wiki/entities/paper-rdt-1b.md)（**复用**） |
| 16 | GraspVLA | Manipulation | — | **已开源** | [cn-os-graspvla](../../wiki/entities/cn-os-graspvla.md)（**复用**） |
| 17 | Tactile-VLA | Tactile | [2507.09160](https://arxiv.org/abs/2507.09160) | **已开源** | [paper-sa-2507-09160-tactile-vla-unlocking-vision-language-action-mod](../../wiki/entities/paper-sa-2507-09160-tactile-vla-unlocking-vision-language-action-mod.md)（**复用**） |
| 18 | ACE-Ego | 2026-VLA | [2606.11241](https://arxiv.org/abs/2606.11241) | **已开源** | [paper-sa-2606-17200-ace-ego-0-unifying-egocentric-human-and-robotic](../../wiki/entities/paper-sa-2606-17200-ace-ego-0-unifying-egocentric-human-and-robotic.md)（**复用**） |
| 19 | LingBot-VLA | 2026-VLA | [2601.18692](https://arxiv.org/abs/2601.18692) | **已开源** | [lingbot-vla](../../wiki/entities/lingbot-vla.md)（**复用**） |
| 20 | DM0 / Dexbotic | 2026-VLA | [2602.08943](https://arxiv.org/abs/2602.08943) | **已开源** | [dexmal-dm05](../../wiki/entities/dexmal-dm05.md)（**复用**） |
| 21 | Motus | 2026-WAM | [2601.04278](https://arxiv.org/abs/2601.04278) | **已开源** | [paper-sa-2512-13030-motus-a-unified-latent-action-world-model](../../wiki/entities/paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md)（**复用**） |
| 22 | EnerVerse-AC | 2026-WAM | [2511.06722](https://arxiv.org/abs/2511.06722) | **已开源** | [paper-sa-2501-01895-enerverse-envisioning-embodied-future-space-for](../../wiki/entities/paper-sa-2501-01895-enerverse-envisioning-embodied-future-space-for.md)（**复用**） |
| 23 | LeTools | 2026-Loco | — | **已开源** | [letools](../../wiki/entities/letools.md)（**复用**） |
| 24 | HumanTracker | 2026-Loco | — | **已开源** | [paper-humantracker](../../wiki/entities/paper-humantracker.md)（**复用**） |
| 25 | LIMMT / GQS | 2026-Loco | [2606.06953](https://arxiv.org/abs/2606.06953) | **已开源** | [limmt-gqs-motion-curation](../../wiki/methods/limmt-gqs-motion-curation.md)（**复用**） |
| 26 | XR-1 | VLA | [2511.02776](https://arxiv.org/abs/2511.02776) | **已开源** | [cn-os-xr-1](../../wiki/entities/cn-os-xr-1.md)（**复用**） |
| 27 | Awesome Legged Locomotion | Awesome | — | **已开源** | [awesome-legged-locomotion-learning](../../wiki/entities/awesome-legged-locomotion-learning.md)（**复用**） |
| 28 | Awesome Physical AI | Awesome | — | **已开源** | [awesome-physical-ai-natnew](../../wiki/entities/awesome-physical-ai-natnew.md)（**复用**） |
| 29 | TempoWAM | WAM | [2608.09492](https://arxiv.org/abs/2608.09492) | **待发布** | [paper-tempowam](../../wiki/entities/paper-tempowam.md)（**新建**） |
| 30 | EmbodiedBrain | Planning | [2510.20578](https://arxiv.org/abs/2510.20578) | **已开源** | [paper-embodiedbrain](../../wiki/entities/paper-embodiedbrain.md)（**新建**） |
| 31 | OpenEAI-VLA | 2026-VLA | [2606.03392](https://arxiv.org/abs/2606.03392) | **待发布** | [paper-openeai-vla](../../wiki/entities/paper-openeai-vla.md)（**新建**） |
| 32 | VLA-Adapter | VLA | [2509.09372](https://arxiv.org/abs/2509.09372) | **已开源** | [paper-vla-adapter](../../wiki/entities/paper-vla-adapter.md)（**新建**） |
| 33 | MemoryWAM | 2026-WAM | [2606.20562](https://arxiv.org/abs/2606.20562) | **已开源** | [paper-memorywam](../../wiki/entities/paper-memorywam.md)（**新建**） |
| 34 | Dexora | 2026-Manipulation | [2605.18722](https://arxiv.org/abs/2605.18722) | **已开源** | [paper-dexora](../../wiki/entities/paper-dexora.md)（**新建**） |
| 35 | Fast-WAM | WAM | [2603.16666](https://arxiv.org/abs/2603.16666) | **已开源** | [paper-fast-wam](../../wiki/entities/paper-fast-wam.md)（**新建**） |
| 36 | Embodied-AI-Daily | Awesome | — | **已开源** | [cn-os-embodied-ai-daily](../../wiki/entities/cn-os-embodied-ai-daily.md)（**新建**） |

## 对 wiki 的映射

- **36/36 独立详情节点**；**0 重复 arXiv 造页**（公众号部分 arXiv 已校正，见各实体页）
- 阅读坐标：[具身前沿算法技术地图](../../wiki/overview/embodied-frontier-algorithms-technology-map.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Locomotion](../../wiki/tasks/locomotion.md)

## 当前提炼状态

- [x] 公众号正文抓取（WebFetch）
- [x] 36 项独立节点（8 新建 / 28 复用）
- [x] 项目页/仓库开源状态核查（步骤 2.5；部分条目为索引级）
