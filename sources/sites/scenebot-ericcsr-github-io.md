# ericcsr.github.io/scenebot（SceneBot 项目页）

- **标题：** SceneBot — Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction
- **类型：** site / project-page
- **URL：** <https://ericcsr.github.io/scenebot/>
- **入库日期：** 2026-06-29
- **配套论文：** [SceneBot（arXiv:2606.27581）](https://arxiv.org/abs/2606.27581) — 归档见 [`sources/papers/scenebot_arxiv_2606_27581.md`](../papers/scenebot_arxiv_2606_27581.md)

## 一句话摘要

Amazon FAR / Stanford / CMU 团队的 **SceneBot** 官方站点：强调 **单一策略** 覆盖自由空间、地形与物体交互；提供 **浏览器内 MuJoCo 实时交互 demo**（G1 人形，键盘控制走/转/上下箱/坐站/踢球与示例序列），以及物体交互、地形穿越、重建场景与遥操作等真机视频板块。

## 公开信息要点（截至入库日）

- **机构：** Amazon FAR、Stanford University、CMU；* equal contribution，† Amazon FAR team co-lead。
- **核心主张：** All demos are using **single policy**；motion and contact tracking in **real-time**。
- **交互 demo：** 独立页面流式加载模型/策略/资产；推荐笔记本或台式机；按键示例序列 **Q-L-L-E-W-N-G-W-Z-P**（上一动作结束后再按下一键）。
- **控制说明（项目页）：**
  - W/A/S/D — 前进/后退/转向
  - Q/E — 原地旋转
  - N/Z — 踩箱/下箱
  - G/P — 搬起/放下箱子
  - L — 坐/站切换
  - K — 踢球；Enter — 自动跑示例序列；Backspace — 重置场景
- **演示板块：** Object Interaction、Terrain Traversal、Object + Terrain、Free Space（舞蹈/踢球/跑步）、Reconstructed Scenes（搬箱、LAFAN 地形、坐椅、楼梯）、Teleoperation。
- **资源状态：** Paper / **Data (Coming Soon)** / **Code (Coming Soon)** / Related Research 链接。

## 为何值得保留

- **非 PDF 证据：** 浏览器可玩 demo 与分区视频比摘要更直观呈现 **contact-prompted tracking** 的长时程组合能力（如箱+楼梯）。
- **与 arXiv 三角互证：** 单策略叙事、交互按键与论文方法节（contact label + reconstruction pipeline）一致，便于维护者核对表述。
- **同系工作锚点：** 与 SONIC、OmniRetarget、CHIP 同属 **通才人形 motion tracking / loco-manipulation** 线；SceneBot 突出 **接触条件接口 + 事后场景重建数据引擎**。

## 关联资料

- 论文归档：[`sources/papers/scenebot_arxiv_2606_27581.md`](../papers/scenebot_arxiv_2606_27581.md)
- 对照 tracker：[`wiki/methods/sonic-motion-tracking.md`](../../wiki/methods/sonic-motion-tracking.md)
- 场景数据对照：[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)
