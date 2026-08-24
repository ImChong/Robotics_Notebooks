# gigabrain-wbc-0.5（GigaBrain-WBC-0.5 项目页）

- **标题：** GigaBrain-WBC-0.5 — A Behavior World Model for Robust Whole-Body Control with Environment Interaction
- **类型：** site / project-page
- **URL：** <https://shepherd1226.github.io/gigabrain-wbc-0.5/>
- **arXiv：** <https://arxiv.org/abs/2608.18234>
- **入库日期：** 2026-08-21（开源状态复核：2026-08-24）
- **配套论文：** [GigaBrain-WBC-0.5（arXiv:2608.18234）](../papers/gigabrain_wbc_0_5_arxiv_2608_18234.md)

## 一句话摘要

GigaAI / 清华等联合提出的 **Behavior World Model（BWM）** 官方站点：因果 Transformer 联合预测 action、next state 与 next latent command 分布；自动 motion–terrain 标注管线从 retarget 运动恢复 3D 接触几何；部署期用 GMM Mahalanobis 在线 retract OOD 命令。展示相对 SONIC 的真机物体/地形交互、极端鲁棒性与 G1→Maker L01 迁移。

## 公开信息要点（截至 2026-08-24 核查）

- **机构：** 清华大学（Tsinghua University）；极佳视界（GigaAI）；上海理工大学（USST）；北京交通大学（BJTU）；中国科学院自动化研究所（CASIA）；中国科学院大学（UCAS）
- **作者：** Ziyang Cheng、Tianshu Tang、Jinxin Lan、Xinze Chen、Yuhan Gong、Zhichao Liu、Changzhong Wu、Yahao Mao、Zongyan Deng、Mingxuan Ma、Huasen Xi、Yilong Liu、Yutong Wu、Xiaofeng Wang、Yang Wang、Yun Ye、Guan Huang、Xiaojie Jin、Zheng Zhu#、Jiwen Lu#
- **平台：** Unitree G1（29 DoF，50 Hz）；跨具身 Maker L01（简单 fine-tune）。
- **核心叙事：**
  - **BWM** — 非纯 reactive tracker；policy 建模「环境如何塑造下一步可行为」
  - **Terrain annotation** — 从 retarget 轨迹恢复 chairs / tables / stairs / boxes 等 **全 3D 几何**（非 2.5D height field）
  - **Online filter** — 用自身预测的 next-command GMM 做 stateless Mahalanobis retract，best-effort 而非急停
- **能力对照表：** 相对 GMT/TWIST/SONIC/HoloMotion-1/Humanoid-GPT/SceneBot/CMP/BFM-Zero，宣称唯一同时覆盖 diverse tracking、teleop、terrain/object interaction、OOD robust、fall robust。
- **Sim-to-sim（MuJoCo，Table 核心）：** Terrain SR **81.3%**（最强基线 15.3%，4.3×）；OOD SR **83.1%**；Fall recovery **99.3%**（16.8× 最强基线 5.9%）。
- **真机：** 与 SONIC 同指令并排对比（搬箱、灭火器、上平台、坐椅/坐箱）；缺失支撑/扰动/OOD 命令 best-effort 演示；完整真机 footage 标注 forthcoming。
- **代码 / 数据（步骤 2.5，2026-08-24 复核）：** 页面 Resources 区 **Code → coming soon**；**无** GitHub / Hugging Face 链接。按 **宣称将开源 / 待发布** 处理。

## 为何值得保留

- **非 PDF 证据：** SONIC 并排真机视频、能力矩阵与四 regime 结果表比 arXiv 更易扫读。
- **与 SceneBot / CMP 对照：** 同为环境交互 tracker，但命令通道为在线 reference window（非 per-link contact label），且 world-model 预测驱动 OOD filter。
- **数据规模线索：** 训练混合 Bones-Seed / MotionMillion / MotionDecode 中识别出的 terrain-interaction 子集（论文 §3.5）。

## 关联资料

- 论文归档：[`sources/papers/gigabrain_wbc_0_5_arxiv_2608_18234.md`](../papers/gigabrain_wbc_0_5_arxiv_2608_18234.md)
- 同系列：[`wiki/entities/paper-sa-2510-19430-gigabrain-0-a-world-model-powered-vision-languag.md`](../../wiki/entities/paper-sa-2510-19430-gigabrain-0-a-world-model-powered-vision-languag.md)
- 跟踪选型：[`wiki/queries/humanoid-motion-tracking-method-selection.md`](../../wiki/queries/humanoid-motion-tracking-method-selection.md)
