# VersaCamVLA: Camera-Configurable VLA Policies for Robotic Manipulation（arXiv:2610.12451）

> 来源归档（ingest）

- **标题：** VersaCamVLA: Camera-Configurable VLA Policies for Robotic Manipulation
- **类型：** paper / vision-language-action / manipulation / multi-view
- **arXiv：** <https://arxiv.org/abs/2610.12451>（v1 提交于 2026-10-08）
- **项目页：** <https://boyaohan.github.io/VersaCamVLA.github.io/>
- **作者：** Boyao Han、Chen Shi、Jingjing Qian、ZhuoTan Tian、Li Jiang
- **机构：** 香港中文大学（CUHK；按项目资料）
- **发表状态：** arXiv 页面标注 NeurIPS 2026 接收
- **一句话说明：** 通过固定大小场景 token 接口，把任意数量与位姿的 RGB 相机视图接入预训练 VLA，以提升部署时对相机数量和未见视角变化的适应性。

## 摘录 1：问题与主张

- 论文指出，常见 VLA 在固定相机配置下训练；部署时相机数量或位姿变化会造成脆弱性。
- VersaCamVLA 将相机集合表征与动作学习解耦：将任意数量、带位姿信息的 RGB 视图映射为固定大小的潜在场景 token。
- **对 wiki 的映射：** 说明多视角输入接口为何需要与动作策略解耦，见实体页「核心机制」。

## 摘录 2：训练中的视角覆盖

- 方法使用多信号目标视图预测，以及 Wrist-Augmented Pose Sampling（WAPS）；后者利用腕部相机的自然运动提供额外视角多样性。
- **对 wiki 的映射：** 将 WAPS 作为相机位姿数据覆盖策略记录，见实体页「核心机制」。

## 摘录 3：部署与评测

- 部署时，轻量空间编码器将场景 token 作为补充视觉条件注入预训练 VLA；论文称无需显式 3D 感知或新视角渲染。
- 论文摘要报告在 RoboTwin、LIBERO 与真实机器人平台上评测，并称在相机数量变化及未见相机位姿下优于既有 VLA 与直接多视角基线；本归档不补写摘要未给出的具体数值。
- **对 wiki 的映射：** 在「实验与评测」与「局限与风险」中区分摘要级结论和完整指标。

## 开源状态

官方项目页关联的 [BoyaoHan/VersaCamVLA 仓库](https://github.com/BoyaoHan/VersaCamVLA) README 截至 2026-10-27 标注 “Code coming soon”；仓库目前提供项目说明与演示素材，尚无可核验的训练或推理运行入口。因此当前可查看演示，不能据此复现实验。

## 一手来源

- 论文：<https://arxiv.org/abs/2610.12451>
- 项目页：<https://boyaohan.github.io/VersaCamVLA.github.io/>
- 官方仓库：<https://github.com/BoyaoHan/VersaCamVLA>
