# Being-M0.7 官方项目页

> 来源归档；核查日期：2026-10-12

- **类型：** site / humanoid latent world-action model
- **项目页：** <https://research.beingbeyond.com/being-m07>
- **页面日期：** 2026-07-14
- **官方 PDF：** <https://research.beingbeyond.com/being-m07/being-m07.pdf>
- **arXiv：** <https://arxiv.org/abs/2610.11283>（v1 首次提交 2026-10-08）
- **机构：** 超越智能（BeingBeyond）
- **GitHub 导航：** <https://github.com/BeingBeyond>，链接指向组织而非 M0.7 实现
- **一句话说明：** 用人中心视频/运动混合监督学习潜在未来，再以动作专家接地到 G1 移动操作。

## 页面内容与版本关系

页面展示混合模态数据、MoT 先验、VR 遥操作数据采集与 G1 演示。预训练原始数据超过一万小时，但包含内部及商业获得来源，不代表整套语料公开。

页面以两段式 pre-training/post-training 介绍，并描述未来 video-motion 计划；10 月 arXiv 明确加入 robot mid-training，随后冻结 prior，动作阶段关闭显式 motion 预测。方法与评测需以指定版本原文为准，不将页面简介的频率概括当作实际专家调用 Hz。

## 开源核查

| 资源 | 核查结论 |
|------|----------|
| 论文、项目介绍、演示 | 公开 |
| M0.7 训练/推理实现 | 页面未列专属仓库；组织仓库与名称检索未找到可运行入口 |
| M0.7 预训练/动作专家权重 | 未找到项目专属下载入口 |
| 处理后的混合语料及 G1 示范 | 未找到完整发布入口；公共数据源各自有许可 |
| Being-H / Being-M0.5 | 不同项目，其仓库不作为 M0.7 的代码开放证据 |

因此记录为“公开论文与演示，未发现 M0.7 可获取实现/权重/语料入口”，不将无法查到视为永久不开源，也不虚构代码许可证。

## 对 wiki 的映射

- [Being-M0.7 统一实体](../../wiki/entities/paper-being-m07-humanoid-latent-wam.md)
- [论文归档与版本记录](../papers/being_m07.md)
