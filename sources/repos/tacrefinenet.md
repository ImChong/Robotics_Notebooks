# TacRefineNet（NoneJou072/tacrefinenet）

> 来源归档

- **标题：** TacRefineNet — 代码仓库（占位）
- **类型：** repo
- **组织：** 小米机器人实验室（Xiaomi Robotics）相关作者；GitHub 账号 `NoneJou072`
- **代码：** <https://github.com/NoneJou072/tacrefinenet>
- **项目页：** <https://sites.google.com/view/tacrefinenet>
- **论文：** <https://arxiv.org/abs/2509.25746>
- **入库日期：** 2026-07-23
- **一句话说明：** 项目页 Code 按钮指向的官方仓；截至入库日为空仓库（无 README、训练/推理入口），属 **宣称将开源 / 待发布**。
- **沉淀到 wiki：** [TacRefineNet（论文实体）](../../wiki/entities/paper-tacrefinenet-tactile-grasp-refinement.md)

## 开源状态（2026-07-23 核查）

| 项 | 结论 |
|----|------|
| 仓库存在 | 是（`main` 创建于 2026-07-15） |
| 可运行内容 | **否** — GitHub Contents API 返回 empty；无训练/推理脚本 |
| 数据集 | 项目页 Dataset 未指向独立发布；勿假设可下载 |
| 复现建议 | 待作者填充 README 后再跟进；当前 wiki 不写源码运行时序图可运行路径 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Tactile Sensing](../../wiki/concepts/tactile-sensing.md) | 压阻 taxel 图 → ViT 触觉编码 |
| [Visual Servoing](../../wiki/methods/visual-servoing.md) | 触觉对偶：目标触觉图伺服腕部增量 |
| [Sim2Real](../../wiki/concepts/sim2real.md) | MuJoCo 触觉仿真全程训练 → 真机零样本 |

## 为何值得保留

- 即使仓为空，项目页已挂官方 Code URL，需归档以免 wiki 误写「暂无代码链接」。
- 后续 lint / 复现跟进时，本页是「是否已填充可运行入口」的检查锚点。
