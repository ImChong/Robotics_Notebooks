# DyPES-VLA: Learning Shared Dynamics Priors and Embodiment-Specific Control for Cross-Embodiment Manipulation（arXiv:2608.06374）

> 来源归档（ingest）

- **标题：** DyPES-VLA: Learning Shared Dynamics Priors and Embodiment-Specific Control for Cross-Embodiment Manipulation
- **缩写 / 框架：** **DyPES-VLA**（Dynamics Priors + Embodiment-Specific control）
- **类型：** paper / vla / cross-embodiment / moe / world-dynamics / manipulation
- **arXiv：** <https://arxiv.org/abs/2608.06374>（Submitted 2026-08-06；PDF：<https://arxiv.org/pdf/2608.06374>）
- **项目页：** <https://livfour.github.io/DyPES-VLA_RELEASE/> — 归档见 [`sources/sites/dypes-vla-github-io.md`](../sites/dypes-vla-github-io.md)
- **作者：** Junfeng Li、Junjie He、Zhide Zhong、Yangyang Zheng 等；通讯 Haoang Li
- **机构：** 香港科技大学广州校区（HKUST-GZ）；可可矩阵（COCO Matrix，上海）
- **入库日期：** 2026-08-08
- **一句话说明：** 跨本体 VLA：用未来帧预测学共享动力学先验（query），再以本体特化 MoE 动作头直接在原生动作空间出控，避免手工统一动作格式。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-08）：** 前端按钮为 **Code (coming soon)**（disabled）；无 GitHub 训练/推理仓。
- **结论：** **宣称将开源 / 尚未发布**（截至入库日无可运行实现）。

## 摘录 1：问题与主张（§1）

- **痛点：** 跨本体 generalist VLA 常 (i) 仅靠动作监督、欠用共享动力学；(ii) 手工把异构动作映到统一格式，难扩展且纠缠「共享交互规律」与「本体控制语义」。
- **主张：** 共享动力学先验（未来预测监督 query）+ 本体特化控制（MoE 动作头原生空间）应解耦；未来预测**只服务先验**，不走 WAM 式未来–动作耦合生成。
- **三族本体：** 单臂（Franka Panda / FR3）、双臂（ALOHA-AgileX / COBOT Magic）、人形（Fourier GR-1 / Unitree G1 + Inspire RH56DFQ）。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-dypes-vla.md`](../../wiki/entities/paper-dypes-vla.md)；与 [VLA](../../wiki/methods/vla.md)、[Any2Any](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md)、[Qwen-VLA](../../wiki/entities/qwen-vla.md) 互链。

## 摘录 2：方法（§3）

- **骨干：** Qwen3-VL-2B + learnable query；未来生成头 SANA-600M；Stage 1 无动作视频预训练；Stage 2 跨本体示范共训。
- **MoE 动作头：** 共享 attention 捕时间结构；静态 router 选本体特化 encoder / FFN expert / decoder；flow matching（4 Euler steps）。
- **推理：** 一次 VLM 前向得 query → MoE 出原生动作 chunk；无测试时视频生成。

**对 wiki 的映射：** 实体页强调「先验共享 / 控制特化」与 WAM 未来–动作耦合的差异。

## 摘录 3：评测（§4）

| 基准 | 设定 | DyPES-VLA | 关键对照读点 |
|------|------|-----------|--------------|
| LIBERO | 单臂 7-DoF | **98.0%** | 略超 Fast-WAM 97.6% / OpenVLA-OFT 97.1% |
| RoboCasa-GR1 | 人形 | **59.25%** | 超 Qwen-VLA 56.7%、LDA-1B 55.4% |
| RoboTwin 2.0 | 双臂 | **89.02%**（两设定均值） | 超 Qwen-VLA ~2.4 pt |
| 真机三本体 × 三任务 | 单 checkpoint 联合微调 1800 demos | **75.6%** 均值 | vs ACT 32.4%、GR00T-N1.6 59.6% |

- 消融：去掉未来监督 / 用共享 dense 头替换 MoE，均掉点；线性探针显示未来监督提升接触 onset/release 可解性。

**对 wiki 的映射：** 用跨本体榜 + 真机统一策略数字写选型读法。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-dypes-vla.md`**；**`sources/sites/dypes-vla-github-io.md`**。
- 注册机构 **coco-matrix**；交叉更新 VLA 方法页。
