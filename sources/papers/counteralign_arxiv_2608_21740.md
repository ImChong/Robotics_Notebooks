# CounterAlign（arXiv:2608.21740）

> 来源归档（paper）

- **标题：** CounterAlign: Counterfactual Supervision for Vision-Language-Action Models
- **类型：** paper
- **作者：** Haru Kondoh、Kei Ota、Asako Kanezaki、Yueh-Hua Wu（Institute of Science Tokyo / AIRoA / RIKEN AIP / 东北大学）
- **arXiv：** <https://arxiv.org/abs/2608.21740>
- **PDF：** <https://arxiv.org/pdf/2608.21740v1>（v1，2026-08-22）
- **项目页：** <https://counteralign.airoa.io>
- **代码：** 未在论文与项目页首页找到公开仓库链接
- **入库日期：** 2026-11-12
- **一句话说明：** 用「成功专家演示 + 指令 / 动作错配重标注」合成反事实负样本，配合对抗判别器与 nnPU 判别器学一个指令—观测—动作对齐奖励，再以 IQL 加优势加权 flow matching 做 VLA 离线 RL。
- **核对范围：** 读了 arXiv v1 PDF 全文：摘要、正文 1–5 节（引言、相关工作、方法、实验、结论）、参考文献、补充材料 1–10 节（相似度函数、重标注算法 1–3、actor 侧重标注、完整算法 4、AWFM 推导、参数量、训练配置、消融分析、真机数据集、CALVIN 附加实验）；图 1–4 只看了文字描述与图注，未逐像素核对图像；未运行任何代码，未做独立复现；OpenReview 页面本环境打不开，未读评审意见。
- **通读：** 已完成

## 核心摘录

1. 问题（第 1 节）：BC 只给正监督；真实机器人的非专家 / 失败轨迹难采，奖励难定义。
2. 方法（第 3 节）：对抗判别器 D_adv（专家 vs 策略生成动作，类 GAIL，纯离线）；重标注判别器 D_rel（原样本为正，指令重标注与动作重标注为负，联合重标注为 unlabeled，nnPU 训练，加熵正则）；奖励 r = (1-w)·log D_adv + w·log D_rel（式 3）。
3. 策略训练（第 3.3 节）：IQL（clipped double Q）；critic 额外用近正样本的指令重标注数据；actor 用 exp(β·(Q−V)) 加权的 flow matching 损失（式 6），基座 π0.5。
4. 仿真（第 4.1–4.3 节）：LIBERO 训练、LIBERO-PRO 评测（Obj / Pos / Sem / Task 四种扰动，每项 100 trials），Pos 与 Task 提升最大，Sem / Obj 无明显差距（作者自述）。
5. 真机（第 4.4–4.5 节、补充 9）：TX-G2（作者称兼容 AGIBot G2），4 个任务，演示共 1,198 条，每任务 10 次扰动初始位姿的试验。
6. 补充（补充 6、7、10）：BC 约 3.4B 参数，方法约 12.7B；LIBERO 训练至少 5 张 H200、约 28 小时；CALVIN ABC→D 平均长度 3.93 → 4.03。

## 对 wiki 的映射

- [paper-counteralign-counterfactual-supervision-vla](../../wiki/entities/paper-counteralign-counterfactual-supervision-vla.md)
