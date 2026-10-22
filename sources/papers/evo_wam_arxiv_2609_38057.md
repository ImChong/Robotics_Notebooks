# EVO-WAM 论文来源归档

## 元数据

- **题目：** EVO-WAM: Evolving World Action Models through Video-Action Verification
- **作者：** Shiyang Zhou, Xionghao Wu, Wenbo Li, Shenghe Zheng, Jiyao Zhang, Songsong Yu, Yijun Yang, Jianhui Liu, Haoze Sun, Senqiao Yang, Li Jiang, Jingyong Su, Haoyang Huang, Zhuotao Tian
- **单位：** 抓取页面未列出
- **预印本：** https://arxiv.org/abs/2609.38057（cs.CV，cross-list cs.RO；2026-09-29 提交）
- **PDF：** https://arxiv.org/pdf/2609.38057
- **项目主页：** https://evo-wam.github.io/
- **代码：** https://github.com/Clausy9/EVO-WAM 当前仅含 README、citation.bib 与 demo 素材。项目侧 news 写明官方代码「pending approval, release date TBD」（已发布权重或数据：未见）。第三方复现：https://github.com/XiaohanLei/open-evo-wam（非官方，未核实）
- **Hugging Face Daily Paper：** https://huggingface.co/papers/2609.38057
- **入库日期：** 2026-10-22

## 论文要解决的问题

世界动作模型（WAM，同时预测未来视频与动作）需要专家示范才能适配到新的机器人任务；论文探讨能否不靠新示范、也不靠外部执行来适配。

## 方法要点（据 arXiv 摘要）

- 加入状态预测与多帧上下文，使模型可以做完整的自回归 rollout。
- 用视觉语言模型挑选完成任务的前缀，并以逆动力学模型检验视频与动作的一致性。
- 只保留经验证的前缀，迭代地在其上重训练。

## 报告的结果（据 arXiv 摘要）

- RoboTwin 2.0 的 7 个任务：Cosmos3 26.9% → 68.0%，DreamZero 28.5% → 46.4%。
- 真机 3 个长时程任务：Cosmos3 20.0% → 76.7%。

## Robotics_Notebooks 映射

- 实体页：[EVO-WAM](../../wiki/entities/paper-evo-wam.md)

## 一手来源

- arXiv 摘要页与项目页（经 WebFetch 摘要）
