# AffectiveVLA（arXiv:2611.07712）

> 来源归档（paper）

- **标题：** AffectiveVLA: Safe and Intent-Aware VLA Control for Human–Humanoid Social Interaction
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.07712>（v1，2026-11-05，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.07712>
- **作者 / 机构：** Nikita Kuzmin、Georgii Demianchuk、Mariya Lezina、Yara Mahmoud、Iana Zhura、Miguel Altamirano Cabrera、Dzmitry Tsetserukou；全部署名机构为 Skolkovo Institute of Science and Technology（斯科尔科沃科学技术学院，莫斯科，首页作者栏）
- **代码：** 论文正文未给出代码 / 项目页链接（步骤 2.5：全文检索无 github 或项目页地址；未另行核查网络，开源状态按「正文未声明」记）
- **入库日期：** 2026-11-09
- **一句话说明：** 在 Unitree G1 上把「RGB 意图识别 → 语言提示 → VLA 全身动作 → 按交互类型调阻抗 → 语音锁存急停」串成流水线，用 VR 遥操作采集约 1,300 条社交触碰演示，微调并对比 4 个 VLA。
- **核对范围：** 通读 arXiv v1 PDF 全部 5 页正文（摘要、§1 引言、§2 相关工作、§3 系统与数据集、§4 方法、§5 实验、§6 结果与局限、参考文献），含表 1 至表 3、式 (1)，并查看第 3 页版面与图 2 题注；图 1、图 2 的图内细节仅读到题注文字。论文提到的补充材料（意图模型六种配置的完整结果）未获取，未读。
- **通读：** 已完成

## 核心摘录

1. **问题设定（§1）：** 握手、碰拳、拥抱要求人形机器人推断人的意图、生成接触丰富的动作并保证接触安全；作者称端到端 VLA 在意图不确定时有误触发风险。作者归纳现有工作：学习式全身策略依赖操作员显式指令 [7]，顺应控制多用固定参数 [11]，视觉语言流水线延迟高且不执行接触 [2]。
2. **系统（§3、§4）：** Unitree G1 + BrainCo 灵巧手，SONIC 全身控制器（上身 17 个电机），PICO 4 Ultra + XRoboToolkit 遥操作；统一 76 维动作空间（64 维 SONIC 运动 token + 每手 6 维手指指令），动作块 40 步，VLA 5 Hz 推理、50 Hz 执行（§3.1、§3.3、§4.2）。
3. **数据（§3.2）：** 最终 1,270 条 episode、19 名参与者；扩展集 982 条（15 人）与测试集 288 条（4 名未见参与者）不重叠；四类标签含 no interaction。
4. **意图门控（§4.1）：** 双路 RGB，16 帧对、15 Hz、约 1 秒历史；TinyViT 编码 + 因果 VideoLightFormer + 注意力融合；概率超过 0.6 且连续 3 个推理步才提交，否则禁止运动执行。
5. **顺应性与急停（§4.3、§4.4）：** 仅缩放 14 个手臂关节的 Kp、Kd，约束 β_i ≥ √α_i（式 1）；表 1 给出 Handshake / Hug / FistBump / Soft 四套档位，作者注明为试点会话得到的临时值。Vosk 离线识别「stop」「hold on」触发锁存急停，仅手动解除。
6. **结果（§5、§6）：** 离线 RMSE 四个 VLA 相近（0.133–0.148，表 2）；在线 150 次试验 / 策略，GR00T 单独宏 F1 0.75，加意图识别为 0.90，碰拳 0.50 到 0.78（表 3）；意图模型测试宏 F1 0.76、准确率 0.736、p95 延迟 6.27 ms（§5.3）。
7. **局限（§6.1）：** 参与者少、仅三种预设交互、单一机器人与全身控制器；评测基于带标签 episode，长时间无交互时的误激活待研究；意图模型对未见参与者（尤其碰拳）泛化可能差；嘈杂环境语音识别可能失败；降低 Kp 无接触力的形式化界，且无重力补偿时手臂略有下垂。

## 对 wiki 的映射

- [paper-affectivevla](../../wiki/entities/paper-affectivevla.md)
