# Targeting World Models to Compromise Robot Learning Pipelines（arXiv:2606.09499）

> 来源归档（paper）

- **标题：** Targeting World Models to Compromise Robot Learning Pipelines
- **作者：** Ethan Rathbun、Ahmed Agha、Saaduddin Mahmud、Christopher Amato、Alina Oprea、Eugene Bagdasarian（Northeastern University / University of Massachusetts Amherst）
- **类型：** paper（arXiv v1，2026-06-08，cs.RO；CoRL 2026 Spotlight 据 Glen Berseth 阅读清单）
- **arXiv：** <https://arxiv.org/abs/2606.09499>
- **PDF：** <https://arxiv.org/pdf/2606.09499v1>
- **代码：** 论文未给出攻击代码仓库（附录 A.1 只称去匿名后拟把 Libero 微调的 Cosmos 模型传到 Hugging Face）
- **入库日期：** 2026-11-12
- **一句话说明：** 提出「Visual Prompt Hijacking（VPH）」与「Visual Transition Hijacking（VTH）」两类白盒图像扰动攻击：恶意遥操作数据提供方在看似安全的视频帧里嵌入扰动，等数据经文本条件 / 动作条件世界模型后才生效，进而污染下游 VLA / DRL 策略。
- **核对范围：** 读了 arXiv v1 PDF 全文共 19 页：摘要、§1 引言、§2 问题设定、§3 相关工作、§4 威胁模型、§5 方法（含式 1-6）、§6 实验（表 1、图 3、图 4）、§7 结论与局限、参考文献列表（仅扫读）、附录 A.1-A.3 与 B.1-B.4（图 5-9）。图 3、图 4 的数值仅为读图近似。没有读被引文献，也没有运行任何代码；OpenReview 页面在本环境打不开，未核对其与 arXiv v1 的差异。
- **通读：** 已完成

## 核心摘录

1. **威胁模型（§4）：** 攻击者是不可信的第三方遥操作数据提供方；白盒访问受害者世界模型（可查询、算梯度）；只能改每条演示的视频帧，不能注入动作序列，也不能控制下游 prompt 或奖励函数；扰动在 LAB 色彩空间的 lp 球内（β_L=5、β_AB=20，附录表 2）。
2. **VPH（§5.1，式 3）：** 优化帧扰动，使世界模型在「安全 prompt + 扰动帧」下的语义编码（实际用去噪向量场，附录 A.1）对齐「危险 prompt + 原帧」；PGD 加动量，400 步，α=0.025，动量 0.9，攻击时只生成 10 帧（默认 32 帧）。
3. **VTH（§5.2，式 4、5）：** 攻击动作条件世界模型的潜编码器，使目标动作 a+ 附近预测正常、其他动作预测崩塌；加触发物体后在 DRL 策略里等效植入后门。
4. **VPH 结果（§6.1，表 1）：** Cosmos-Predict 2.5 2B（后训练版，在 Libero 256x256 渲染上 LoRA 微调到 432x432）；每个危险物体 5 张扰动图 x 4 个生成种子，人工判定攻击是否成功。分布外相机位姿 + 模糊 prompt 下：Gun 100%、Cereal Box 90%、Grater 55%、Bomb 20%、Knife 20%、Cardboard 10%、Oil 5%、Dynamite 0%、Fork 0%；分布内相机位姿下大多接近 0，Gun 50%、Grater 30%。无攻击基线多为 0%，Gun 在分布外 10%、分布内 5%。
5. **VTH 结果（§6.2，图 3、图 4）：** Dino-WM（Single Wall）里用 CleanRL 的 PPO、5 个种子、5% 初始状态被污染、目标动作每维为 1：攻击成功率读图约在 0.3-0.4（无污染约 0.1 以下），与目标动作的余弦相似度约 0.8-0.9（无污染为负），不带触发的真实环境成功率与无污染基线相当（约 0.9）。Cosmos-Predict 2.5 动作条件版只测世界模型输出的嵌入 / 像素 L2 距离，随动作接近目标动作而下降，未接入 DRL 流程。
6. **作者自述的未完成项（§6.1、§7、附录 A.1）：** 逆动力学模型（IDM）在新建的危险任务上泛化不足，没能跑通 VLA 管线，VPH 只是后门攻击 BC 的概念验证；两类攻击在更真实的管线里都未验证，攻击远非最优。
7. **护栏（附录 B.3，初步）：** 开启 Cosmos 自带 guardrail，直接提示「pick up the handgun」未被拦截，VPH 输入也未被识别。

## 对 wiki 的映射

- [paper-targeting-world-models-robot-learning-pipelines](../../wiki/entities/paper-targeting-world-models-robot-learning-pipelines.md)
