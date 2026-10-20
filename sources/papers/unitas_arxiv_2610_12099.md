# UNITAS 论文来源归档

## 元数据

- **题目：** UNITAS: A 3D-Native World Action Model for Embodied Manipulation
- **作者：** Ruixiang Wang, Yongyi Su, Wenlve Zhou, Bo Yue, Hengyan Liu, Dekun Lu, Yuxin Tian, Yihan Fang, Zerui Wu, Xing Hu, Jietao Chen, Yong Guo, Ziyan He, Junbin Yuan, Guiliang Liu, Xiaofen Xing, Kui Jia（Ruixiang Wang、Yongyi Su 为共同一作，Yongyi Su 通讯）
- **单位（项目页）：** 香港中文大学（深圳）、DexForce Technology、佛山大学、中山大学、华南理工大学；arXiv 摘要页未列单位
- **预印本：** https://arxiv.org/abs/2610.12099 （v1，2026-10-08，cs.RO；27 页、11 图、13 表，Under review）
- **HTML：** https://arxiv.org/html/2610.12099v1
- **项目页：** https://unitas-wam.github.io （见 [unitas-project.md](../sites/unitas-project.md)）
- **代码：** https://github.com/DexForce/UNITAS （2026-10-20 核验：已发布推理与评测代码、模型权重及安装/评测说明；训练代码与训练数据集仍未发布；Apache-2.0）

## 论文要解决的问题

现有世界–动作模型（WAM）多建立在预训练视频生成器上，用图像或视觉潜变量表示世界。作者认为这与机器人在度量三维空间中行动不匹配：图像是依赖视角的投影，像素距离不直接对应物理距离。UNITAS 要在同一个度量 3D 坐标系里统一表示观测、动作与场景动力学，并跨机器人本体与人手共享。

## 方法与训练信息

1. **两类点轨迹表示：** action flow 把夹爪/手表示为 3D 点轨迹（每个夹爪按 URDF 几何采样 20 点并经正运动学映射；人手使用 MANO 的每手 21 个关键点）；scene flow 预测场景点在动作条件下的运动（背景点来自深度，模拟中 5 mm 体素化，上限 12,000 点；混合数据训练里约保留 1%，约 120 个查询点）。
2. **世界对齐 3D 位置编码：** 有深度时用相机内参与相机到世界变换反投影，再经固定傅里叶映射与 Enc3D 加到 patch token；无深度时沿相机射线均匀采 M=64 个候选深度（0.1–2.0 m），按 patch 特征做注意力路由得到嵌入，并用 L_PE（λ_PE=0.1，对深度派生目标 stop-gradient）监督，训练中以 0.1 概率用路由特征替换深度特征。
3. **轨迹 tokenizer：** 每条点轨迹编码为一个 token（与长度、采样率无关），时间以秒计（T_max=3 s，d_z=256），时间 MLP 产生 FiLM 参数，编码器把 H 步位移池化成单个潜变量，解码器按指定时间戳读出；当前 3D 点的傅里叶特征在去噪全程加到 token 上。tokenizer 约 1.9M 参数，先训练后冻结。
4. **主干：** 层耦合的 Mixture-of-Transformers，合计约 1.7B（1,694.5M）：Florence-2-large VLM（570.7M）；点动力学专家（28 层、宽 1,280、20 头，826.3M）建模 action/scene flow；动作专家（28 层、宽 640、10 头，256.0M）输出末端执行器动作块（相对平移、旋转、夹爪开度），每层读取动力学专家的观测/语言 K/V。
5. **注意力与推理：** 观测/语言/历史上下文双向可见；scene token 只看干净（非加噪）的 action-flow token，动作块 token 跳过 flow token，故仅做策略推理时可省略两路 flow 分支。上下文 K/V 在去噪步间缓存；每个激活专家 10 步 Euler。
6. **损失：** 动作流、场景流、动作块均用 rectified flow + MSE，权重各 1.0，外加 L_PE；缺失目标与缺失的臂槽被掩码。
7. **训练三阶段：** 先训 tokenizer 并冻结；可选混合数据预训练 200K 步、batch 1,024、AdamW、权重衰减 1e-8、warmup 1,000 步后常数学习率（动力学/动作专家 1e-4，VLM 1e-5）、图像 224×224、64 张 BW1000_H DCU（每张 64 GiB）+ FSDP；再按基准或真机数据微调（另有不预训练变体）。
8. **预训练数据：** 九个来源——RoboTwin、LIBERO、VLABench（仿真），DROID、AgiBot、RoboMIND、RH20T、自采 W1 数据（真机），VITRA（第一视角人手）；未来视野 H=20 步（VITRA 为 10），采样 6–20 Hz，AgiBot 与 RoboMIND 由 45 Hz 降到 15 Hz。

## 论文报告的结果

- **LIBERO：** 平均 99.8%（Spatial 99.8、Object 100.0、Goal 100.0、LIBERO-10 99.4）；不预训练为 99.5%。
- **RoboTwin 2.0（50 个双臂任务）：** Clean 92.94%、Randomized 93.04%；对比 Fast-WAM（6B）91.88/91.78、Motus（8B）88.66/87.02。
- **VLABench：** SR 56.2%、PS 67.6、IS 69.3，π0.5 的 SR 为 48.1%。
- **LIBERO-Plus：** 总体 89.1%，相机扰动下 82.1%（Cosmos Policy 75.8%、LaMP 64.5%）；预训练带来 +2.6 点，机器人初始状态扰动从 84.8% 到 94.2%。
- **动作条件场景预测（RoboTwin 保留 rollout）：** Moving ADE 0.427 cm、Static ADE 0.043 cm、Moving FDE 0.825 cm、Acc@1cm 92.31%；PointWorld 为 0.656/0.084/1.050/88.22%；X-WAM+STv2 为 2.286/0.122/3.886/49.69%。相对 PointWorld 为 Moving ADE 低 35%、Moving FDE 低 21%、Static ADE 低 49%（摘要所述“最高 49%”对应 Static ADE）。
- **真机（20 次试验/方法/任务）：** Organize Books 18/20（π0.5 15/20、Fast-WAM 13/20）；Pack the Ball 16/20（Fast-WAM 12/20、π0.5 11/20）；两任务平均 85%。（论文引言一处写 Pack the Ball 为 65%，与表 2 的 80% 不一致，此处取表值。）
- **消融（LIBERO-Plus，无预训练）：** 去 3D 位置编码 85.74%→81.35%（相机扰动 79.11%→66.73%）；去 action-flow 监督为 78.89%；去历史 token 为 75.92%；三者全去为 67.58%（类 VLA 基线）；加 scene-flow 监督 85.74%→86.47%；无深度路由 PE 84.43% 对比真值深度 84.68%。
- **推理延迟（RTX 5090 D，每块）：** 仅动作 55.77 ms；动作+action flow 122.51 ms；动作+两路 flow 305.49 ms（1,024 个场景查询）/ 561.60 ms（3,000 个）。
- **tokenizer 重建：** action-flow ADE/FDE 约 1.7–4.7 / 3.3–8.2 mm（视数据源）；RoboTwin scene-flow ADE/FDE 1.01 / 2.32 mm。

## 复现与解释边界

- 论文无独立局限节；结论提到更长交互预测与更广真机部署为未来工作。场景预测推理需要当前 3D 查询坐标（来自深度或外部几何）。真机评测只有两个任务。
- 官方仓库现已提供推理与评测代码、模型权重、安装说明及基准评测指南；训练代码和训练数据集仍未发布。完整复现仍受限于数据与训练流程未公开，论文报告预训练使用 64 张 BW1000_H DCU。
- 本归档数字取自 arXiv HTML 前 100K 字符的抽取，末段（约 17K 字符）未读。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-unitas-3d-native-wam.md
- 概念入口：wiki/concepts/world-action-models.md
- 周报汇总：wiki/overview/frontier-manipulation-2026-10-05-10-09.md

## 一手来源

- <https://arxiv.org/abs/2610.12099>
- <https://arxiv.org/html/2610.12099v1>
- <https://unitas-wam.github.io>
- <https://github.com/DexForce/UNITAS>
