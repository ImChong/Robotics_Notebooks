# XRZero-G0: Pushing the Frontier of Dexterous Robotic Manipulation with Interfaces, Quality and Ratios（arXiv:2604.13001）

> 来源归档（paper / 技术报告）

- **标题：** XRZero-G0: Pushing the Frontier of Dexterous Robotic Manipulation with Interfaces, Quality and Ratios
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2604.13001>（v1 2026-04-14，v2 2026-04-16，cs.RO，CC BY 4.0）
- **HTML：** <https://arxiv.org/html/2604.13001v2>
- **PDF：** <https://arxiv.org/pdf/2604.13001>
- **项目页：** <https://x2robot.com/x2go>（英文：<https://x2robot.com/en/x2go>）
- **代码：** <https://github.com/X-Square-Robot/XRZero-G0>（仅 README + 配图，无训练/采集代码）
- **数据集：** <https://huggingface.co/datasets/x-square-robot/XRZero-G0-3K>
- **作者：** James Wang, Primo Pu, Zephyr Fung, ..., Qian Wang, Roy Gan, Hao Wang（共 22 人，X Square Robot）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2604.13001v2` 全文，含摘要、§1 引言、§2 相关工作（Table 1）、§3 方法（Interfaces / Quality / Ratios / Policy Compatibility）、§4 G0-Dataset、§5 实验（RQ1–RQ4、Fig. 5–8）与 §6 结论及未来工作；未运行真机或复现数据集训练
- **通读：** 已完成

## 核心摘录（通读 arXiv:2604.13001v2 全文）

- **问题定义：** 无本体手持采集（UMI 范式）存在三大瓶颈：① 腕相机视觉 SLAM 在弱纹理/动态环境下漂移（Interfaces）；② 开环采集缺乏自动质检导致隐蔽运动学异常影响策略收敛（Quality）；③ 缺乏廉价无本体数据与高价真机数据的系统配比规律（Ratios）。
- **Interfaces（§3.1）：** PICO 4 VR 头显 inside-out 6-DoF 跟踪（Table 1 自报位置精度 $\le 4\text{ mm}$，对照 UMI $\sim 10\text{ mm}$、FastUMI $\sim 8\text{ mm}$）+ 头显可调主视角 RGB + 双腕相机（3 视角 30 Hz）；手柄刚性固定在 H 型（按压式粗抓取）和 G 型（手指驱动精细操作）夹爪上，双夹爪间距按目标双臂基线标定；背包边缘计算单元完成时空对齐与上传。注：Table 1 的 Modalities (V/T/A) 列标注为 $\checkmark/\checkmark/\checkmark$（视觉/触觉/听觉），而公开 HF 子集仅包含 3 视角图像与 14 维末端/夹爪状态动作。
- **Quality（§3.2）：** ① 视觉模糊帧剔除与静止帧降采样；② 按目标机器人 URDF 做运动学重定向与 IK 检查，剔除关节限位、奇异点和自碰撞片段；③ 每类任务抽样在目标双臂（CX001 / EX001）上严格开环物理回放验证；④ 切分子任务并标注关键帧与操作物体；报告有效率最高达 85%。
- **Ratios 与 G0-Dataset（§3.3–§4）：** G0-Dataset >2,000 h、3,000 个长尾任务，峰值采集速度 93.2 episode/h；预训练用大量无本体数据学视觉语义与 affordance，微调加入少量真机数据作 Few-Shot Physical Anchoring。
- **实验（§5）：**
  - RQ1 效率：相比主从遥操作，单条采集平均用时从 35→15 s（2.33×）、75→40 s（1.88×）、120→70 s（1.71×）。
  - RQ2 回放：CX001 与 EX001 上实现 1:1 空间回放（定性）。
  - RQ3 纯无本体扩量：抓葡萄/茄子/香蕉（300→500 条）呈线性增长，Wall-OSS 在茄子与香蕉达 75.0%；双臂插花扩至 2,000 条，Wall-OSS 在 $H=0.4\text{ m}$ 达 70%、未见高度 $H=0.45\text{ m}$ 达 60%。
  - RQ4 配比律（500 条纯真机基线）：1:1（500 真机 + 500 robot-free）把插花从 50% 提到 75%；10:1（500 robot-free + 50 真机）在叠毛巾（87.5%）与抓香蕉（75.0%）追平纯真机基线，综合采集成本约为真机遥操作的 1/20。
- **局限与未来工作（§6.2）：** 背包边缘计算单元偏重限制超长时采集；需探索真机锚定数据理论下界；当前集中于静态桌面双臂，未来向全身移动操作、强接触及柔性体拓展。

## 对 wiki 的映射

- [xrzero-g0](../../wiki/entities/xrzero-g0.md)
- 项目页与 HF 子集核查：[x2robot-xrzero-g0.md](../sites/x2robot-xrzero-g0.md)
