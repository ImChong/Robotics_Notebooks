# ASENA（arXiv:2609.39207）

> 来源归档（paper；2026-10-16；核对 v2）

- **标题：** ASENA: Self-evolving Agents for Embodied Navigation
- **论文：** <https://arxiv.org/abs/2609.39207>
- **HTML（v2）：** <https://arxiv.org/html/2609.39207v2>
- **版本：** v1 于 2026-09-30 提交；v2 于 2026-10-12 更新
- **作者：** An-Chieh Cheng、Isabella Liu、Edmund Bu、Johan Bjorck、Hongxu Yin、Zhengyi Luo、Jan Kautz、Linxi “Jim” Fan、Yuke Zhu、Sifei Liu
- **机构：** NVIDIA、University of California San Diego、The University of Texas at Austin
- **官方项目页：** <https://research.nvidia.com/labs/gear/asena/>
- **许可证：** arXiv HTML 标注 CC BY 4.0
- **对 wiki 的映射：** [ASENA](../../wiki/entities/paper-asena-self-evolving-agents.md)

## 核心摘要

ASENA 把通用 coding agent 接入人形机器人的感知、计算、受监督执行和持久化经验。Agent 可读取传感器、编写与执行程序、检查记录、修复失败，并复用笔记和可执行技能；这些改进发生在工作区与程序层，模型权重保持固定。

ASENA-VLN 是可选的 4B 单目视觉语言导航策略，以路线指令、视觉问答和几何生成的原子导航任务训练，输出机身坐标系轨迹。在 R2R / RxR 验证集上的成功率分别为 68.7% / 70.2%。

## 方法与系统边界

1. 统一 robot harness 暴露观测、几何计算、导航、抓取、动作规划和执行记录接口；Agent 通过组合程序完成任务。
2. 导航可调用 LiDAR 占据图几何规划，或把 ASENA-VLN 当作可选 waypoint 工具。
3. 抓取候选来自 SAM 2 分割与深度，也可选用 GraspGen-X；随后通过 IK 和运动规划生成手臂动作。
4. 手臂轨迹在执行前按实测关节状态重建，并检查关节限制、自碰撞、场景净空和持物空间。
5. 生成的全身姿态先在 MuJoCo 中检查姿态连续性、运动限制和自碰撞；真机运动需校验并由操作者批准。独立导航安全机制可拒绝里程计过期或电量偏低时的命令，并有心跳停止与急停。
6. SONIC 执行全身运动；可选 YAM 控制器负责夹爪，语音走机载扬声器。

## 持久经验与策略学习

运行记录以 MCAP 保存传感器与机器人状态、动作、反馈、程序和可视化。Agent 回放记录，定位意图与执行差异并修复程序，再将有效笔记或技能保留到工作区。仿真演进评测使用 16 个并行 worker 的私有工作区，轮次间由 improver 汇总经验；模型与控制器权重固定。

ASENA-VLN 基于 Qwen3-VL-4B-Instruct 扩展轨迹与像素目标 token。输入最多 8 帧前向 RGB 历史和结构化指令，可带目标位姿、锚点或运动限制；输出最多 8 个机身坐标 waypoint（x、y、朝向）。像素目标用于辅助监督，论文评测实际使用轨迹 token。

作者用 MP3D 与 HM3D 几何构造并验证 9 类原子导航任务，涵盖定距移动、转向、定向接近、目标对齐、进房间、穿门、走廊跟随和楼层切换；每条指令配有 waypoint 轨迹与像素目标监督。训练混合 R2R、RxR、FGR2R、ScaleVLN、SRDF、ObjectNav 与多视角 VQA。

## 关键结果

- **独立 VLN 策略：** R2R val-unseen 成功率 68.7%、SPL 64.2%；RxR 成功率 70.2%、SPL 59.7%。
- **Agent + 策略：** 加入 ASENA-VLN 后，Claude Sonnet 5 在 R2R、RxR 的成功率各提升 11 个百分点；GPT-6 Astra 本身基线较强，RxR 成功率加入策略后略降，说明收益依赖 agent 后端。
- **交互效率：** VLN 工具使 Claude Sonnet 5 的平均工具调用分别下降 59.2% / 62.5%；GPT-6 Astra 下降 46.9% / 51.0%（R2R / RxR）。
- **重复任务演进：** 相同 100 项任务集经过 10 轮工作区演进，R2R 成功率从 72% 到 98%，RxR 从 65% 到 89%；模型权重固定，评测使用模拟器反馈。
- **具身问答：** HM-EQA 冻结工作区第 17 轮准确率 81.2%；MT-HM3D 的最佳冻结结果出现在第 4 轮，为 66.5%。两者是不同基准与轮次。
- **Jetson AGX Thor 推理：** 补充材料在 G1 的 Thor 上测量较早 checkpoint；FP8 + 缩减输出词表报告 877 ms 端到端延迟、3.20 倍于 PyTorch BF16。该结果不是主评测 checkpoint，也不代表整套 coding-agent 任务延迟。

## 真机演示与局限

Unitree G1 上展示四个无预建地图任务：搜索卫生间位置并返回指路、在 pantry 查找咖啡设备、比较零食库存并用全身手势回答、拾取罐装物品并携带返回后交接。论文将这些作为定性能力组合演示；真实任务耗时约 10–20 分钟，并指出仍需在多物体、多布局与多用户场景重复测量可靠性，以及加强动态环境安全验证。

## 开源状态

截至 2026-10-16，核查到的官方 NVIDIA 项目页提供方法、演示和结果，但没有列出代码、权重或数据集下载入口；未找到可核验的官方训练或部署实现，不能将实验与推理测量误作可复现代码发布。

## 参考来源

- [arXiv v2 HTML](https://arxiv.org/html/2609.39207v2)
- [NVIDIA GEAR ASENA 项目页](https://research.nvidia.com/labs/gear/asena/)