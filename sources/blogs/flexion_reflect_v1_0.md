# Flexion Reflect v1.0 — The Path Towards Long-Horizon Autonomous Humanoid Work

> 来源归档（blog / Flexion 官方）

- **标题：** Flexion Reflect v1.0 - The Path Towards Long-Horizon Autonomous Humanoid Work
- **类型：** blog
- **作者：** Flexion Team
- **原始链接：** https://flexion.ai/news/flexion-reflect-v1.0
- **发表日期：** 2026-06-29
- **入库日期：** 2026-06-30
- **抓取方式：** 官方新闻页直接抓取（WebFetch）
- **一句话说明：** Flexion 发布 **Reflect v1.0** 机器人智能平台：以自研 **Reflect-VLM** 作任务级 mission control，下层 **VLA + RL 运动技能** 与 **Reflex 全身控制器** 闭环，配套 **FlexComm** 低延迟运行时与 **3DGS 全栈仿真**；演示单条自然语言指令下跨楼层取件、乘电梯、开箱上架的 **全程无人工介入** 长程自主任务，并在 16 步 mission 评测上报告 **SFT+RL 90%** 端到端完成率（SFT 仅 38%）。

## 核心摘录（归纳，非全文）

### 问题与里程碑

- **长程自主的复合失败：** 导航 95%、抓取 90%、规划偶发误判在时间上**相乘**而非相加，难以组成可靠系统。
- **演示任务（单条 NL 指令）：** 取楼下快递（可走楼梯）→ 乘电梯上楼 → 开箱 → 将零食放入货架空抽屉；**全程自主、无操作员**。
- **v1.0 相对 v0（2025-11）：** 架构分层不变，但语义感知更丰富、mission controller 更反应式、运动层更强、控制器更紧、运行时更快；**最大变化：RL 不再局限于单技能，而是贯穿低层控制到高层决策各层**。

### 四层架构（Reflect）

| 层级 | 模块 | 职责 |
|------|------|------|
| **Mission** | **Reflect-VLM** | 第一人称视觉 + 结构化 tool 调用；语义地图查询/全局路径；NL 任务与中途改指令 |
| **Motion** | **VLA + RL skills** | 导航、场景交互（门/电梯）、操作；仿真训练 + 域随机化 + 在线感知反馈 |
| **Control** | **Reflex** | 实时全身控制：平衡、力感知交互、楼梯/扰动鲁棒；可跨本体迁移 |
| **Runtime** | **FlexComm + 监控 + 3DGS Sim** | 微秒级同机通信、全链路可观测、部署前全管道仿真回放 |

### Reflect-VLM（任务级推理）

- **语义地图 tool：** 建筑扫描 → 语言接地全局地图；可 NL 查询区域/物体、请求全局路径；用户可标注地标。
- **任务配置：** 改 prompt 即可换任务，无需改代码；支持**执行中改指令**（mid-mission replan）。
- **现成 VLM 不足：** 常过早发出「逻辑上合理」的下一步 tool call，而不视觉验证前置条件是否满足。
- **RL 对长程可靠性关键：** 16 步 mission 评测（首次犯错前进度）— **base VLM 几乎立即失败**；**SFT 38%** 端到端完成；**SFT+RL 90%**。

### 运动层（像素 → 物理交互）

- **训练：** 多数技能在仿真中训练，自定义视觉编码器 + 针对性域随机化；需在线感知反馈适应姿态/物体变化。
- **接触丰富技能：** 开门、搬箱（100 g–3.5 kg 同策略）、箱体重定位（全身协调）、电梯按钮（符号+连续控制边界）、灵巧工具开箱（**遥操作数据训 VLA + WBC 闭环**；作者称自由移动人形上可靠性仍难，下一代用 RL）。
- **局部导航：** 全局路径 + 连续局部适应；动态障碍避让、崎岖地形。

### Reflex 全身控制

- 实时同时满足平衡、执行器限制、安全约束；上肢操作时仍稳定；**100+ 次连续上下楼梯**；外力扰动下保持跟踪。
- 模块名 **Reflex**，强调高级 locomotion 与力感知全身控制，**最小人工 effort 跨形态部署**。

### 鲁棒性与重试

- **运动层：** RL 学到的局部恢复（O.O.D. 抓取失败重试、箱子被推开时调整）。
- **Agent 层：** 相机 feed 检测 off-nominal → 重规划。

### 软件栈（FlexComm 等）

- **FlexComm：** 同主机通信延迟 **数十–数百微秒**；相对 ROS DDS **同机加速最高 ~40%**、**CPU ~30% 节省**；抗 WiFi 漫游/断连；多机/云扩展且降低串扰。
- **可观测性：** 从 reasoning trace 到内核/网卡日志的自动采集与实时展示。
- **3D Gaussian Splatting 仿真：** 观测足够逼真供 VLM/感知、几何足够供高低层策略、速度足够闭环、可逐决策调试；真机部署前跑**全管道**。

### 局限（作者自述）

- 有界任务分布；部分物体仍难抓；mission controller 视觉假设仍可能错；恢复行为未覆盖全部失败模式。
- 下一步：更广技能分布、更强失败检测/恢复、更强 sim 评测、更紧 runtime 预算、端到端 verifiable reward 的 mission 推理。

### 与 v0 前序

- [Flexion Reflect v0](https://flexion.ai/news/flexion-reflect-v0)（2025-11-20）首次公开架构；v1.0 为同一设计的工程与能力跃迁。

## 对 wiki 的映射

- [flexion-reflect-v1](../../wiki/entities/flexion-reflect-v1.md)（系统/平台实体 + Mermaid 长程自主流水线）
- 交叉：[VLA](../../wiki/methods/vla.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Vision-Language Navigation](../../wiki/tasks/vision-language-navigation.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[KinetIQ Ascend](../../wiki/entities/kinetiq-ascend.md)、[Curr-0](../../wiki/entities/current-robotics-curr0.md)

## 可信度与使用边界

- **公司官方博客**，非 peer-reviewed；定量除 mission 完成率外多为演示级自报。
- **未公开** Reflect-VLM / VLA / Reflex 权重、训练数据规模与完整开源栈。
- **FlexComm** 与 ROS2 对比为作者自测，需独立基准验证。
- 视频展示为特定建筑与任务分布，**不宜外推为通用家庭/工业部署**。

## Citation

```bibtex
@article{
    flexion2026reflectv1,
    author = {Flexion Team},
    title = {Flexion Reflect v1.0 - The Path Towards Long-Horizon Autonomous Humanoid Work},
    journal = {Flexion News},
    year = {2026},
    url = {https://flexion.ai/news/flexion-reflect-v1.0},
}
```
