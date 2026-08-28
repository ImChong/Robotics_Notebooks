# Pollen Robotics Microduck 项目页

> 来源归档

- **标题：** Microduck — A tiny biped robot you can teach new tricks
- **类型：** site（产品 / 项目门户）
- **机构：** Pollen Robotics
- **链接：** https://pollen-robotics.com/microduck
- **代码（Runtime / SDK）：** https://github.com/pollen-robotics/microduck
- **代码（RL 训练）：** https://github.com/pollen-robotics/microduck_rl
- **社区：** https://discord.com/invite/pollen-community-519098054377340948
- **入库日期：** 2026-08-28
- **一句话说明：** Pollen 官方 Microduck 产品页：约 25 cm / 800 g 开源软件栈桌面双足，宣传「仿真训练 → 真机部署 → 再训 → 发布策略」闭环；整机预售，软件与 RL 栈已开源。
- **开源状态：** **已开源**（软件 Apache-2.0；RL 仓另声明硬件设计文件 CC BY-SA-NC）。整机本身是商品（介绍价 $399，宣称 2026 圣诞前发货），不是 Open Duck Mini 式 DIY BOM 套件。
- **仓库归档：** [microduck.md](../repos/microduck.md)、[microduck_rl.md](../repos/microduck_rl.md)
- **沉淀到 wiki：** [pollen-microduck](../../wiki/entities/pollen-microduck.md)、[pollen-microduck-rl](../../wiki/entities/pollen-microduck-rl.md)

---

## 步骤 2.5 核查（2026-08-28）

1. **项目页 CTA / Open source 区** 明确指向 GitHub：SDK、仿真与完整 RL 训练栈公开；口号 *What the robot runs is what you can read, fork and retrain*。
2. **开放程度：已开源**
   - Runtime / 机载软件：[`pollen-robotics/microduck`](https://github.com/pollen-robotics/microduck)（Apache-2.0，默认分支 `main`，Rust）
   - 训练：[`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl)（Apache-2.0，默认分支 `develop`，mjlab + PPO）
   - 社区：Pollen Discord 邀请链（项目页 *Join the flock*）
3. **未在项目页作为 DIY 入口公开的部分：** 整机按商品预售（四色外壳、配件包）；Onshape 导出配置在 RL 仓 MJCF 侧，不把本页读成「买零件自己焊一台」。
4. **交叉：** 本页 ↔ 两仓归档 ↔ wiki 实体页。

## 产品页摘录（策展）

| 项 | 内容 |
|----|------|
| 定位 | *A 25 cm open-source biped you train yourself with reinforcement learning. Playable out of the box.* |
| 规格 | 15 电机；25 cm；800 g；相机 + LiDAR + 两路 IMU；机载策略环 **50 Hz**；盒内宣称 7 个已训动作 |
| 闭环口号 | Train in sim → Deploy on the robot → Refine the simulation → Publish the policy（可用本机或 Hugging Face Jobs） |
| 盒内行为（产品页） | Walk（速度跟踪）、Sit & stand、Kick、Grab（喙触地舀起）、Roller skating、Get back up |
| 套装 | 整机 $399（机器人 + 电池 + USB-C + 手柄）；Charger pack $39；Dev pack $119；Accessory pack $39 |
| 发货 | 预售开放；四色外壳；宣称 2026 圣诞前发货（税运另计） |
| 机载 CLI 示例 | `ssh microduck` 后 `robotctl monitor / configure / update` |

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 整机、Runtime、产品边界 | `wiki/entities/pollen-microduck.md` |
| mjlab 训练、BAM、背隙、奖励课 | `wiki/entities/pollen-microduck-rl.md` |
| 同机构移动人形 | `wiki/entities/pollen-reachy2.md` |
| 对照：社区 DIY 迷你鸭 | `wiki/entities/open-duck-mini.md` |
