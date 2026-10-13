# yhx1203/parkour_mjlab

> 来源归档

- **标题：** Parkour MJLab
- **类型：** repo / humanoid and quadruped reinforcement learning / parkour
- **所有者：** yhx1203
- **代码：** <https://github.com/yhx1203/parkour_mjlab>
- **项目页：** 仓库未列独立项目页
- **主要平台：** Unitree G1、Unitree Go2
- **许可证：** GitHub 仓库元数据未声明许可证；根目录未发现 LICENSE 文件（核查日期：2026-10-13）
- **发布：** 当前 GitHub Releases 页面为空
- **核查日期：** 2026-10-13
- **一句话说明：** 基于 MJLab 的强化学习代码库，包含 G1 感知跑酷 PHP teacher/student 训练与 MuJoCo Sim2Sim，以及 Go2 PIE、AMP + Parkour、G1 Stair 等任务。

## README 核查摘录

- 环境示例为 Python 3.11；仓库通过 `pip install -e .` 安装，并列出 YAML、Boost、Eigen、spdlog、fmt 等系统依赖。
- G1 的 PHP 部分先分别训练 locomotion、low-step、high-step、low-climb、high-climb 五类 teacher，再以五个 teacher checkpoint 训练 student；README 提供训练与回放命令。
- README 的回放示例引用本地路径 `logs/rsl_rl/g1_php_student/model_23000.pt`；这只是命令示例，不证明该 checkpoint 已随仓库或 release 发布。
- PHP 的 MuJoCo Sim2Sim 路径可将 student 导出 ONNX，随后通过 `deploy.php.sim2sim` 的入口运行自定义三障碍场景，并以终端命令控制。
- README 要求从 PHP 官方仓库的 [training-motion-examples-v1 release](https://github.com/amazon-far/php_parkour/releases/tag/training-motion-examples-v1) 下载五组动作数据。这里是对 PHP 的 MJLab 社区实现，与 [Amazon FAR 官方实现](https://github.com/amazon-far/php_parkour) 不同。
- 同仓库也列出 Go2 PIE 训练/回放及原生 MuJoCo Sim2Sim、AMP + Parkour Sim2Sim、G1 Stair 的 SDK2 Sim2Sim；因此它是覆盖多个任务的代码库，不只是 PHP 项目镜像。

## 开放边界与待核实项

- **代码：** GitHub 上公开可读；但仓库没有声明许可证，不能据此假设可按开源许可修改、再分发或商用。
- **权重：** 用户提供的截图中，仓库作者评论称训练模型已上传；截图未包含权重地址。README 有本地 checkpoint 路径示例，当前 GitHub Releases 页面为空。本次无法确认权重的实际下载位置及对应版本。
- **复现范围：** README 展示的是示例命令与任务入口，不代表这些命令已在此处运行，也不构成论文报告指标的复现验证。

## 对 wiki 的映射

- [Parkour MJLab 独立代码库节点](../../wiki/entities/parkour-mjlab.md)
- [PHP 论文与官方实现实体](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)
- [PHP 官方仓库归档](./php_parkour.md)

## 参考入口

- [Parkour MJLab README](https://github.com/yhx1203/parkour_mjlab/blob/main/README.md)
- [PHP 官方仓库](https://github.com/amazon-far/php_parkour)
- [MJLab](https://github.com/mujocolab/mjlab)
