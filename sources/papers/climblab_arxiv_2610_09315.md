# ClimbLab 论文来源归档

## 元数据

- **题目：** ClimbLab: MATLAB Simulation Platform for Legged Climbing Robotics
- **方法：** ClimbLab（开源 MATLAB 腿式攀爬仿真器）
- **作者：** Kentaro Uno, Warley F. R. Ribeiro, Yusuke Koizumi, Keigo Haji, Koki Kurihara, William Jones, Kazuya Yoshida
- **单位：** 东北大学（Tohoku University）空间机器人实验室（以官方仓库 README 为准）
- **预印本：** https://arxiv.org/abs/2610.09315
- **HTML：** https://arxiv.org/html/2610.09315v1
- **版本：** arXiv 页面显示 2026-10-07 提交；comments 写明是 CLAWAR 2021 会议论文集的作者版本（正式版 doi:10.1007/978-3-030-86294-7_20）。即论文本身为 2021 年工作，周报按 arXiv 上线日期归入本周。
- **代码：** https://github.com/Space-Robotics-Laboratory/ClimbLab（BSD 3-Clause；2026-10-12 克隆核查，最后提交 2021-08-27）
- **项目页：** 无独立项目页

## 论文要解决的问题

腿式攀爬研究缺少便于教学、动力学分析与控制/步态原型验证的轻量开源仿真工具。ClimbLab 允许把任意带浮动基座的多肢机器人建成关节式多体，在任意环境（坡度、重力、地面刚度可调，任意点云作地形）中仿真行走与攀爬。

## 方法与训练信息

- 无学习成分，是物理仿真/步态规划平台。
- 主循环八阶段：初始化 → 环境感知 → 运动规划 → 平衡评估 → 控制 → 物理引擎 → 可视化 → 结束保存。
- 动力学：浮动基座多刚体，H q̈ + c + g = u + JᵀF_r，Runge-Kutta 数值积分，由同作者的 SpaceDyn 工具箱求解动力学与正运动学。
- 接触：三向弹簧–阻尼地面反力模型 F_r = −K_s(x − x_c) − D_s ẋ；因有夹持器，不强制单侧接触约束；夹持脱离由平衡判据或超过最大夹持力触发。
- 感知：模拟相机点云（含视场模型），按几何和夹持器设计检测可抓取点。
- 步态：周期爬行步态（crawl）与非周期对角步态；足端/基座样条轨迹 + 解析逆运动学。
- 控制：PD 关节力矩跟踪期望关节角，仅此一种控制律。
- 稳定性指标：支撑多边形/质心、Tumble Stability Margin（TSM）、Gravito-Inertial Acceleration（GIA）裕度；论文推荐 TSM 与 GIA（含最大夹持力）。
- 地形：平面、分形粗糙地形、由攀岩岩点测试场传感数据提取的粗糙地形。

## 论文报告的结果

1. **悬崖抓持（HubRobo）：** 质量 2.4 kg 的四足；夹持力 13.0 N；坡度每 10° 扫描；在 100° 悬垂不平坡面可保持，110° 失败。
2. **陡坡攀爬（ANYmal B）：** 夹持附着力 200 N、坡度 45°，步态周期 T=8 s、摆动 2 s；正常步态（占空比 0.75）对比对角步态：

| 指标 | 正常 | 对角 |
|---|---|---|
| 平均 TSM（m） | 0.361 | 0.359 |
| 最小 TSM（m） | 0.000 | 0.104 |
| 平均 GIA 裕度（m/s²） | 11.8 | 11.9 |
| 最小 GIA 裕度（m/s²） | 1.10 | 5.39 |
| 平均最大关节力矩（N·m） | 28.7 | 26.7 |
| 平均 RMS 关节力矩（N·m） | 13.3 | 11.5 |

   对角步态 RMS 力矩低约 13.5%，最小稳定裕度明显更高，均值几乎相同。
3. **感知与落脚点选择（ALPHRED）：** 半人高机器人，RealSense D435 下倾 60°，平面上随机离散可落点；对角基座步态 T=4 s、摆动 1 s、摆高 0.05 m；仅运动学（略去动力学），用于验证局部落脚点选择是否朝目标推进。

## 复现与解释边界

- 官方仓库 README 要求 MATLAB 2018b 及以上 + SpaceDyn v2r0（随仓库 `lib/spacedyn_v2r0` 提供）；论文本身未写 MATLAB 版本。
- 仓库 README 称运行 `src/climb_main.m`，但 2026-10-12 克隆的仓库里入口实为 `src/main_sim.m`（`config` 变量选预设）；以仓库实际文件为准。
- 仅实现昆虫型与哺乳型四足；仅 PD 控制；仅两类步态；接触为基础弹簧阻尼；示例 3 无动力学；全身运动规划（倾翻稳定、能耗、可操作度）未实现；无真机验证对比。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-climblab-matlab-legged-climbing-sim.md
- 仓库归档：sources/repos/climblab.md
- 任务入口：wiki/tasks/locomotion.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.09315>
- arXiv HTML 全文：<https://arxiv.org/html/2610.09315v1>
- 官方仓库：<https://github.com/Space-Robotics-Laboratory/ClimbLab>
