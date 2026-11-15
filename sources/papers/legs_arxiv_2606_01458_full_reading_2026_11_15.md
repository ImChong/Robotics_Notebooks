# LEGS：原文通读与实现复核（2026-11-15）

- **类型：** paper
- **原文：** <https://arxiv.org/html/2606.01458>
- **版本：** 2026-11-15 获取的 HTML；具体正文、附录和代码范围如下。
- **代码：** <https://github.com/hojjunekim/LEGS>
- **核对日期：** 2026-11-15
- **核对范围：** arXiv:2606.01458 HTML（2026-11-15 获取）的正文 §1–6、附录 A–J（重建、分阶段成功、颜色校准、随机化、mesh 基线、成本、生成协议、轨迹说明、相机匹配实验与训练超参数）；另打开作者项目页，读官方 README、run_collection.sh、TaskManager 与 MainRenderPipeline 全文。未执行模型/硬件。
- **通读：** 已完成

本次复核单独存档；[历史摘录](legs_arxiv_2606_01458.md)保留原有日期和发布状态，不能代替下述版本及协议核对。

## 原文与实现核对

- [现行原文](https://arxiv.org/html/2606.01458)报告 1,230 次主实验与额外 240 次 matched-camera 试验；原先归档的1,110次是历史口径，不混入现行汇总。
- 18-D 接口是双侧腕位姿/夹爪共14维，加底座 vx/vy/yaw/height 4维；MuJoCo物理500Hz、记录命令30Hz，两者不是同一频率。
- 主表九个 backbone×task 单元各10次，单一训练 seed。聚合 LEGS(200)60/90，teleop22/90，mesh30/90；小单元不能独立证明统计显著。匹配自动曝光的 Task3 为18/60（LEGS200）对6/60（mesh），说明前端组合效果，不独立识别3DGS与颜色校准各自贡献。
- 外观增广包含评测目标场景、物体与提示词，是目标适配，非未见物体零样本泛化；100/80/40属于LEGS-aug(200)，不是50条数据条件。
- [作者项目页](https://hojjunekim.github.io/legs/)已链到[官方LEGS仓](https://github.com/hojjunekim/LEGS)；代码Apache-2.0，资产/子模块各自许可。README明确真G1策略client不在本次发布内；公开离线和仿真评测不等于完整真机发布。
- 官方仓树 `3581449c`：[run_collection.sh](https://github.com/hojjunekim/LEGS/blob/main/run_collection.sh)依次生成场景、启动录制、快照CSV、离线渲染与LeRobot导出；MuJoCo/SONIC由两个外部终端预先启动。[TaskManager](https://github.com/hojjunekim/LEGS/blob/main/scripts/main/task/manager.py)按RUNNING/FAILED/SUCCESS推进；[渲染器](https://github.com/hojjunekim/LEGS/blob/main/scripts/main/renderer/main_render_pipeline.py)分开构建背景和前景层。

## 对 wiki 的映射

- [LEGS 详情](../../wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)
