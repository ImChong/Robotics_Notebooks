# MotorFit：执行器辨识与仿真验证工具（一手源码）

- **类型：** repo / 电机执行器模型辨识
- **上游组织：** motorfit
- **官方仓库：** https://github.com/motorfit/motorfit
- **官方文档：** https://motorfit.github.io/
- **上游 BAM：** https://github.com/Rhoban/bam
- **许可证：** Apache-2.0
- **核查版本：** main；pyproject.toml 版本 0.1.0；2026-11-17
- **核对范围：** 通读 README.md、README.zh-CN.md、docs/development/architecture.md、docs/getting-started/{quickstart,workflow}.md、docs/guide/{acquisition,identification,validation}.md、docs/reference/{parameters,backends}.md、docs/case-studies/rs05.md；核查 pyproject.toml、.gitmodules 与 src/motorfit/cli.py、acquisition/recorder.py、offline.py、identification/bam/runner.py、identification/validate_simulation.py 的运行入口；未执行硬件测试或逐行审计全部源码。
- **通读：** 已完成
- **开放程度：** 公开 Python 软件、RobStride/达妙驱动、采集/辨识/仿真回放代码和三条离线记录；完整 RS05 111 条数据、CAD、视频在独立归档，不随 git clone 提供。
- **实体页：** [MotorFit](../../wiki/entities/motorfit.md)
- **官方文档归档：** [MotorFit Docs](../sites/motorfit-official-docs.md)

## 经官方资料核查的技术点

1. 基于 Rhoban BAM 的 M1–M6 摩擦模型辨识；M1 估计 torque_ratio、armature、q_offset、command_delay、friction_base、friction_viscous。
2. Linux SocketCAN 采集 RobStride RS00–RS06 与 Damiao 16 个 MIT 型号配置（包括 DM4310/DM4340/DM8009）；各型的编码限制不同。
3. 采集保存 raw.json、samples.jsonl、frames.jsonl；process 重采样后 fit 优化，--eval 使用固定模型对完整留出实验评估。
4. 跨物理引擎 replay 当前仅 M1 + torque-MIT + 单摆夹具；支持 MuJoCo、Newton（SolverMuJoCo）、mjlab/MJWarp 的相应回放运行时。
5. 官方案例使用 RS05 摆锤 111 条录制（三档负载，每档 37）；36 条最终测试冻结 BAM M1 的平均位置 MAE 约 2.035 mrad；200 g 释放/停止有未解释残差，不能宣传成所有电机的普适精度。

## 官方源码与文档入口

- [CLI 命令](https://github.com/motorfit/motorfit/blob/main/src/motorfit/cli.py)
- [CAN 采集 recorder](https://github.com/motorfit/motorfit/blob/main/src/motorfit/acquisition/recorder.py)
- [BAM process / fit 桥接](https://github.com/motorfit/motorfit/blob/main/src/motorfit/identification/bam/runner.py)
- [离线 demo 实现](https://github.com/motorfit/motorfit/blob/main/src/motorfit/offline.py)
- [跨引擎 replay](https://github.com/motorfit/motorfit/blob/main/src/motorfit/identification/validate_simulation.py)
- [RS05 实验原文](https://github.com/motorfit/motorfit/blob/main/docs/case-studies/rs05.md)

## 对 wiki 的映射

- [MotorFit](../../wiki/entities/motorfit.md)
- [关节执行器参数辨识](../../wiki/methods/joint-actuator-parameter-identification.md)
- [关节辨识实验设计](../../wiki/methods/sim2real-joint-sysid-experiment-design.md)
