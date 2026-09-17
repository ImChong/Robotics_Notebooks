# 部署（上真机）

从「仿真里能跑」到「真机上能跑」，中间隔着一整套工程问题：控制频率与推理频率怎么解耦、硬件接口与驱动器怎么接、出错了进什么安全状态、怎么留下可复盘的日志。这一模块收敛的是真机落地阶段最容易踩坑、也最难从论文里学到的那部分。

**站内入口：** [Sim2Real](../../../wiki/concepts/sim2real.md) · [Sim2Real 上机检查单](../../../wiki/queries/sim2real-checklist.md) · [控制频率与推理频率解耦](../../../wiki/concepts/control-inference-frequency-decoupling.md) · [机器人安全状态机](../../../wiki/concepts/robot-safety-state-machine.md)
