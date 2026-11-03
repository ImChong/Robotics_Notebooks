# NablaEngine

> 来源归档（草稿）

- **代码：** https://github.com/OamIcs/NablaEngine
- **论文：** [∇Mimic，arXiv:2610.41654v1](../papers/nablamimic-2610-41654.md)
- **项目页：** [∇Mimic](../sites/nablamimic.md)
- **入库日期：** 2026-11-03

README 描述其为专用于 Unitree G1 的可微 GPU 物理引擎，支持 JAX 与 PyTorch；输入含状态与关节目标，接口提供仿真步及其解析梯度。仓库给出安装、验证与 quick start；其声称的性能结果需按具体 GPU、批量及对比配置解读。

README 限定支持范围：球体、胶囊与地面接触、关节限位、PD 位置执行器及指定 MuJoCo 求解和积分设置；其他模型不被普遍支持。许可为 Apache-2.0，第三方材料许可见仓库说明。

**待办：** 阅读 `jax_ffi`、`torch_ffi` 和 CUDA 实现，核对调用链后补源码运行时序图；尚未运行验证。
