# CausalWM 官方推理实现

- **URL：** <https://github.com/AetherLabsAI/CausalWM>
- **固定版本：** [a6523ad66f509170642ab4fcb60e564e124f9c5c](https://github.com/AetherLabsAI/CausalWM/tree/a6523ad66f509170642ab4fcb60e564e124f9c5c)
- **核对范围：** 根README全文；inference.py的参数校验、组件/文本/观测加载、generate/main；sampler.py的schedule校验及run_full_scene_cot完整函数。2026-11-13，仅静态核读，未运行GPU。
- **通读：** 已完成

CLI验证三份local资产与输出目录→分阶段搬text encoder/VAE/Transformer到GPU→LTX2Scheduler生成三组sigma→run_full_scene_cot逐stage更新active tokens、固定完成流与观测→VAE解码→恢复RGB0/flow0、导出视频和provenance。CFG默认1，当前CLI至少2步/stage。121帧640×480/16fps是生成内容时轴，不是推理吞吐。

公开范围为PAI-Bench-G专用TI2V代码与checkpoint链接，训练/RL、动作多视角模型未发布；依赖LTX-2 Community License、Gemma条款。源码时序与论文设计分开核对；未下载/复现结果。
