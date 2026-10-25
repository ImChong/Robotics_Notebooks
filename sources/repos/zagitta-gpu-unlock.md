# Zagitta/gpu-unlock

> 来源归档

- **标题：** GPU unlock
- **类型：** repo（Linux GPU 内核模块）
- **链接：** <https://github.com/Zagitta/gpu-unlock>
- **许可：** GPL-2.0-only（以仓库 LICENSE 为准）
- **入库日期：** 2026-10-25
- **一句话说明：** 在 NVIDIA 驱动绑定前设置 SM speed-select 寄存器，针对部分 RTX 30 系列 Ampere 卡恢复 FP16/BF16/TF32 输入、FP32 累加模式的指令发射速率。
- **代码：** <https://github.com/Zagitta/gpu-unlock>（**已开源**）
- **沉淀到 wiki：** 是 → [GPU Unlock](../../wiki/entities/gpu-unlock.md)

## 开源状态与范围

仓库提供 Linux 内核模块、安装脚本和基准代码。README 将支持范围描述为 GA102、GA103、GA104、GA106 的指定 GeForce 设备 ID；表中实测主要来自 RTX 3090 Ti、3090、3080、3070 和 3060。个别型号虽可完成解锁流程，但性能仍可能保持半速，不能只按芯片或型号推断结果。

## 关键资料

- README 记录模块在 NVIDIA 驱动加载前运行、写入 speed selectors 并执行 PCIe Function Level Reset；项目不是修改后的 NVIDIA 驱动。
- README 中 RTX 3090 Ti 的 8192³ GEMM 数据为：FP16 87.46→118.39 TFLOPS、BF16 87.50→157.41 TFLOPS、TF32 42.58→82.61 TFLOPS。属于项目列出的具体功耗与基准条件。
- 用户报告显示提升取决于显卡功耗上限和具体矩阵任务；矩阵密集型任务可能明显受益，受显存带宽限制的 LLM decode 和 INT8 通常变化不大。
- 兼容性结果来自用户反馈，存在未验证卡型及失败案例。安装需要修改启动参数并加载内核模块，应先审阅源码并准备恢复启动配置。

## 对 wiki 的映射

- [GPU Unlock 实体页](../../wiki/entities/gpu-unlock.md)
- [NVIDIA CUDA](../../wiki/entities/nvidia-cuda.md)
