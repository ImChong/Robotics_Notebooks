# JustAResearcher/gpu-unlock-ada

> 来源归档

- **标题：** GPU unlock Ada port
- **类型：** repo（Linux GPU 内核模块与诊断基准）
- **链接：** <https://github.com/JustAResearcher/gpu-unlock-ada>
- **许可：** GPL-2.0-or-later（仓库 LICENSE 文件）
- **入库日期：** 2026-10-25
- **一句话说明：** 将 GPU Unlock 的预驱动 speed-select 流程移植到 RTX 4090，并提供实验性 RTX 4070 Ti SUPER 支持及寄存器/指令吞吐验证材料。
- **代码：** <https://github.com/JustAResearcher/gpu-unlock-ada>（**已开源**）
- **上游项目：** <https://github.com/Zagitta/gpu-unlock>
- **沉淀到 wiki：** 是 → [GPU Unlock](../../wiki/entities/gpu-unlock.md)

## 开源状态与实测边界

仓库发布模块、固件生成和诊断/基准源码。README 对 RTX 4090（AD102）报告 FP16/BF16→FP32 累加的寄存器驻留指令吞吐约 2 倍；报告注明最初实验仅一张实体卡、成功冷启动一次，之后有一次独立用户复现。它明确区分指令吞吐与实际应用加速：同卡 cuBLAS GEMM 的 FP16 增幅依矩阵尺寸约 12–34%，BF16 在所测尺寸约 16–20% 或接近不变。

RTX 4070 Ti SUPER 路径为单卡实验验证；RTX 4090 D 和其他新目标带有实验开关或未验证边界。README 也说明 RTX 4060 Ti 16GB 仅有寄存器读取验证，尚无同卡前后吞吐对照。不可将这些结果推广为“4090 推理速度翻倍”。

## 对 wiki 的映射

- [GPU Unlock 实体页](../../wiki/entities/gpu-unlock.md)
