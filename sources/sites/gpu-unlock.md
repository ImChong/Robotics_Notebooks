# GPU Unlock 官方项目入口

> 来源归档

- **标题：** GPU Unlock（Ampere 上游与 Ada 移植）
- **类型：** site（官方 GitHub 项目入口）
- **链接：** <https://github.com/Zagitta/gpu-unlock>
- **Ada 移植与 RTX 4090 资料：** <https://github.com/JustAResearcher/gpu-unlock-ada>
- **入库日期：** 2026-10-25
- **一句话说明：** Linux 内核模块在 NVIDIA 驱动绑定前调整特定 GPU 的 SM speed-select 配置；图中 RTX 30 系列与 RTX 4090 的结果来自两个不同仓库。
- **代码：** [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock)（**已开源**）
- **Ada 移植代码：** [JustAResearcher/gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada)（**已开源**）
- **沉淀到 wiki：** 是 → [GPU Unlock](../../wiki/entities/gpu-unlock.md)

## 开源边界与主张核对

| 资料 | 项目页核查结果 |
|------|----------------|
| Zagitta 上游 | 官方 GitHub README 覆盖指定 Ampere / RTX 30 系列设备；公布的收益是特定卡与 GEMM 设置下的基准结果，兼容性表同时保留失败或未验证项。 |
| Ada 移植 | 独立 GitHub 仓库报告 RTX 4090 的约 2 倍 FP16/BF16→FP32 累加指令吞吐；作者明确说明这不是整应用、GEMM 或推理速度翻倍。 |

截图把 RTX 3090 与 RTX 4090 的数字合并叙述。入库时按各自项目 README 和测试报告分开归档；不将“固件漏洞”“算力暴涨 80%”视作适用于所有卡和工作负载的已证实结论。

## 资料互链

- [Ampere 上游代码仓库归档](../repos/zagitta-gpu-unlock.md)
- [Ada / RTX 4090 移植仓库归档](../repos/justaresearcher-gpu-unlock-ada.md)
- [GPU Unlock wiki 实体页](../../wiki/entities/gpu-unlock.md)
