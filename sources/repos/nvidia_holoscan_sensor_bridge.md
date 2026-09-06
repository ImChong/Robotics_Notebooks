# nvidia-holoscan/holoscan-sensor-bridge

> 来源归档

- **标题：** Holoscan Sensor Bridge
- **类型：** repo
- **组织：** nvidia-holoscan
- **代码：** <https://github.com/nvidia-holoscan/holoscan-sensor-bridge>
- **文档：** <https://docs.nvidia.com/holoscan/sensor-bridge/>
- **HoloHub：** <https://nvidia-holoscan.github.io/holohub/modules/holoscan-sensor-bridge/>
- **Stars：** ~53（2026-09-06）
- **入库日期：** 2026-09-06
- **一句话说明：** **Bring Your Own Sensor (BYOS) over Ethernet** — FPGA 低延迟传感器流 + Holoscan operator 管线；含 IMX274/IMX477 示例与 **hololink** 设备驱动框架。
- **沉淀到 wiki：** [`wiki/entities/holoscan-sensor-bridge.md`](../../wiki/entities/holoscan-sensor-bridge.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（Apache-2.0） |
| **代码** | <https://github.com/nvidia-holoscan/holoscan-sensor-bridge> |
| **FPGA IP** | 文档/HoloHub 另链 **FPGA IP Source**（硬件 IP 非本仓单一交付物） |
| **依赖** | **NVIDIA Holoscan SDK**（HoloHub 标注 min SDK **4.2**；测试 **4.2.0**） |

## README 要点（2026-09-06）

- **定位：** FPGA 采集外设数据 → **UDP** → 主机 **ConnectX GPUDirect** 直写 GPU → 接入 **Holoscan pipeline**。
- **示例硬件：** [Lattice CertusPro-NX Sensor Bridge](https://www.latticesemi.com/products/developmentboardsandkits/certuspro-nx-sensor-to-ethernet-bridge-board) + **IMX274**；[Microchip Ethernet Sensor Bridge](https://www.microchip.com/en-us/products/fpgas-and-plds/boards-and-kits/ethernet-sensor-bridge) + **IMX477**。
- **语言：** C++ / Python / Verilog 等；operator 含 `RoceReceiverOp`、`LinuxReceiverOp`、`CsiToBayerOp`、推理与 CRC 校验等（HoloHub 列表）。
- **扩展：** `hololink_module` 设备驱动教程；`new_sensors` 文档教 Python/C++ **sensor object** 与 I2C 寄存器访问。

## 对 wiki 的映射

- 实体：[`wiki/entities/holoscan-sensor-bridge.md`](../../wiki/entities/holoscan-sensor-bridge.md)
- 产品页：[`sources/sites/nvidia-holoscan-sensor-bridge.md`](../sites/nvidia-holoscan-sensor-bridge.md)
- 文档 intro：[`sources/sites/holoscan-sensor-bridge-docs-intro.md`](../sites/holoscan-sensor-bridge-docs-intro.md)
