# TI 电池监测与 NVIDIA Jetson Thor 官方资料复核

- **类型：** 官方技术资料 / 原文复核
- **核对日期：** 2026-11-16
- **核对范围：** TI BQ76952 产品页完整 Product details、Features、Description 技术正文；NVIDIA Jetson Thor 页面完整产品技术正文、模块与开发套件规格、脚注及适用软件链接目录。未读取 NDA 下安全文档或所有链接的数据手册；本页不是对原二手综述全文的通读声明。
- **通读：** 已完成

## 原文与版本

- [TI BQ76952](https://www.ti.com/product/BQ76952)；本次读取材料 SHA-256：`e523850b6ec77cba270dba2c019940dec896cad5fbe1f43962be08d134d531d4`。
- [NVIDIA Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)；本次读取材料 SHA-256：`ba7a2dada5cad088b4ead4c6c3fcbada8d8aed5db7ce4141f737ea252eebeb36`。

## 核对结论

- BQ76952 是 3–16 串电池监测/保护器，支持 Li-ion、Li-polymer 与 LiFePO4；有电压、电流、温度保护和均衡能力，但一个监测 IC 不等于完整电池包或认证 BMS。
- 产品页列出高侧 NFET 驱动、独立 ADC、库仑计量、外部热敏电阻及 I2C/SPI/HDQ 接口；电池化学体系、串数、阈值与功率开关需按实际电芯/拓扑配置。
- T5000 规格为最高 2070 FP4 sparse TFLOPS、128 GB LPDDR5X、273 GB/s 与 40–130 W 功率范围；这不是所有模型的实测速率，也不能与 INT8 TOPS 或其他精度直接横比。
- T4000 的内存/功耗等与 T5000 不同；模块、开发板和外设总系统功耗应分开做预算。厂商页面不能证明其在人形市场的占有率。

## 对 wiki 的映射

- [humanoid-hardware-101-power-compute-electronics](../../wiki/overview/humanoid-hardware-101-power-compute-electronics.md)
