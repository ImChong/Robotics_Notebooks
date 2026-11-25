# YHorizon-JM｜开源一体化关节电机

- **类型：** repo
- **仓库：** <https://github.com/YHorizon-Robot/YHorizon-JM>
- **官方项目页：** <https://yhorizon.tech/jm/>
- **固定版本：** `de0f16cdb62d3cb30dfe241e76dd17d5c769195b`
- **入库日期：** 2026-11-25
- **核对范围：** 官方项目页、仓库 README / CHANGELOG / LICENSE / SECURITY、机械和硬件树、固件板级配置、入口、ADC/PWM/外环中断、伺服与组同步实现、CAN 协议、上位机与 SDK 文档。静态核查，未编译、烧录或上台架。
- **通读：** 已完成

## 证据与边界

- [机械树](https://github.com/YHorizon-Robot/YHorizon-JM/tree/de0f16cdb62d3cb30dfe241e76dd17d5c769195b/mechanical) 有装配体 STEP、CNC STEP、30/31 齿编码器模型、传动齿轮 PDF 和机械 BOM；README 中的装配体文件名已过时，应从目录打开。 [电控树](https://github.com/YHorizon-Robot/YHorizon-JM/tree/de0f16cdb62d3cb30dfe241e76dd17d5c769195b/hardware) 有 EasyEDA Pro 工程、Gerber、原理图 PDF、电控 BOM。文件存在不等于加工适配或验收通过。
- 公开版标称直径 57 mm、厚 52 mm、约 300 g，WK4310H 无框电机与 8:1 行星减速，额定 3 Nm、最大 6 Nm、最高供电 48 V；没有独立验证持续力矩热条件、峰值时长、效率、反驱或寿命。
- [固件树](https://github.com/YHorizon-Robot/YHorizon-JM/tree/de0f16cdb62d3cb30dfe241e76dd17d5c769195b/firmware)：GD32F303CCT6 默认 current 模式，支持 voltage 模式。ADC 提前处理能否走通取决于状态；PWM 更新处理预计算占空比或回退；4 kHz 外环定时器更新模式参考，主循环负责通信、辨识和状态服务。
- [SDK](https://github.com/YHorizon-Robot/YHorizon-JM/tree/de0f16cdb62d3cb30dfe241e76dd17d5c769195b/sdk)：C++17 / Python 加载同一原生动态库，Rust 可直接依赖 crate。与 [Rust 上位机](https://github.com/YHorizon-Robot/YHorizon-JM/tree/de0f16cdb62d3cb30dfe241e76dd17d5c769195b/host-rs) 使用 Classic CAN Motor Protocol V1.0。旧 Python RS05 / DM4310 界面不是 YHorizon-JM 控制后端，GD32C113 / CAN FD 属后续规划。README 所列部分 `docs/` 路径不在固定版本 Git 树；使用实际 SDK 协议和源码核对。
- [固定版本 LICENSE](https://github.com/YHorizon-Robot/YHorizon-JM/blob/de0f16cdb62d3cb30dfe241e76dd17d5c769195b/LICENSE) 指定软件 GPL-3.0-only、机械和电控 CERN-OHL-S-2.0，声明在遵守义务时允许商用；第三方文件另看自己的许可。项目站说明仍称禁止商用，与仓库文件不一致。笔记不裁决该差异，商业集成前应向权利人确认授权范围，闭源路径是另行签署的商业许可，不是未签模板。

## Wiki 映射

- [YHorizon-JM 关节模组](../../wiki/entities/yhorizon-jm.md)
