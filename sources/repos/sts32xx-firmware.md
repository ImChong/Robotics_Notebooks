# STS32XX Firmware（Feetech 伺服固件）

- **类型：** 官方源码仓库
- **链接：** <https://github.com/roboninecom/STS32XX-Firmware>
- **许可证：** Apache-2.0；第三方代码许可见仓库 NOTICE。
- **核对版本：** `444f775c05e637c3743ec12ade6c1f9ae174177e`，2026-11-03。
- **核对范围：** README、DETAILED_DOCUMENTATION、QUICKSTART，以及运行入口和控制相关源码。
- **一句话说明：** 面向 GD32F130 的 STS3215 TTL 伺服固件，提供兼容模式和带局部轨迹规划、位置控制及重力补偿的扩展模式。

## 入口与组成

- [详细文档](https://github.com/roboninecom/STS32XX-Firmware/blob/444f775c05e637c3743ec12ade6c1f9ae174177e/DETAILED_DOCUMENTATION.md) 与 [快速上手](https://github.com/roboninecom/STS32XX-Firmware/blob/444f775c05e637c3743ec12ade6c1f9ae174177e/QUICKSTART.md)。
- `npm run replay -- CONFIG.json RECORDING --run`：`tools/poses/replay-cli.js` 加载录制及可选重力模型，经 `host/arm-control/session.js` 启动控制器。
- `firmware/app/arm_extension.c` 处理扩展指令；`firmware/app/runtime.c` 调度控制；`firmware/core/control/arm_control.c` 和 `arm_motion.c` 实现控制和轨迹。
- 固件及上位机实现公开；无论文或系统性精度数据集被该文档作为本项目材料列出。

## 测试与边界

README 展示 SO-101、SO-102 刷写前后视频，强调演示数字仅描述录制场景。用户提供的约 35→10 mm 和 25→0–2 mm 具体读数未能从文档文本独立核验，暂不作为已验证的量化结果。齿隙和机架变形仍影响末端位置。固件刷写可能损坏舵机或擦除标定；停止通信不会自动释放力矩，须按官方说明进行备份、标定和安全检查。

## 对 wiki 的映射

- [STS32XX Firmware](../../wiki/entities/sts32xx-firmware.md)
