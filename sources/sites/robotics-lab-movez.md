# Robotics Lab（Movez）— 在线 3D 开发板交互教学站

- **类型：** site（作者官方在线演示）
- **项目名称：** Robotics Lab
- **开发者 / 发布者：** Movez（@0xMovez）
- **项目页：** https://robotics-lab-app.vercel.app/
- **首发声明：** [Movez / X（2026-11-10）](https://x.com/0xMovez)；[作者帖文镜像](https://teapawt.net/0xMovez?filter=posts)
- **发布日期：** 2026-11-10（按作者公告）
- **核对日期：** 2026-11-12
- **源码：** 未找到可核实的本项目公开源码 GitHub URL；**开放状态待核实**，不是已证实闭源
- **许可证：** 未找到可核实的项目许可证
- **核对范围：** 尝试访问项目站点，网页提取工具返回内部错误、未获得正文；完整核读作者首发声明，另核验第三方体验记录以区分可观察操作与未验证的宣传。未完成对官方交互页面全部功能或代码的核验。
- **通读：** 未完成
- **一句话说明：** Movez 发布的浏览器 3D 开发板学习实验室，作者称支持八款开发板、元件装配与虚拟焊接、LED/传感器/电机教学实验；仿真精度与项目源码未核实。
- **沉淀到 wiki：** [Robotics Lab（Movez）](../../wiki/entities/robotics-lab-movez.md)

## 站点核验记录

- **在线地址确认：** 作者明确给出 https://robotics-lab-app.vercel.app/ 作为入口；独立体验文章也链接到同一地址。
- **访问限制：** 2026-11-12 通过当前网页读取工具直接打开该 Vercel 站点未能取得可解析内容；这**不代表网站下线**，不能据此声称界面已全部实测。
- **公开仓库与数据：** 现有公开入口未提供可核实的、归属于该 Robotics Lab 的官方代码仓库或数据集地址；不凭其他名字近似的 Robotics Lab 仓库冒认源码。

## 能力与核验状态

| 说法 / 功能 | 证据级别 | 边界 |
|---|---|---|
| 免费、无需注册 | 作者声明；Neon Gamewall 实际浏览器打开 | 是否始终免费、后续政策未知 |
| 8 款开发板 | 作者声明，第三方概述 | 完整八款型号未核对；ESP32 不在作者帖的明确举例中 |
| Arduino Uno / Raspberry Pi 4 | 作者首发列举 | 不代表 Raspberry Pi 的 Linux/内核逻辑已被仿真 |
| 18 个元件拆装、模拟焊接 | Neon Gamewall 的 Uno 实测 | 实测的是可视化教学交互，不等于焊料热物理模型 |
| Hello, LED / D13 HIGH、LOW | Neon Gamewall 的 Chrome 实测 | 不能据此确认其它七款板卡的固件执行精度 |
| 传感器读取、电机驱动 | 作者宣称 | 尚无可核验的电机动力学/总线/负载精度证据 |
| Fable 5.2 纯代码生成 3D | 作者自述 | 缺少源码、生成记录和资产清单佐证 |
| 节省 500 美元套件成本 | 宣传对比 | 无标准化 BOM、课程效果或成本对照试验 |

## 与 Robotics Bench 区分

同一作者历史介绍过 [Robotics Bench](../../wiki/entities/robotics-bench.md)（robotics-engineer.vercel.app，Colorfigures 相关，宣称 13 类硬件）；这里的 Robotics Lab 是 **robotics-lab-app.vercel.app**，2026-11-10 发布，作者宣称 Fable 5.2 和 8 款开发板。两者不能视为同一项目版本或共用硬件清单。

## 后续核验事项

1. 使用可访问的浏览器实测八款板卡、各学习模式与移动端操作。
2. 核对是否真支持运行用户自定义固件、I/O 模型、传感器信号与电机动态，或仅内建固定课程动作。
3. 从作者官方入口寻找明确的 GitHub、许可证或技术说明；若有源码再补运行时序图。

## 参考来源

- [作者 2026-11-10 首发声明归档](../blogs/movez-robotics-lab-2026-11-10.md)
- [Neon Gamewall 2026-11-11 体验记录](https://neongamewall.com/game/robotics-lab)
