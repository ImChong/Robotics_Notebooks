# aigeekops/learn-esp32

> 来源归档（repo）

- **标题：** Learn ESP32 · ESP32 交互学习站
- **项目页：** <https://aigeekops.github.io/learn-esp32/>
- **代码：** <https://github.com/aigeekops/learn-esp32>
- **类型：** 浏览器端 ESP32 学习课程与示例生成器
- **许可证：** MIT（第三方依赖按各自许可证）
- **核查日期：** 2026-10-24
- **关联项目页：** [Learn ESP32 课程站](../sites/aigeekops-learn-esp32.md)
- **相关知识页：** [Wokwi](../../wiki/entities/wokwi.md)（固件仿真工具）

## 一句话摘要

一个以 Vite、TypeScript、Three.js 实现的交互课程站，围绕 ESP32-DevKitC V4 / ESP32-WROOM-32E 提供 16 课 3D 场景、引脚与接线说明、参数化 Arduino C++ 模板以及浏览器内教学行为模型。

## 课程与代码入口

| 路径 / 模块 | 内容 |
|---|---|
| `docs/CURRICULUM.md` | 16 课课程定义、器件选择、接线、参数和模拟范围 |
| `src/lessons.ts` | 课程、引脚和参数定义 |
| `src/firmware.ts` | Arduino C++ 示例模板 |
| `src/simulation.ts` | 浏览器内教学行为模型 |
| `docs/REFERENCES.md` | ESP32 板卡、Arduino-ESP32 与传感器的一手参考资料 |
| `docs/validation/2026-10-24-firmware.json` | 作者附带的当日默认固件编译摘要 |

README 提供 `npm ci`、`npm run dev` 启动方式，以及 `npm test`、`npm run build`、`npm run firmware:export` 和 `npm run firmware:check` 等命令。真机代码以 Arduino-ESP32 3.3.12 为基线。

## 开源状态与工程边界

**已开源：** GitHub 仓库公开，根目录 MIT 许可证，并附第三方声明。前端代码和课程示例均可查阅；仓库不包含云端后端或在线硬件服务。

- 页面模拟交互并展示示例，不会在浏览器中执行任意 C++，也不会连接访问者的真实板卡或局域网。
- README 与验收文档记录默认 16 课固件可通过指定 Arduino-ESP32 工具链编译；文档明确未完成实物接线、GPIO/ADC/PWM 测量或无线连接验收。
- 项目首页说明其为社区学习项目，与 Arduino、Espressif 及参考站 Learn Arduino 无隶属关系；界面灵感与第三方素材许可分别以其声明为准。

## 一手来源

- [官方 GitHub 仓库](https://github.com/aigeekops/learn-esp32)
- [README](https://github.com/aigeekops/learn-esp32/blob/main/README.md)
- [课程定义](https://github.com/aigeekops/learn-esp32/blob/main/docs/CURRICULUM.md)
- [开发验收记录](https://github.com/aigeekops/learn-esp32/blob/main/docs/DEVELOPMENT_REPORT.md)
- [参考资料记录](https://github.com/aigeekops/learn-esp32/blob/main/docs/REFERENCES.md)
- [ESP-IDF 官方 ESP32 入门指南](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html)
