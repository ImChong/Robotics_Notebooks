# Movez：Robotics Lab / 免费在线 3D 机器人工程实验室（2026-11-10）

- **资料类型：** blog（作者社交平台首发公告；不是论文或技术白皮书）
- **发布者：** Movez（@0xMovez）
- **发布日期：** 2026-11-10
- **原始发布账号：** https://x.com/0xMovez
- **原始帖文可读镜像：** https://teapawt.net/0xMovez?filter=posts （检索 2026-11-10 带有 Robotics Engineering、Fable 5.2、8 real boards 的作者帖文）
- **作者提供的项目链接：** https://robotics-lab-app.vercel.app/
- **核对日期：** 2026-11-12
- **核对范围：** 逐项核对作者 2026-11-10 原始公告全文中有关价格、板卡数量、3D 生成方式、组装/焊接与 LED/传感器/电机交互的全部描述；通过作者账号镜像读取其完整发文。未取得独立技术文档，亦未读取到项目网页内部实现。
- **通读：** 已完成
- **沉淀到 wiki：** [Robotics Lab（Movez）](../../wiki/entities/robotics-lab-movez.md)

## 作者公告的核心声明（归纳，非独立性能验证）

1. 作者提供一个无需注册、可免费体验的在线 Robotics Lab，称可在浏览器中学习 GPIO。
2. 作者宣称有 **8 款**真实开发板的 3D 可视化，明确举例 **Arduino Uno、Raspberry Pi 4**；这条公告未给出全部八款名单，不能据此确认 ESP32 型号。
3. 作者展示的交互类型包括放置芯片、接口和排针、模拟焊接，以及 LED 点亮、传感器读取、电机驱动的接线实验。
4. 作者称所有 3D 模型由 **Fable 5.2** 以代码生成、未使用现成模型文件或照片。此为作者自述；未获取源代码/构建配置以独立验证。
5. 「无需购买 500 美元以上入门套件」是作者宣传对比，**不是成本节省实验**，也不代表真实硬件实验可全部替代。

## 阅读与证据边界

- **已核实的是作者确有上述公开声明**，不是这些功能的精度、覆盖范围、底层工程实现均获得独立验证。
- 项目站点在本次网页读取工具中无法提取正文；参见 [项目网址归档](../sites/robotics-lab-movez.md)。
- 另有独立体验文章 [Neon Gamewall（2026-11-11）](https://neongamewall.com/game/robotics-lab)：实测 Arduino Uno 18 部件拆装、元件识别问答，以及 Hello, LED 演示打印 D13 HIGH/LOW。它属于**第三方体验**，不是官方文档；且该体验者未验证其余七款板卡、完整课程或移动端。
- 没有找到可确认归属于本项目的公开 GitHub 仓库、许可证或完整技术设计文档，不能称其「开源」。

## 与已有项目区别

- [Robotics Bench](../../wiki/entities/robotics-bench.md) 的地址是 robotics-engineer.vercel.app，介绍 Colorfigures 制作的互动等距板卡演示；**不能与**本项目 Robotics Lab（robotics-lab-app.vercel.app、作者 Fable 5.2 声明）**合并或混用模型数量**。

## 参考来源

- 作者发文：[@0xMovez](https://x.com/0xMovez)，2026-11-10；[作者完整发文镜像](https://teapawt.net/0xMovez?filter=posts)。
- 项目入口：[Robotics Lab](https://robotics-lab-app.vercel.app/)。
- 独立体验：[Neon Gamewall / Robotics Lab](https://neongamewall.com/game/robotics-lab)（2026-11-11）。
