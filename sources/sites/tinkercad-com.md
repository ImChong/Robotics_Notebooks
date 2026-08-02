# Tinkercad（tinkercad.com）

> 来源归档

- **标题：** Tinkercad — Free web app for 3D design, electronics, and coding
- **类型：** site（Autodesk 在线 STEM / 创客教育平台）
- **来源：** Autodesk, Inc.（欧特克）
- **链接：** https://www.tinkercad.com/
- **帮助 / Learn：** https://www.tinkercad.com/learn
- **Circuits 学习入口：** https://www.tinkercad.com/learn/circuits
- **入库日期：** 2026-08-02
- **一句话说明：** 浏览器端免费应用，把 **3D 设计**、**电路仿真（Arduino / micro:bit）** 与 **Codeblocks 积木编程** 合在同一账号与课堂体系里；面向 K–12 / 入门创客与机器人启蒙，不是开源机械 CAD 或 RL 物理仿真器。
- **开源状态：** **确认未开源** — 闭源 SaaS（Autodesk 产品）；免费使用，无公开训练/仿真内核源码仓库。可导出 STL/OBJ 等设计资产到下游工具。
- **沉淀到 wiki：** [Tinkercad](../../wiki/entities/tinkercad.md)

---

## 平台定位（官网公开文案摘要，2026-08-02）

官网首页（JS 渲染；以公开文案与 Learn 导航为准）：

- **口号级定位：** 「All you need is a 'what if...'」— 免费 Web App，覆盖 **3D design、electronics、coding**；宣称全球逾 **1 亿** 用户、**8 亿+** 设计。
- **三大卖点：** Free for everyone（免下载）/ Learn by doing / Safe for all ages（ad-free、KidSAFE COPPA 认证）。
- **三大工作区：**
  1. **3D Design** — 积木式实体造型，面向可打印零件与产品概念。
  2. **Circuits** — 面包板与虚拟元件；从 LED 到 Arduino / micro:bit 编程仿真。
  3. **Codeblocks** — Scratch 风格积木，驱动参数化 / 动态 3D 造型。
- **课堂：** Start Classroom / Join Class；课程对齐 **Common Core** 与 **NGSS**；学科目录含 **Robotics** 等。

---

## 能力分区（与机器人栈的映射）

| 工作区 | 公开能力 | 机器人语境读法 |
|--------|----------|----------------|
| **3D Design** | 形状布尔、分组、STL/OBJ/GLTF/SVG/USDZ 等导出；可转 Fusion 360 | **入门结构件 / 外壳 / 夹具草模**；再进 [FreeCAD](../../wiki/entities/freecad.md) 或 Fusion 做制造级参数化 |
| **Circuits** | Arduino Uno、BBC micro:bit、ATtiny 等；Blocks / Text（C++）/ 组合视图；传感器与舵机类元件 | **焊板前的 MCU 入门仿真**；与 [Wokwi](../../wiki/entities/wokwi.md) 同属固件教学层，MCU 覆盖面更窄 |
| **Codeblocks** | 变量、条件、模板块；积木生成参数化 3D | 程序化造型启蒙；研究级脚本 CAD 见 [Text-to-CAD](../../wiki/concepts/text-to-cad.md) / CadQuery |
| **Sim Lab** | 重力、碰撞、材料属性；舵机/步进/连续电机等机构示意 | **STEAM 物理玩具级** 机构直觉，**不能** 替代 MuJoCo / Isaac 接触动力学 |

---

## Circuits 教程结构（learn/circuits 导航，2026-08-02）

公开分类包括：

- **Basics** — 仿真启动、元件编辑/布线、面包板、欧姆定律、串并联
- **micro:bit** — LED 矩阵、按键/运动传感、温湿度光、面包板外设
- **Arduino** — 数字/模拟 IO、串口监视器、PIR/光敏/温度/超声波等
- **Arduino Kit 项目序列** — Spaceship Interface、Love-o-Meter、Color Mixing Lamp 等入门项目

编程路径：Arduino 可用 **Blocks → Blocks+Text（自动生成 C++）→ Text**；micro:bit 为 Scratch 风格积木（接近 MakeCode 体验）。

---

## 与知识库邻近工具对照

| 工具 | 层 | 相对 Tinkercad |
|------|----|----------------|
| [Wokwi](../../wiki/entities/wokwi.md) | MCU + 外设仿真 | ESP32/STM32/Pico、GDB、wokwi-cli CI；更偏工程 bring-up |
| [KiCad](../../wiki/entities/kicad.md) | 原理图 / PCB / Gerber | 制造级 EDA；Tinkercad Circuits 仅教学仿真，不替代打样真值 |
| [FreeCAD](../../wiki/entities/freecad.md) / [Blender](../../wiki/entities/blender.md) | 机械 B-rep / DCC | 制造与资产层；Tinkercad 适合零门槛草模与课堂 |
| [SimpleFOC](../../wiki/entities/simplefoc.md) | FOC 固件 | Arduino 栈可先在 Tinkercad 学 GPIO/PWM/串口，再上真驱动板 |

---

## 抓取说明

- **官网首页** 为重度客户端渲染，直抓 HTML 多为 i18n key；本归档以 **2026-08-02** 公开文案、Learn/Circuits 导航结构及 Autodesk 入门材料摘要为准。
- **源码：** 产品闭源；无独立开源仿真内核仓库可收录至 `sources/repos/`。

## 参考链接

- 官网：<https://www.tinkercad.com/>
- Learn：<https://www.tinkercad.com/learn>
- Circuits Learn：<https://www.tinkercad.com/learn/circuits>
- Autodesk Getting Started Guide（PDF）：<https://damassets.autodesk.net/content/dam/autodesk/www/pdfs/tinkercad-getting-started-guide.pdf>
