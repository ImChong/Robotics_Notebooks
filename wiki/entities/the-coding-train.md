---
type: entity
project_id: the-coding-train
project: https://thecodingtrain.com/
code: https://github.com/CodingTrain/thecodingtrain.com
tags: [creative-coding, education, javascript, open-source]
status: complete
updated: 2026-10-10
related:
  - ./p5js.md
  - ./ml5js.md
sources:
  - ../../sources/sites/the-coding-train.md
  - ../../sources/repos/thecodingtrain-website.md
summary: "The Coding Train 是 Daniel Shiffman 发起的创意编程学习社区，通过结构化视频 Track、单集教程、挑战和直播，教授 Processing、JavaScript、p5.js 与浏览器端机器学习；在机器人语境下适合作为交互可视化、模拟演示和初学者编程材料。"
---

# The Coding Train

**The Coding Train** 是面向初学者与好奇学习者的创意编程社区，由 Daniel Shiffman 于 2015 年发起；以系列课程、Coding Challenges 和直播教授编程与数字创作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| JS | JavaScript | p5.js 与网页交互课程使用的主要语言 |
| API | Application Programming Interface | ml5.js 教程中调用模型能力的程序接口 |
| ML | Machine Learning | The Coding Train 的 ml5.js Track 所介绍的浏览器机器学习主题 |

## 为什么重要

- **课程是可导航的，不只是视频清单：** 官方 Tracks 将视频按学习顺序组织；初学者可从基础语法走向画布、图像、声音、模拟与机器学习。
- **从代码到可见反馈：** p5.js 的交互草图让变量、事件、向量、噪声与模拟的效果立即可见，适合教学和快速构想。
- **有一条浏览器 ML 学习路径：** 官方站提供 ml5.js 课程，ml5.js 官网也将 The Coding Train 的姿态和手部检测内容列作示例，形成「库文档 + 逐步演示」的互补。
- **机器人相关用途：** 可借其材料入门 JavaScript、视觉化演示和交互原型；它本身不是机器人中间件、仿真器或实时控制系统。

## 核心结构与学习路径

官方站把内容组织为三类入口：按主题排序的视频 Track、独立 Coding Challenges、以及直播与社区支持。对机器人研究者，比较自然的路径是：

1. 用 JavaScript 基础 Track 建立变量、函数、对象与事件循环概念。
2. 用 [p5.js](./p5js.md) 做画布绘图、交互和二维/三维可视化。
3. 用 [ml5.js](./ml5js.md) 体验浏览器端姿态、手部、面部或音频模型。
4. 将网页原型与机器人端分开：通过明确的网络接口传输经过校验的数据，再由有安全约束的机器人控制栈消费。

## 工程实践

| 需求 | 合适起点 | 接入机器人前还需补足 |
|------|----------|--------------------|
| 教学可视化 | p5.js Track / Coding Challenges | 仿真器和物理模型需另选 |
| 摄像头交互演示 | ml5.js Track 与官方示例 | 相机标定、时间同步、延迟测量与置信度门限 |
| 远程操作界面原型 | p5.js 画布与浏览器事件 | 鉴权、限速、断连保护和机器人侧急停 |

开源核查：课程网站的官方源代码仓公开且采用 MIT；这一许可只说明网站代码仓，课程视频及第三方库各自遵循其发布条款。

## 局限与风险

- 视频课程是学习材料，不是经过机器人安全验证的控制接口；演示代码需核对版本、依赖与硬件假设。
- 浏览器摄像头推理适合交互原型与低风险演示。真机使用还需处理隐私、网络延迟、坐标变换、故障保护与安全认证。
- 各 Track 的内容和示例版本会变化；复现时优先使用站点当前课程与对应官方文档。

## 关联页面

- [p5.js](./p5js.md) — 课程中用于网页创意编程和可视化的库。
- [ml5.js](./ml5js.md) — 课程中的浏览器端机器学习接口。
- [计算机视觉骨干网络](../concepts/vision-backbones.md) — 区分教育演示模型与机器人策略所需视觉表征。

## 参考来源

- [The Coding Train 官方教学站归档](../../sources/sites/the-coding-train.md)
- [CodingTrain/thecodingtrain.com 仓库归档](../../sources/repos/thecodingtrain-website.md)
- [The Coding Train 官方网站](https://thecodingtrain.com/)
- [About The Coding Train](https://thecodingtrain.com/about)

## 推荐继续阅读

- [Code! Programming with p5.js](https://thecodingtrain.com/tracks/code-programming-with-p5-js)
- [A Beginner's Guide to Machine Learning in JavaScript with ml5.js](https://thecodingtrain.com/tracks/ml5js-beginners-guide)
