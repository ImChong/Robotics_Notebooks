---
type: entity
tags:
  - tactile-sensing
  - magnetic-sensing
  - visuo-tactile
  - manipulation
  - hardware
  - open-source
  - new-york-university
status: complete
updated: 2026-10-13
arxiv: "2506.09994"
code: https://github.com/notvenky/eFlesh
related:
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../concepts/contact-rich-manipulation.md
  - ./paper-taco-tactile-sensor-benchmark.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/eflesh_arxiv_2506_09994.md
  - ../../sources/repos/eflesh.md
  - ../../sources/sites/e-flesh-com.md
summary: "eFlesh（arXiv:2506.09994，NYU）是可定制的磁触觉皮肤：把磁铁嵌入 cut-cell 3D 打印微结构，用 Hall 磁力计读取形变；支持接触/力估计、滑移检测与视触觉精细操作，设计工具和代码开源。"
---

# eFlesh：可定制磁触觉皮肤（arXiv:2506.09994）

**eFlesh**（*Highly customizable Magnetic Touch Sensing using Cut-Cell Microstructures*，[论文](https://arxiv.org/abs/2506.09994)，[项目主页](https://e-flesh.com/)，[代码](https://github.com/notvenky/eFlesh)）由 **New York University** 团队提出。它把磁铁封装在可定制的 3D 打印弹性结构里，外力让磁铁相对 Hall 磁力计移动；磁场变化经学习模型转成接触位置、法向力、剪切力或滑移判断。

## 一句话理解

**把“按压导致的软结构形变”转成“磁铁位置变化”，再用磁力计读出变化。** 柔性层无需铺设密集电气走线，结构外形与刚度可按手指、夹爪或足部定制。

## 结构与信号链

```mermaid
flowchart LR
  FORCE[接触力 / 表面形变] --> MICRO[cut-cell 弹性微结构变形]
  MICRO --> MAG[嵌入磁铁发生位移]
  MAG --> HALL[分离式 Hall 磁力计读取磁场]
  HALL --> MODEL[回归 / 分类模型]
  MODEL --> OUT[接触位置、力、滑移或策略输入]
```

1. **打印结构：** TPU 等材料组成参数化微结构，可通过单元尺寸、材料刚度和空间分层调整机械响应。
2. **嵌入磁铁：** 磁铁置于结构中层的口袋内；外壳压缩或剪切时磁铁随之移动。
3. **读取磁场：** 独立磁力计电路板位于传感结构下方或内部，测量磁场向量变化。
4. **学习解码：** 模型从原始磁信号回归接触位置/力，或分类滑移；同一信号也可作为视触觉策略输入。

## 主要结果（论文设定）

| 能力 | 论文报告 | 怎么读 |
|---|---:|---|
| 接触定位 | RMSE 0.5 mm | 接触点与深度回归 |
| 法向力估计 | RMSE 0.27 N | 论文标定数据范围内的预测误差 |
| 剪切力估计 | RMSE 0.12 N | 平面切向力回归 |
| 未见物体滑移检测 | 95% 准确率 | 线性分类器在留出的物体上评估 |
| 视触觉精细操作 | 平均成功率 >90%；相对 vision-only 提升约 41% | 四种接触丰富任务；特定 xArm 7、示范数据和评测协议 |

实验中的操作任务包括 USB 插入、插头插入、信用卡滑动和白板擦拭。视觉编码器与磁信号 MLP 特征输入 Transformer 行为克隆策略，策略预测动作 chunk。

## 设计工具与复现路径

- 官方代码仓库提供 C++ 微结构构建工具、regular.ipynb 与 cut-cell.ipynb，将凸形 OBJ/STL 转成贴合目标形状的网格结构。
- 可调整 cell size、等效刚度和局部层参数；随后添加磁铁口袋与磁力计板安装槽，导出并 3D 打印。
- 仓库 README 的主要流程是克隆（含 submodules）→按 README 配置依赖→构建 microstructure inflator→运行 notebook→生成 STL。
- 项目主页提供 eFlesh STL、可视化器、样品申请和磁力计入口；代码仓库标注 MIT License。

**材料成本口径：** 作者称普通现成磁铁成本低于 **$5**，不代表整套传感器总价；仍需 TPU、3D 打印、磁力计 PCB 和装配。TacO 基准论文将用于比较的 eFlesh 传感器列为约 **$35**，口径与仅磁铁成本不同。

## 研究价值

- **可按形状定制：** 将同一传感原理延伸到指尖、夹爪和机器人足等外形，减少专用硅胶模具或密集布线依赖。
- **不止触点检测：** 既可估计接触/力和滑移，也能把触觉读数接入视觉策略，以改善遮挡下的精细接触操作。
- **接触丰富操作证据：** [TacO 触觉传感器基准](./paper-taco-tactile-sensor-benchmark.md) 将 eFlesh 与 FSR、FlexiTac、eGain、Daimon 和 Contact Mic 等纳入同一真机策略对比，提供跨硬件参照。

## 局限与工程风险

- **磁干扰：** 传感器之间的磁串扰及外部磁性/电子设备可改变读数。论文通过相邻磁铁交替极性来减弱串扰；集成前仍需在目标机器人上测试。
- **结构参数会影响读数：** 柔软度、几何与磁铁布置共同决定灵敏度和承载。嵌入磁铁也会增加结构刚度，不能只按 STL 外形推断性能。
- **需任务标定：** 报告的毫米/牛顿误差依赖特定样件、磁力计、训练数据和校准设置；换材料、尺寸或硬件后应重新采数验证。
- **CAD 工具限制：** 仓库面向凸形 OBJ/STL 输入及特定几何依赖；复杂非凸形状或更细微结构可能需要手动拆分、改模和更高打印分辨率。
- **“廉价”不等于即插即用：** 需要加工、嵌磁、板卡固定、磁干扰隔离和模型标定。

## 关联页面

- [Tactile Sensing（触觉感知）](../concepts/tactile-sensing.md) — 触觉模态与传感器路线总览
- [Visuo-Tactile Fusion（视触觉融合）](../concepts/visuo-tactile-fusion.md) — 将视觉与触觉编码结合用于操作
- [Contact-Rich Manipulation（接触丰富操作）](../concepts/contact-rich-manipulation.md) — 插入、擦拭与滑动等任务
- [TacO 触觉传感器操作基准](./paper-taco-tactile-sensor-benchmark.md) — eFlesh 与五种其他传感器的真机对比
- [模仿学习](../methods/imitation-learning.md) — 视触觉策略使用示范训练
- [Manipulation 任务](../tasks/manipulation.md)

## 参考来源

- [论文来源归档](../../sources/papers/eflesh_arxiv_2506_09994.md)
- [代码仓库归档](../../sources/repos/eflesh.md)
- [项目主页归档](../../sources/sites/e-flesh-com.md)
