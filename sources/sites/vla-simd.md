# vla.simd 项目页

> 来源归档（site）

- **标题：** vla.simd: Efficient CPU Inference for Language-Conditioned Manipulation
- **类型：** site
- **链接：** <https://vla-simd.github.io/>
- **arXiv：** <https://arxiv.org/abs/2609.24274>
- **入库日期：** 2026-10-18
- **一句话说明：** 展示 `vla.simd` CPU 推理引擎与 `IMPACT` 策略的官方网站，包含互动式 SMK 寄存器 tile 计算器及 SO-101、UR10e 真机多任务实验视频与置信区间统计。
- **沉淀到 wiki：** [`wiki/entities/paper-vla-simd.md`](../../wiki/entities/paper-vla-simd.md)
- **通读：** 已完成
- **核对范围：** 通读离线抓取的项目官网 HTML `sites/vla_simd.html`（L1-496 全文），重点核对顶部导航栏链接、摘要与 4 个核心数字卡片（`≈1.4×`、`41 Hz`、`76.4%`、`≤2.6%`）、互动式 SMK Tiling 算法模块说明、Method 图注及 SO-101 / UR10e 6 组真机视频元数据与页脚声明。

## 开源状态

- **双盲匿名代码可见、实名 GitHub 与权重待论文接收后发布**（2026-11-14 官网 DOM 核查）：官网导航栏未放置 `github.com` 链接，而是指向双盲匿名仓库：
  - 论文预印版镜像：`https://anonymous.4open.science/api/repo/lerobot-vla-simd/file/docs/vla.simd.pdf`
  - 推理引擎代码：`https://anonymous.4open.science/r/vla-simd`
  - LeRobot 接入分支：`https://anonymous.4open.science/r/lerobot-vla-simd`
  - 导航下方注释注明 *"The author has been anonymized from the linked repositories."*，页脚 `<footer>` 明确写出 *"Code and checkpoints will be released on acceptance."*。

## 项目页核查要点

- **核心指标横幅（`#highlights`）：** 相对编译版 PyTorch 中位加速 `≈1.4×`；`IMPACT` 在树莓派 Pi 5 标称动作供给 `41 Hz`；无机器人预训练下 LIBERO 四套件平均成功率 `76.4%`；语言条件模块在每次查询计算量占比 `≤2.6%`。
- **SMK Tiling Algorithm 互动演示（`#smk`）：** 页面内嵌 `#smk-app` 互动组件，对应论文 Algorithm 1 与约束公式 `(2)-(4)`（寄存器上限、FMA 延迟隐藏、访存带宽匹配及 `SPILLS` 栈溢出排除），支持按硬件参数对比 `AVX2 (6×16)` 与 `NEON (4×16)` 候选 tile。
- **真机展示口径（`#robot`）：**
  - **SO-101 Multi-task（IMPACT fp32，共场景指令跟随）：** 共享场景同帧切换指令，平均往返延迟 `996.6 ms`；`"Put the tape into the box"` `90% (70–97)`、`"Put the tape into the cup"` `85% (64–95)`、`"Put the cup into the box"` `60% (39–78)`；页面指出更改放置目的地比更改抓取目标更容易区分。
  - **SO-101 Long-horizon（20 条示教，独立抽屉 checkpoint）：** `"Put the tape into the drawer"`（开抽屉→放胶带→关抽屉）`65% (43–82)`，往返 `996.6 ms`。
  - **UR10e + SmolVLA（跨主机 CPU 部署对比，视频 3× 速）：** Apple M4 往返 `1177 ms`、成功率 `60% (39–78)`；AMD Ryzen 5 5500 往返 `2343 ms`、成功率同样 `60% (39–78)`（`12/20`），官网如实提示小样本置信区间重叠尚不足以分辨主机延迟差异。
