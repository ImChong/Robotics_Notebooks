# CSF 官方项目页

> 来源归档（site / 项目页）

- **类型：** 论文项目页
- **链接：** <https://lzyang2000.github.io/csf/>
- **论文：** <https://arxiv.org/abs/2610.12467>
- **代码：** <https://github.com/lzyang2000/csf>
- **核查日期：** 2026-10-12
- **一句话说明：** 项目页展示场景语义门控、采样过程 CBF-QP、执行期安全盾，以及 Unitree G1 对箱子出拳和检测到人后改写动作的对比。

## 页面核查

- 作者单位：California Institute of Technology（6 位作者）与 New York University（Daniel Weng）。
- 项目页提供 Paper、Code、视频与交互 Demo；同一“Throw a forward punch”指令下，动作可对箱子执行，在人出现时改为安全动作。
- 系统分三段：自然语言规则与实时场景实体选择约束；冻结生成器在每个采样步由安全参考 CBF-QP 过滤；场景变化时，运行时安全盾只改写未执行参考。
- 四种生成器为 Kimodo、ARDY、ECHO、MotionHiFlow；真机 G1 部署使用 ARDY、CSF 与 SONIC 跟踪策略。
- README / 代码归档见 [sources/repos/lzyang2000_csf.md](../repos/lzyang2000_csf.md)。

## 对 wiki 的映射

- [wiki/entities/paper-csf-contextual-safety-filter.md](../../wiki/entities/paper-csf-contextual-safety-filter.md)
