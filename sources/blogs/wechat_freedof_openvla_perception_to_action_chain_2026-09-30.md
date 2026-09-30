# 机器人看懂之后，下一步怎么动？以原始 OpenVLA 为例，走通图像、指令与动作的链路

> 来源归档（blog / 微信公众号 · 自由度 FreeDof）

- **标题：** 机器人看懂之后，下一步怎么动？以原始 OpenVLA 为例，走通图像、指令与动作的链路
- **类型：** blog
- **作者：** 自由度FreeDof（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/meBziOvU7pEjMqfOPLgaTw
- **发表日期：** 2026-09-30（页面 `ct` 时间戳）
- **入库日期：** 2026-09-30
- **抓取方式：** Cursor WebFetch（本环境未预装 `wechat-article-for-ai`）
- **系列：** 本文为「原始 OpenVLA 链路」篇；文内预告姊妹篇《OpenVLA 之后，VLA 改了什么？》对照 OFT、π₀、CogACT、GR00T、Helix 等
- **一句话说明：** 以可核对的开源实现走通 **单帧图像 + 指令 → Prismatic 主干 → 7 维动作 token 自回归 → 反归一化 → 下游 IK/控制**；强调 **模型边界 vs 机器人侧**、训练阶段表与复现坑（分箱/unnorm_key）。
- **步骤 2.5（开源核查）：** [openvla.github.io](https://openvla.github.io/) 与 [openvla/openvla](https://github.com/openvla/openvla) **已开源**（权重、微调、Bridge/WidowX 评测脚本）；与既有 [sources/repos/openvla.md](../repos/openvla.md) 结论一致。

## 核心摘录（归纳，非全文）

### 端到端因果链（文内主线）

> 单帧 RGB + 语言指令 → DINOv2 ∥ SigLIP → 按位置拼接 → Projector(MLP) → Llama 2（视觉 token 插在 BOS 后）→ 同一步 7 个动作分量各自离散 token、自回归生成 → 解码 + `unnorm_key` 反归一化 → **OpenVLA 预测结束** → 机器人侧构造末端目标、IK、关节跟踪 → 刷新观测再推理。

### 模型 vs 控制器边界

| 对象 | 原始 OpenVLA 是否作为 NN 输入/输出 |
|------|-------------------------------------|
| 单帧主相机 RGB | ✓ 输入 |
| 任务语言 | ✓ 输入 |
| 历史帧 / 显式关节·夹爪状态 | ✗ 不在默认接口（控制器可读本体，≠ 进网络） |
| 7D 末端增量 + 夹爪（一步） | ✓ 输出（经 token 与反归一化） |
| IK、碰撞、限速 | ✗ 下游控制 |

**遮挡夹爪：** 网络仍会输出动作；若单帧无法区分对称歧义（夹爪在杯左/右），缺历史与本体时 **无法可靠消歧** — 属观测设计局限，非「看不见就不输出」。

### 7 维单步动作（Bridge 类接口）

| 分量 | 含义 |
|------|------|
| 前 3 | 末端位置增量（坐标系依数据集） |
| 中 3 | 末端旋转增量 |
| 最后 1 | 夹爪命令 |

**不是** 七个关节角，也 **不是** 七个未来时刻；是自回归 **同一时刻** 的七个标量编码。

### 训练阶段表（文内）

| 阶段 | 数据/能力 |
|------|-----------|
| DINOv2 / SigLIP / Llama 各自预训练 | 视觉、图文、文本续写 |
| Prismatic VLM | 图文生成文本 |
| 通用 OpenVLA | ~97 万轨迹示教 → 动作 token |
| 目标任务微调 | 新机器人/相机/任务 LoRA 等 |
| 推理 | 参数冻结；执行一步 **不** 自动改权重 |

**教师强制：** 训练时用 **正确** 动作前缀预测下一 token；推理用 **自生成** 前缀 — 前分量错误可拖累后分量。

### 复现细节（文内强调对照代码）

- **分箱：** 论文写 256 档；公开实现用 256 边界 → **255 区间中心** + 末端索引裁剪 — 复现勿自行改分箱。
- **归一化：** 多数分量用 **第 1 / 第 99 百分位** 映射到 [−1, 1]；推理 **`unnorm_key`** 必须与权重、动作定义配套。
- **Projector（双塔 OpenVLA-7B）：** Linear(2176→8704) → GELU → Linear(8704→4096) → GELU → Linear(4096→4096)。
- **取层：** DINOv2 **第 23** 块、SigLIP **第 26** 块 patch 特征（Prismatic/LLaVA 倒数第二层约定）。
- **速度：** 论文口径 ~**6 次完整推理/秒**（单卡 RTX 4090，无 compile）— 模型侧频率 ≠ 机器人控制频率。

### Bridge/WidowX 部署路径

OpenVLA 末端增量 → 末端目标 → **机器人侧 IK** → 关节目标 + 夹爪；IK **不在** token 语义内。评测循环：**刷新观测 → 生成完整一步 → 调执行接口**（可非阻塞）。

### 文内局限归纳

- 单帧 + 无本体 → 遮挡/对称歧义
- 离散分箱 → 精度上限；7 token 串行 → 延迟
- 长时程衔接依赖 **控制循环与执行接口**，非模型一次输出整段轨迹

## 对 wiki 的映射

- **不新建论文实体** — 交叉补强 [paper-openvla](../../wiki/entities/paper-openvla.md)「推理与执行边界」与 [openvla 软件实体](../../wiki/entities/openvla.md)
- [VLA 方法页](../../wiki/methods/vla.md)、[VLA 演进谱系](../../wiki/overview/vla-evolution-lineage.md)
- 仓库归档：[openvla.md](../repos/openvla.md)；论文 source：[openvla_arxiv_2406_09246.md](../papers/openvla_arxiv_2406_09246.md)

## 参考来源（原文脚注论文，此处仅索引）

- OpenVLA arXiv:2406.09246；Prismatic arXiv:2402.07865；DINOv2；SigLIP — 详见文内列表
