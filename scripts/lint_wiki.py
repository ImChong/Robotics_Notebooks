#!/usr/bin/env python3
"""
lint_wiki.py — 自动化 wiki 健康检查脚本
基于 Karpathy LLM Wiki 模式，检测以下问题：
  1. 孤儿页（无其他 wiki 页面链接到它）
  2. 缺少"关联页面"或"关联"区块的页面
  3. 缺少"参考来源"区块的页面
  4. 缺少"英文缩写速查"区块的页面（信息型 backlog；新建/大幅改写页须补齐）
  4b. 英文缩写速查区块位置错误（须在「一句话定义」之后、「为什么重要」之前）
  5. 内链断链（链接目标文件不存在）
  6. 空壳页面（内容过少，< 200 字）
  7. Sources 孤儿（sources/papers 中链接到不存在 wiki 页）
  8. 陈旧页面（sources 文件比对应 wiki 页新，需 review）
  9. 矛盾检测（同一概念在不同页面有相反描述）
 10. Frontmatter 缺少 type 字段（V8 新增）
 11. 知识库活跃度检查（V8/V32：优先 wiki/roadmap 近 30 天 git 提交；git 不可用时回退 log.md）
 12. concepts/methods/tasks 缺少 summary/description 字段（V10 新增）
 13. formalizations/ 公式变量在正文是否有物理含义解释（V21 新增）
 14. 高频引用的 methods/ 缺少 queries/ 操作指南或 comparisons/ 对比页（V22 新增，信息型）
 15. wiki/entities/paper-* 元数据基线（V23/V31 新增，信息型）：
     - frontmatter 至少含 arxiv/venue/code 三类来源键之一；
     - 正文至少含「方法栈 / 评测 / 与其他工作对比」三段式；
     - 正文须含 ``## 结论``（后续 ingest 必做；历史 backlog 信息型）。
 16. 陈旧声明巡检（V24 新增，信息型）：正文含「SOTA / state-of-the-art /
     当前最强 / 最新」等绝对化措辞，但 frontmatter updated 早于库内共享 tag
     的更晚页面时，提示复核断言时效性。
 17. 缺页概念巡检（V24 新增，信息型）：正文以 **加粗** 或 `反引号` 形式被多页
     高频引用、但库内无独立 concepts/methods/formalizations 页的术语，输出
     "建议新建页"候选清单，作为后续 ingest/query 选题入口。

用法：
  python3 scripts/lint_wiki.py
  python3 scripts/lint_wiki.py --write-log   # 同时将报告插入 log.md 顶部
  python3 scripts/lint_wiki.py --report      # 保存 markdown 报告到 exports/lint-report.md
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
CANONICAL_FACTS_FILE = REPO_ROOT / "schema" / "canonical-facts.json"

# methods/ 页面入链数超过该阈值即视为"高频引用"，需有对应 queries/comparisons 操作落地
METHOD_PRACTITIONER_INBOUND_THRESHOLD = 3

# 绝对化/时效性措辞（stale claim 巡检 V1）：正文出现这些词，但该页 frontmatter
# updated 早于库内共享 tag 的更晚页面时，给出 INFO 级提示，提示复核断言时效性。
STALE_CLAIM_PATTERNS = [
    r"\bSOTA\b",
    r"state[- ]of[- ]the[- ]art",
    r"当前最强",
    r"目前最强",
    r"当前最佳",
    r"目前最佳",
    r"业界最强",
    r"迄今最强",
    r"最新",
]

# 陈旧声明巡检的误报豁免：命中绝对化措辞不等于本页在下时效性断言。以下五类是
# 结构性误报，按命中处的上下文豁免，避免为迁就正则去改写本就正确的正文：
#   1) 否定语境：「这是部署证据，不是策略 SoTA」「不要把它读成又一个 SoTA」
#      「这一行不可直接当 SOTA 通才」等辟谣式写法，本身就在否认该断言；否认也
#      可落在动词上（「0.1–0.2 pp 不足以支撑『全面 SOTA』叙事」「WM 不负责单独
#      变成 SOTA VLA」），与「不是 / 不可」同为「不 + 谓词」的辟谣；中文里
#      辟谣也可后置（「0.066 vs 0.067，读『SOTA 碾压』会过读」），落笔顺序相反、
#      语义同为否认，故前置线索回看、后置线索前看，都限制在命中词同句内；
#   2) 库内页面名：「VLA SOTA Leaderboard」是 entities/vla-sota-leaderboard.md 的
#      页面标题，正文引用它属导航，不是本页断言；
#   3) 运行时对象：「服务端只保留最新 pending 帧」「取最新状态」「只在最新候选
#      过门后刷新」描述系统行为，不会随领域进展过时；写成行内公式的运行时量
#      （「内层根据最新 \(\mathbf{q}\) 生成指令」）同属此类，只是用符号而非名词落笔；
#   4) 英文缩写速查区块：`| SOTA | State of the Art | 排行榜对照参考 |` 是词条
#      释义表（写作规范要求的固定区块），在给缩写下定义而非给结论下断言，
#      与「常见误区」区块同属结构性区块，扫描前整段剥离；
#   5) 对照基线的类别名：「相对经典/SOTA 基线，证书平均收窄 51.6%」里的 SOTA 指
#      被比较的那批既有方法，是对照组标签而非本页对自身的断言；与 2) 的页面名
#      引用同属指称，故按命中词后紧跟的「基线 / baseline」豁免。
STALE_CLAIM_NEGATION_CUES: tuple[str, ...] = (
    "不是",
    "并非",
    "不算",
    "称不上",
    "不要",
    "而非",
    "不应",
    "不可",
    "不足以",  # 「0.1–0.2 pp 不足以支撑『全面 SOTA』叙事」：否认落在谓词上
    "不负责",  # 「WM 不负责单独变成 SOTA VLA」：同为「不 + 谓词」的辟谣
    "未必",
    "勿",
)
# 命中词前回看的字符数（并限制在同一句内：遇句末标点/换行即截断）
STALE_CLAIM_NEGATION_WINDOW = 30
STALE_CLAIM_SENTENCE_BREAKS = "。！？；\n"
# 后置否定线索：把「读成 X」的读法判为过度解读，与前置的「不是 X」同为辟谣
STALE_CLAIM_OVERREAD_CUES: tuple[str, ...] = (
    "过读",
    "过度解读",
    "读过头",
)
# 命中词后前看的字符数（同样限制在同一句内）
STALE_CLAIM_OVERREAD_WINDOW = 30
# 「最新」后紧跟的运行时对象名词；中间允许夹一段英文/数字标识（如「最新 pending 帧」）
STALE_CLAIM_RUNTIME_OBJECT_RE = re.compile(
    r"[\sA-Za-z0-9_./-]{0,16}(?:帧|状态|观测|位姿|数据|快照|消息|指令|读数|候选)"
)
# 「SOTA / state-of-the-art + 基线」是对照组的类别名（「相对经典/SOTA 基线，证书
# 平均收窄 51.6%」），指的是被比较的那批既有方法，不是本页对自身的时效性断言；
# 与「库内页面名引用」同属导航/指称而非结论，故按命中词后**紧跟**的
# 「基线 / baseline」豁免；只允许夹一两个空白，避免把跨短语的命中一并放行。
STALE_CLAIM_BASELINE_LABEL_RE = re.compile(r"\s{0,2}(?:基线|baselines?)", re.I)
# 「最新」后紧跟的行内公式运行时量（如「最新 \((\mathbf{q},\mathbf{e})\)」）：与
# 「最新状态/读数」同为运行时对象，只是把量写成符号而非名词，同样不随领域进展过时。
STALE_CLAIM_RUNTIME_MATH_RE = re.compile(r"\s{0,2}\\\([^\n]{0,80}?\\\)")
# 命中词所在的「英文/数字/空格/连字符」连续片段，用于还原被引用的页面标题
STALE_CLAIM_SPAN_CHAR_RE = re.compile(r"[A-Za-z0-9 -]")

# 缺页概念巡检 V1：正文以 **加粗** 或 `反引号` 形式被多页引用、但库内无独立
# concepts/methods/formalizations 页的术语，给出"建议新建页"候选（INFO 级）。
# 单 token 词形（可含连字符 / 加号），用于排除路径、文件名、整句等噪声。
MISSING_CONCEPT_TERM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+\-]{1,30}$")
# 被至少这么多不同页面引用才视为"高频"，避免低频术语刷屏。
MISSING_CONCEPT_PAGE_MIN_PAGES = 6
# 候选输出上限，避免淹没健康报告。
MISSING_CONCEPT_PAGE_MAX_CANDIDATES = 15
# 明显非概念的高频 token（frontmatter 键 / 布尔值等），不计入候选。
MISSING_CONCEPT_STOPWORDS: set[str] = {
    "type",
    "tags",
    "related",
    "sources",
    "updated",
    "summary",
    "description",
    "true",
    "false",
    "null",
    "id",
    "title",
    # code：各页正文里的 `code` 均为 frontmatter 来源键引用（「官方入口见
    # frontmatter `code` / 项目页」「配套 SDK 入口见 frontmatter `code` 字段」），
    # 少数指项目页上的 **Code** 按钮，同为仓库入口指针，非机器人概念/方法/
    # 形式化，不应建独立页；与 type/tags/sources 同类 frontmatter 键停用词。
    "code",
    "wiki",
    "md",
    "http",
    "https",
    "main",
    # venue：各页正文里的 `Venue` / **Venue** 均为 paper-* 实体页「核心信息」表的
    # 发表信息行标签（`| **Venue** | ECCV 2026 |`），与 frontmatter 的 venue 来源键
    # 同义（见 _check_paper_entity_metadata 的 source_keys），是出版元数据字段名，
    # 非机器人概念/方法/形式化，不应建独立页；与 arxiv/code/type/tags 同类停用词。
    "venue",
    # clip：正文里既指 CLIP 视觉-语言模型，又指力矩「限幅」动词（torque clip），
    # 两义被小写 slug 合并，非单一可成页概念，作停用词不再误报为「缺独立页」。
    "clip",
    # arxiv：预印本托管/出版平台（基础设施），非机器人概念/方法/形式化，
    # 不应建独立 concepts/methods 页；与 http/https/main 同类基础设施停用词。
    "arxiv",
    # license：各页正文里的 `LICENSE` 均为仓库许可证文件名 / SPDX 口径引用
    # （如「仓库 `LICENSE` 文件」「以 LICENSE 为准」），是法务/文件名 token，
    # 非机器人概念/方法/形式化，不应建独立页；与 md/arxiv 同类基础设施停用词。
    "license",
    # train：正文里的 `train` / **train** 均为动词「训练」或 `train/eval` 数据
    # 划分口径（如「先 train 再 eval」「train split」），非可成页的机器人概念，
    # 与 clip（限幅动词）同类语义噪声，作停用词不再误报为「缺独立页」。
    "train",
    # uv：各页正文里的 `uv` 均为 Astral 的 Python 包管理器命令（`uv run` /
    # `uv sync` / `uv pip install`，复现路径工具链引用），是打包/环境基础设施，
    # 非机器人概念/方法/形式化，不应建独立页；与 arxiv/license/md 同类基础设施停用词。
    "uv",
    # sequencediagram：各页正文里的 `sequenceDiagram` 均为 Mermaid 图类型关键字
    # （论文实体页「源码运行时序图」章节的写作约定，如「代码发布后应补
    # `sequenceDiagram`」），是文档语法 token，非机器人概念/方法/形式化，
    # 不应建独立页；与 md/http 同类文档基础设施停用词。
    "sequencediagram",
    # printf：各页正文里的 `printf` / `fprintf` 均为 C 标准库打印调用，作为
    # 「实时控制环里禁做阻塞 I/O」的反例出现（UART bring-up 调试、EtherCAT
    # 循环延迟累计等），是语言/工具链 token，非机器人概念/方法/形式化，
    # 不应建独立页；与 uv/md 同类基础设施停用词。
    "printf",
    # stop：各页正文里的 `STOP` / `stop` 是运行时命令名或枚举值，三义被小写
    # slug 合并——BehaviorTree/Groot 控制命令（`STOP`→复位→`RESUME`，其语义已在
    # concepts/behavior-tree-vla-orchestration.md 的命令表逐条释义）、agent 自
    # 校正循环的动作枚举（img2threejs 的 `continue`/`refine-code`/`stop`）、
    # 导航评测的片段类别（Green for Go 的 stop 片段）。非单一可成页概念，
    # 与 clip（模型名 vs 限幅动词）同类语义噪声。
    "stop",
}

# 高频术语但「已在 entities/ 或非同名 stem 的 methods 页有恰当归属」，
# 不应再按裸 token 误报为「缺独立 concepts/methods 页」。映射到既有页：
#   amp        → methods/amp-reward.md + overview/humanoid-amp-motion-prior-survey.md
#   armature   → concepts/armature-modeling.md（电枢惯量建模，概念页）
#   damping    → concepts/impedance-control.md（质量-弹簧-阻尼里阻尼矩阵 B_d 的物理
#                含义与临界阻尼取法）+ queries/legged-humanoid-rl-pd-gain-setting.md
#                （legged_gym / Isaac Lab 的 `stiffness`/`damping` 怎么设）+
#                methods/joint-actuator-parameter-identification.md（辨识值写回
#                MuJoCo `armature`/`damping`/`frictionloss` 三件套）：与 armature 同为
#                MuJoCo/Isaac Lab 关节属性 token，本库一律在阻抗/PD 增益/辨识页按
#                参数维度记述，不单建概念页
#   g1         → entities/unitree-g1.md（硬件，归 entities）
#   gmr        → methods/motion-retargeting-gmr.md（General Motion Retargeting，方法页）
#   heracles   → entities/paper-heracles-humanoid-diffusion.md（具体系统）
#   joint      → 三义各有归属，均非同名 stem：URDF/MuJoCo 的关节元素归
#                concepts/urdf-robot-description.md + concepts/armature-modeling.md
#                （`joint` 上的 armature/反射惯量）；WAM 分类学的 **Joint** 族
#                （联合预测未来与动作，对 **Cascaded**）归
#                concepts/world-action-models.md；论文消融条件名「joint」
#                （Green for Go）属单页实验设定。与 damping 同为 MuJoCo 关节属性
#                token，本库按参数/分类维度记述，不单建概念页
#   lcm        → concepts/lcm-basics.md（Lightweight Communications and Marshalling
#                的 canonical 概念页：UDP 组播 pub/sub、类型描述语言与多语言强类型
#                序列化，slug 与页面 stem 不同名）+ concepts/ipc-inter-process-communication.md
#                （IPC 谱系里的一档）+ comparisons/ros2-vs-lcm.md（与 ROS 2 的选型
#                对照）：与 ethercat / ros2 / urdf 同属「缩写 slug ≠ 页面 stem」，
#                不应按裸缩写误报为缺页
#   lerobot    → entities/lerobot.md（Hugging Face 具身智能全栈框架，库/工具）
#   mjlab      → entities/mjlab.md（库/工具）
#   mujoco     → entities/mujoco.md（仿真器/工具）
#   sonic      → methods/sonic-motion-tracking.md（具体方法）
#   locomotion → tasks/locomotion.md（+ humanoid-/hybrid-locomotion，任务域，归 tasks）
#   loco-manipulation → tasks/loco-manipulation.md（移动操作，任务域，归 tasks）
#   step       → entities/step2urdf.md（CAD 交换格式，已在 concepts/text-to-cad.md 引介）
#   twist2     → entities/paper-twist2.md（具体系统/论文，归 entities）
#   ethercat   → concepts/ethercat-protocol.md（实时总线概念，slug 与页面 stem 不同名）
#   urdf       → concepts/urdf-robot-description.md（统一机器人描述格式，slug 与
#                页面 stem 不同名；该页已含定义/缩写速查/n_q vs n_v 等完整内容）
#   mit        → schema/institutions.json 机构（麻省理工），归属机构注册表，不建概念页
#   vlm        → comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md（VLM/VLN/VLA/
#                VLX/WM 五类的 canonical 定义与 I/O 边界，唯一 VLM 家族来源
#                sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md 即
#                入库于此）+ concepts/vision-language-feature-fusion.md（跨模态
#                融合机制）+ methods/vla.md（按「底座」维度记述）+
#                overview/hub-embodied-foundation-model.md（选型闭环枢纽）：
#                模型家族标签而非独立可成页机制，另建 concepts/vlm.md 只会与
#                分类学对比页重复同一来源
#   qwen3-vl   → methods/vla.md（VLA 底座列表）+ methods/star-vla.md +
#                concepts/foundation-policy.md：外部通用 VLM 产品型号，与
#                DINOv2 / SigLIP / Wan2.2 / Qwen2.5-VL 同类，本库一律在
#                方法/概念页按「底座」维度记述，不单建概念页
#   qpos       → formalizations/articulated-body-algorithms.md（$q \in R^{n_q}$
#                广义坐标定义）+ formalizations/motion-retargeting-objective.md
#                （重定向目标里的机器人广义坐标 $\mathbf{q}^r_t$）+
#                concepts/urdf-robot-description.md（$n_q$ vs $n_v$ 维数差）+
#                concepts/floating-base-dynamics.md（浮动基广义位置/速度）：
#                MuJoCo 状态数组的字段名，本体是已建页的广义坐标 $q$，与
#                armature/damping 同为 MuJoCo 属性 token，本库按参数/形式化维度
#                记述，不单建概念页
#   rl         → methods/reinforcement-learning.md（canonical 定义页，缩写 slug
#                与页面 stem 不同名）+ methods/safe-rl.md / model-based-rl.md /
#                value-based-reinforcement-learning.md /
#                hierarchical-reinforcement-learning.md（子族）+
#                comparisons/rl-vs-il.md / mpc-vs-rl.md / wbc-vs-rl.md（选型对照）：
#                与 wbc / wam / urdf 同属「缩写 slug ≠ 页面 stem」，不应按裸缩写
#                误报为缺页
#   reset      → entities/gymnasium.md（`reset` / `step` 标准方法集合、四入口节点图、
#                「先 step 再 reset 会被 OrderEnforcing 拦下」误区条）+
#                entities/xpolicylab.md（策略侧 `reset` 清 episode 状态）：
#                环境/策略 API 的方法名（episode 复位），与 stop（运行时命令名）
#                同类 token，语义已在 API 表逐条释义，不单建概念页
#   rgb-d      → concepts/embodied-perception-six-spatial-representations.md
#                （缩写速查 + 「深度」层的 RGB-D SLAM / 尺度对齐口径）+
#                formalizations/3d-coordinate-transforms-vision-robotics.md
#                （深度获取路径表：结构光 / ToF 直接测距，及反投影链路）+
#                concepts/2d-to-3d-semantic-lifting-gap.md（RGB-D 补尺度路线）：
#                传感模态标签而非独立可成页机制，与 vlm 同类，另建概念页只会与
#                上述表征/几何页重复同一来源
#   ros2       → concepts/ros2-basics.md（canonical 定义页：节点/话题/服务/DDS 分层，
#                slug 与页面 stem 不同名）+ concepts/rmw-interface.md（中间件抽象层）
#                + concepts/dds-communication.md + comparisons/ros2-vs-lcm.md
#                （选型对照）+ entities/navigation2.md / unitree-ros2.md（发行版与
#                厂商包）：与 wbc / urdf / rl 同属「slug ≠ 页面 stem」，不应按裸名
#                误报为缺页
#   action     → formalizations/mdp.md（动作空间 $A$ 的 canonical 定义）+
#                methods/action-chunking.md（多步动作序列输出）+
#                concepts/world-action-models.md（动作后果预测）+
#                methods/uhas-unified-hand-action-space.md（统一手部动作空间）：
#                命中处全是数据字段名 / API 键名（RobotWin JSONL 的 `action` 帧类型、
#                UMI 导出的 20 维 `action`、ROS pick/place **action** 接口、演示数据
#                格式清单里的 `action` 列），与 qpos / reset / step 同为 token 而非
#                机制，本库按形式化/方法维度记述，不单建概念页
#   sim-to-real → concepts/sim2real.md（canonical 定义页，全称写法与页面 stem
#                `sim2real` 不同名）+ concepts/physics-fidelity-sim2real-gap.md
#                （保真度缺口）+ concepts/processor-in-the-loop-sim2real.md
#                （PIL 环）+ overview/paper-notebook-category-10-sim-to-real.md
#                （论文笔记分类）：与 ethercat / urdf / wbc 同属「slug ≠ 页面
#                stem」，同一概念的连字符全称写法，不应按裸名误报为缺页
#   zero-shot  → concepts/sim2real.md（zero-shot transfer 是该页主线口径，
#                全页 27 处）+ overview/hub-cross-embodiment.md（跨具身零样本
#                迁移枢纽）+ concepts/embodied-fm-latency-generalization-tradeoff.md
#                （泛化侧取舍）：迁移/评测的 **条件状语**（zero-shot 迁移、
#                zero-shot 泛化、0% 任务数据），与 rgb-d / vlm 同为描述性标签
#                而非独立可成页机制，另建概念页只会与 sim2real 重复同一来源
#   base       → 三义各有归属：URDF/MuJoCo 的基座连杆（TITA `base` 接触即
#                terminate、相机到 `base`/`head` 的 TF）归
#                concepts/urdf-robot-description.md + concepts/floating-base-dynamics.md
#                （固定基座 vs 浮动基座、\(q_b\) 位姿）；权重档名（EgoSteer
#                「自有数据微调用 **Base**」）属外部模型型号；消融/基线条件名
#                （HumanoidArena 的 **Base** 任务档、AMP survey 的 **Base** = 仅
#                Stage 1）属单页实验设定。与 joint 同为多义 token，本库按本体/
#                实验维度记述，不单建概念页
#   cartpole-v1 → concepts/cartpole.md（canonical 概念页，注册名写法与页面 stem
#                不同名）+ entities/gymnasium.md（注册表入口）：Gymnasium 的环境
#                注册 id，与 sim-to-real / urdf 同属「id 写法 ≠ 页面 stem」，
#                不应按裸 id 误报为缺页
#   isaac-cartpole-v0 → concepts/cartpole.md（同一 cart-pole 概念的 GPU 并行版，
#                该页「从 CPU 玩具跨到 GPU 训练栈」一节逐条记述）+
#                entities/isaac-lab-default-environments.md（默认环境清单）：
#                与 cartpole-v1 同为环境注册 id，不单建概念页
#   libero     → entities/libero-benchmark.md（130 个机械臂任务的终身学习/迁移
#                基准，slug 与页面 stem 不同名）：基准数据集归 entities，
#                与 lerobot / mjlab 同类，不应按裸名误报为缺 concepts 页
#   libero-plus → entities/libero-benchmark.md「扰动增强变体：LIBERO-Plus」专节
#                （相机/布局/语言/噪声/纹理扰动条件、分项读法与跨页数字不可横比
#                的口径，并交叉链接 LAWA / StellaVLA / GaussianDream++ / Kairos /
#                Rift / Flex-π / SLIM-0.5B 各自报告值）：它是同一批 LIBERO 任务的
#                扰动条件而非新机制，与 libero 同归该基准实体页，不单建概念页
#   onpolicyrunner → concepts/rl-runner.md（Runner 层的 canonical 概念页，
#                「rsl_rl 叫 `OnPolicyRunner`」即该页对号入座的锚点）+
#                entities/rsl-rl.md（该类的实现库）：具体实现的类名，本体是已建
#                页的 on-policy Runner 抽象，与 qpos / reset 同为代码 token 而非
#                机制，不单建概念页
MISSING_CONCEPT_COVERED_ELSEWHERE: set[str] = {
    "act",  # 已由 methods/action-chunking.md（机制）+ entities/paper-act.md（论文）覆盖，
    # 缩写 slug 与页面 stem 不同名；与 wbc / rl / wam 同类
    "action",
    "base",  # 基座连杆 / 权重档名 / 消融条件名三义，已由 URDF + 浮动基座等页覆盖
    "amp",
    "armature",
    "cartpole-v1",  # Gymnasium 环境注册 id，已由 concepts/cartpole.md 覆盖
    "damping",  # MuJoCo/Isaac Lab 关节属性，已由阻抗控制 + PD 增益 / 参数辨识页覆盖
    "ethercat",  # 已由 concepts/ethercat-protocol.md 覆盖（slug 与页面 stem 不同名）
    "g1",
    "gmr",
    "heracles",
    "isaac-cartpole-v0",  # Isaac Lab 环境注册 id，已由 concepts/cartpole.md 覆盖
    "joint",  # 关节属性 / WAM Joint 族 / 消融条件名三义，已由 URDF + WAM 等页覆盖
    "lcm",  # 已由 concepts/lcm-basics.md 覆盖（缩写 slug 与页面 stem 不同名）
    "lerobot",  # 已由 entities/lerobot.md 覆盖（框架/工具，与 mjlab / mujoco 同类）
    "libero",  # 已由 entities/libero-benchmark.md 覆盖（基准，slug 与页面 stem 不同名）
    "libero-plus",  # LIBERO 的扰动增强套件，已由 entities/libero-benchmark.md 专节覆盖
    "mit",  # 机构（schema/institutions.json），非概念，不应建 concepts/methods 页
    "mjlab",
    "mujoco",
    "onpolicyrunner",  # rsl_rl 的 Runner 类名，已由 concepts/rl-runner.md 覆盖
    "qpos",  # MuJoCo 状态数组字段名，已由广义坐标 $q$ 的形式化/概念页覆盖
    "qwen3-vl",  # 外部 VLM 底座型号，已在 methods/vla.md 等按「底座」维度记述
    "reset",  # 环境/策略 API 方法名（episode 复位），已由 entities/gymnasium.md 释义
    "rgb-d",  # 传感模态标签，已由六种空间表征 / 三维坐标变换等页覆盖
    "rl",  # 已由 methods/reinforcement-learning.md 覆盖（缩写 slug 与页面 stem 不同名）
    "sim-to-real",  # 已由 concepts/sim2real.md 覆盖（全称写法与页面 stem 不同名）
    "ros2",  # 已由 concepts/ros2-basics.md 覆盖（slug 与页面 stem 不同名）
    "sonic",
    "wbc",  # 已由 concepts/whole-body-control.md 覆盖（slug 与页面 stem 不同名）
    "wam",  # 已由 concepts/world-action-models.md 覆盖（缩写 slug 与页面 stem 不同名）
    "zero-shot",  # 迁移/评测的条件状语，已由 concepts/sim2real.md 等页覆盖
    "locomotion",
    "loco-manipulation",
    "step",
    "twist2",
    "urdf",  # 已由 concepts/urdf-robot-description.md 覆盖（slug 与页面 stem 不同名）
    "vlm",  # 模型家族标签，已由 comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md 定义
}

# 仅用于信息提示、不计入 lint 失败总数的检查 key
INFO_ONLY_KEYS: set[str] = {
    "missing_pages",
    "missing_concept_pages",
    "missing_abbrev_glossary",
    "methods_without_practitioner_query",
    "paper_missing_source_meta",
    "paper_missing_three_sections",
    "paper_missing_conclusions",
    "dataset_missing_metadata",
    "stale_claims",
    "physics_concept_crosslink",
    "contact_control_crosslink",
    "embodied_fm_crosslink",
    "eval_benchmark_crosslink",
    "actuator_drive_chain_crosslink",
    "perception_stack_crosslink",
}


def load_canonical_facts() -> dict:
    """加载 schema/canonical-facts.json 矛盾检测规则数据。

    历史上规则与 ~150 条事实数据写在 lint() 内联字典里，文件膨胀到 ~700 行
    数据 + ~300 行代码。外移到 JSON 后，新增/修改事实属于纯数据变更，
    不再需要改代码、过 ruff、动 imports。
    """
    if not CANONICAL_FACTS_FILE.exists():
        return {}
    return json.loads(CANONICAL_FACTS_FILE.read_text(encoding="utf-8"))


# 只扫描 wiki/ 下的 markdown 文件
def get_wiki_pages() -> list[Path]:
    return sorted(WIKI_DIR.rglob("*.md"))


# 移除代码块（``` ... ``` 和 ` ... `），避免提取代码示例中的假链接
def strip_code_blocks(content: str) -> str:
    # 移除围栏代码块
    content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
    # 移除行内代码
    content = re.sub(r"`[^`]+`", "", content)
    return content


# 从文件中提取所有内部链接目标（相对路径 .md 文件）
def extract_internal_links(content: str, source_path: Path) -> list[Path]:
    targets = []
    content = strip_code_blocks(content)
    # 匹配 markdown 链接 [text](path)，只取 .md 文件且不是 http
    for match in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", content):
        href = match.group(2).strip()
        if href.startswith("http") or href.startswith("#"):
            continue
        # 去掉锚点
        href = href.split("#")[0]
        if not href.endswith(".md"):
            continue
        resolved = (source_path.parent / href).resolve()
        targets.append(resolved)
    return targets


def has_section(content: str, patterns: list[str]) -> bool:
    """检查是否存在某个 ## 级别的区块（匹配关键词）"""
    for pat in patterns:
        if re.search(rf"^##\s+.*{pat}", content, re.MULTILINE | re.IGNORECASE):
            return True
    return False


def word_count(content: str) -> int:
    """简单估算字数（中英文混合）"""
    # 中文字符 + 英文单词
    chinese = len(re.findall(r"[\u4e00-\u9fff]", content))
    english = len(re.findall(r"\b[a-zA-Z]+\b", content))
    return chinese + english


def has_source_reference(content: str) -> bool:
    """检查页面是否引用了 sources/ 下的原始资料。

    覆盖率统计关心的是 wiki 页是否能追溯到原始资料，而不是资料是否一定来自
    sources/papers/。repo、blog、note 等来源同样是有效 ingest 来源。
    """
    return bool(re.search(r"(?:\.\./)*sources/[^)\s]+\.md\b", content))


def strip_misconception_sections(content: str) -> str:
    """移除"常见误区/误区"区块，避免把辟谣内容误判为事实矛盾。"""
    lines = content.splitlines()
    kept = []
    skip_level = None
    heading_re = re.compile(r"^(#{2,6})\s+(.*)$")
    for line in lines:
        m = heading_re.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip().lower()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if any(key in title for key in ["常见误区", "误区", "misconception", "pitfall"]):
                skip_level = level
                continue
        if skip_level is None:
            kept.append(line)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# lint() 子检查函数
# ---------------------------------------------------------------------------


def _build_link_index(
    pages: list[Path], page_set: set[Path]
) -> tuple[dict[Path, list[Path]], dict[Path, list[str]]]:
    """建立每个页面被哪些其他页面链接的索引，并收集断链。"""
    inbound: dict[Path, list[Path]] = {p.resolve(): [] for p in pages}
    broken_links: dict[Path, list[str]] = {}

    for page in pages:
        content = page.read_text(encoding="utf-8")
        links = extract_internal_links(content, page)
        for target in links:
            if target in page_set:
                inbound[target].append(page.resolve())
            elif not target.exists():
                broken_links.setdefault(page.resolve(), []).append(
                    str(target.relative_to(REPO_ROOT))
                    if target.is_relative_to(REPO_ROOT)
                    else str(target)
                )
    return inbound, broken_links


def _empty_results() -> dict[str, Any]:
    """初始化空的检查结果字典。"""
    return {
        "orphan_pages": [],
        "missing_related": [],
        "missing_sources": [],
        "missing_abbrev_glossary": [],
        "abbrev_glossary_wrong_order": [],
        "broken_links": [],
        "stub_pages": [],
        "missing_pages": [],
        "missing_concept_pages": [],
        "broken_source_refs": [],
        "sources_orphans": [],
        "stale_pages": [],
        "outdated_pages": [],
        "contradictions": [],
        "missing_type": [],
        "log_inactive": [],
        "missing_summary": [],
        "query_format": [],
        "formalization_no_formula": [],
        "formalization_unexplained_vars": [],
        "readme_badge": [],
        "orphan_count": [],
        "method_missing_link": [],
        "method_missing_sections": [],
        "entity_missing_outgoing": [],
        "tool_missing_institution": [],
        "wikilink_syntax": [],
        "unclosed_autolinks": [],
        "methods_without_practitioner_query": [],
        "paper_missing_source_meta": [],
        "duplicate_arxiv": [],
        "paper_missing_three_sections": [],
        "paper_missing_conclusions": [],
        "dataset_missing_metadata": [],
        "stale_claims": [],
        "physics_concept_crosslink": [],
        "contact_control_crosslink": [],
        "embodied_fm_crosslink": [],
        "eval_benchmark_crosslink": [],
        "actuator_drive_chain_crosslink": [],
        "perception_stack_crosslink": [],
        "_ingest_covered": 0,
        "_ingest_total": 0,
    }


def _check_per_page(
    pages: list[Path],
    inbound: dict[Path, list[Path]],
    broken: dict[Path, list[str]],
    results: dict[str, Any],
) -> None:
    """逐页检查：孤儿、关联区块、参考来源、断链、空壳、wikilink 语法、source 引用。"""
    for page in pages:
        resolved = page.resolve()
        content = page.read_text(encoding="utf-8")
        rel = page.relative_to(REPO_ROOT)

        if page.name.lower() not in ("readme.md", "index.md"):
            if not inbound.get(resolved):
                results["orphan_pages"].append(str(rel))

        is_meta_page = (
            page.name.lower() in ("readme.md", "index.md")
            or "references/" in str(rel)
            or "roadmaps/" in str(rel)
        )
        if not is_meta_page and not has_section(content, ["关联", "related", "已有页面", "关系"]):
            results["missing_related"].append(str(rel))

        is_meta_sources = page.name.lower() in ("readme.md", "index.md") or "references/" in str(
            rel
        )
        if not is_meta_sources and not has_section(content, ["参考来源", "sources", "参考"]):
            results["missing_sources"].append(str(rel))

        is_meta_abbrev = page.name.lower() in ("readme.md", "index.md") or any(
            seg in str(rel) for seg in ("references/", "roadmaps/")
        )
        if not is_meta_abbrev and not has_section(
            content, ["英文缩写速查", "abbreviation glossary", "abbreviations"]
        ):
            results["missing_abbrev_glossary"].append(str(rel))

        if not is_meta_abbrev:
            from wiki_abbrev_section import is_abbrev_glossary_well_placed

            if has_section(
                content, ["英文缩写速查", "abbreviation glossary", "abbreviations"]
            ) and not is_abbrev_glossary_well_placed(content):
                results["abbrev_glossary_wrong_order"].append(str(rel))

        if resolved in broken:
            for b in broken[resolved]:
                results["broken_links"].append(f"{rel} → {b}")

        if word_count(content) < 200:
            results["stub_pages"].append(f"{rel} ({word_count(content)} 字)")

        if not is_meta_page:
            results["_ingest_total"] += 1
            if has_source_reference(content):
                results["_ingest_covered"] += 1

        wl_stripped = strip_code_blocks(content)
        for m in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", wl_stripped):
            token = m.group(1).strip()
            if re.match(r"^[\d\.\-\s,]+$", token):
                continue
            results["wikilink_syntax"].append(f"{rel}: {m.group(0)}")

        for line_no, line in enumerate(wl_stripped.splitlines(), start=1):
            if re.search(r"<https?://[^\s>]+(?=\s*\||\s*$)", line):
                results["unclosed_autolinks"].append(f"{rel}:{line_no}: {line.strip()}")

        stripped = strip_code_blocks(content)
        for m in re.finditer(r"\[([^\]]*)\]\(([^)]+sources/[^)]+\.md)[^)]*\)", stripped):
            href = m.group(2).split("#")[0]
            resolved_src = (page.parent / href).resolve()
            if not resolved_src.exists():
                results["broken_source_refs"].append(f"{rel} → {href}")


def _check_missing_concepts(pages: list[Path], results: dict[str, Any]) -> None:
    """全局扫描提及但缺少对应 wiki 页面的技术概念。"""
    existing_stems = {p.stem.lower() for p in pages}
    watch_terms = {
        "MPPI": "model-based-rl",
        "DMP": "dmp",
        "GAE": "gae",
        "HER": "her",
        "POMDP": "pomdp",
        "Pontryagin": "optimal-control",
        "DDPG": "policy-optimization",
        "MARL": "marl",
        "ContactNet": "contact-net",
    }
    all_content = "".join(page.read_text(encoding="utf-8") for page in pages)
    term_counts: dict[str, int] = {}
    for term, slug in watch_terms.items():
        count = len(re.findall(rf"\b{re.escape(term)}\b", all_content))
        if count >= 2 and slug not in existing_stems:
            term_counts[term] = count
    for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        results["missing_pages"].append(
            f"{term} （出现 {count} 次，建议新建 wiki/{watch_terms[term]}.md）"
        )


def _check_missing_concept_pages(pages: list[Path], results: dict[str, Any]) -> None:
    """缺页概念巡检 V1（信息型，不阻塞 CI）。

    统计正文中以 ``**加粗**`` 或 `` `反引号` `` 形式高频出现（被
    ``MISSING_CONCEPT_PAGE_MIN_PAGES`` 个以上不同页面引用），但库内没有独立
    concepts/methods/formalizations 页的术语，输出"建议新建页"候选清单，
    作为后续 ingest/query 选题入口。与 ``_check_missing_concepts``（人工 watch
    列表、负责把已知术语映射到既有 slug）互补：本检查是全库自动统计。
    """
    # 已有独立页 stem（仅 concepts/methods/formalizations 视为"已建页"）
    covered_stems: set[str] = set()
    for page in pages:
        rel = page.relative_to(REPO_ROOT) if page.is_relative_to(REPO_ROOT) else page
        parts = rel.parts
        if "wiki" in parts:
            idx = parts.index("wiki")
            if len(parts) > idx + 1 and parts[idx + 1] in {
                "concepts",
                "methods",
                "formalizations",
            }:
                covered_stems.add(page.stem.lower())

    # 术语 slug → 引用它的页面集合；slug → 首见展示形（保留原大小写）
    term_pages: dict[str, set[str]] = {}
    term_display: dict[str, str] = {}
    for page in pages:
        content = page.read_text(encoding="utf-8")
        body = re.sub(r"^---\n.*?\n---", "", content, flags=re.DOTALL)
        body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)  # 仅去围栏块，保留行内反引号
        rel_str = str(page.relative_to(REPO_ROOT)) if page.is_relative_to(REPO_ROOT) else str(page)
        terms_in_page: set[str] = set()
        for m in re.finditer(r"\*\*([^*\n]+)\*\*", body):
            terms_in_page.add(m.group(1).strip())
        for m in re.finditer(r"`([^`\n]+)`", body):
            terms_in_page.add(m.group(1).strip())
        for term in terms_in_page:
            if not MISSING_CONCEPT_TERM_RE.match(term):
                continue
            slug = term.lower()
            if (
                slug in MISSING_CONCEPT_STOPWORDS
                or slug in MISSING_CONCEPT_COVERED_ELSEWHERE
                or slug in covered_stems
            ):
                continue
            term_display.setdefault(slug, term)
            term_pages.setdefault(slug, set()).add(rel_str)

    candidates = [
        (slug, len(refs))
        for slug, refs in term_pages.items()
        if len(refs) >= MISSING_CONCEPT_PAGE_MIN_PAGES
    ]
    candidates.sort(key=lambda x: (-x[1], x[0]))
    for slug, count in candidates[:MISSING_CONCEPT_PAGE_MAX_CANDIDATES]:
        results["missing_concept_pages"].append(
            f"{term_display[slug]}（被 {count} 个页面以加粗/反引号引用，"
            f"但无独立 concepts/methods/formalizations 页，建议评估新建）"
        )


def _build_git_mtime_map() -> dict[Path, float]:
    """Return {absolute Path → unix ts of most recent commit touching it}.

    Uses a single `git log --name-only` pass over the entire history so it
    runs in <1s on typical repos. The filesystem `mtime` is unreliable in
    fresh clones (e.g. cloud Agent containers) where every checked-out file
    has the same `mtime` regardless of when its content actually changed.
    Returns an empty map if the repo isn't a git checkout or git fails.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "log",
                "--format=__COMMIT__:%ct",
                "--name-only",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            return {}
    except (subprocess.SubprocessError, OSError):
        return {}

    mtime_map: dict[Path, float] = {}
    current_ts: float | None = None
    for line in result.stdout.splitlines():
        if line.startswith("__COMMIT__:"):
            try:
                current_ts = float(line.removeprefix("__COMMIT__:"))
            except ValueError:
                current_ts = None
        elif line and current_ts is not None:
            p = (REPO_ROOT / line).resolve()
            # git log is reverse-chronological → first occurrence wins
            if p not in mtime_map:
                mtime_map[p] = current_ts
    return mtime_map


def _path_has_local_changes(path: Path) -> bool:
    """工作区相对 HEAD 是否有未提交改动（含 staged）。"""
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", rel],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        return False


def _effective_mtime(path: Path, git_mtime_map: dict[Path, float]) -> float:
    """Git commit time；未跟踪文件或本地有未提交改动时用 max(git, fs)。"""
    fs_mtime = path.stat().st_mtime
    git_mtime = git_mtime_map.get(path)
    if git_mtime is None:
        return fs_mtime
    if _path_has_local_changes(path):
        return max(git_mtime, fs_mtime)
    return git_mtime


def _frontmatter_updated_date(path: Path) -> date | None:
    """解析 wiki frontmatter 的 updated: YYYY-MM-DD（维护者显式复核日期）。"""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not fm_match:
        return None
    upd_m = re.search(r"^updated:\s*(\d{4}-\d{2}-\d{2})", fm_match.group(1), re.MULTILINE)
    if not upd_m:
        return None
    try:
        return date.fromisoformat(upd_m.group(1))
    except ValueError:
        return None


def _frontmatter_block(content: str) -> str:
    """提取 frontmatter（--- 与 --- 之间的 YAML 文本），无则返回空串。"""
    m = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    return m.group(1) if m else ""


def _frontmatter_tags(fm_block: str) -> set[str]:
    """解析 frontmatter 列表式 tags 为小写标签集合（无则空集）。"""
    m = re.search(r"^tags:\s*\n((?:\s*-\s*.+\n?)+)", fm_block, re.MULTILINE)
    if not m:
        return set()
    tags: set[str] = set()
    for line in m.group(1).splitlines():
        tm = re.match(r"\s*-\s*(.+?)\s*$", line)
        if tm:
            tags.add(tm.group(1).strip().lower())
    return tags


def _stale_claim_negated(body: str, start: int, end: int) -> bool:
    """命中词同句内出现否定线索 → 辟谣式写法，不是本页在下断言。

    前置线索回看（「不是策略 SoTA」），后置线索前看（「读『SOTA 碾压』会过读」）。
    """
    before = body[max(0, start - STALE_CLAIM_NEGATION_WINDOW) : start]
    for brk in STALE_CLAIM_SENTENCE_BREAKS:
        before = before.rsplit(brk, 1)[-1]
    if any(cue in before for cue in STALE_CLAIM_NEGATION_CUES):
        return True
    after = body[end : end + STALE_CLAIM_OVERREAD_WINDOW]
    for brk in STALE_CLAIM_SENTENCE_BREAKS:
        after = after.split(brk, 1)[0]
    return any(cue in after for cue in STALE_CLAIM_OVERREAD_CUES)


def _stale_claim_span_slug(body: str, start: int, end: int) -> str:
    """把命中词所在的英文片段还原成页面 slug（``VLA SOTA Leaderboard`` → ``vla-sota-leaderboard``）。"""
    while start > 0 and STALE_CLAIM_SPAN_CHAR_RE.fullmatch(body[start - 1]):
        start -= 1
    while end < len(body) and STALE_CLAIM_SPAN_CHAR_RE.fullmatch(body[end]):
        end += 1
    return re.sub(r"[ -]+", "-", body[start:end].strip(" -").lower())


def _stale_claim_hit(body: str, page_stems: set[str]) -> str | None:
    """返回正文里第一个「真·绝对化断言」，命中全属结构性误报时返回 ``None``。"""
    for pat in STALE_CLAIM_PATTERNS:
        for m in re.finditer(pat, body, re.IGNORECASE):
            if _stale_claim_negated(body, m.start(), m.end()):
                continue
            if _stale_claim_span_slug(body, m.start(), m.end()) in page_stems:
                continue
            if m.group(0) == "最新" and (
                STALE_CLAIM_RUNTIME_OBJECT_RE.match(body, m.end())
                or STALE_CLAIM_RUNTIME_MATH_RE.match(body, m.end())
            ):
                continue
            if STALE_CLAIM_BASELINE_LABEL_RE.match(body, m.end()):
                continue
            return m.group(0)
    return None


def _check_stale_claims(pages: list[Path], results: dict[str, Any]) -> None:
    """陈旧声明（stale claim）巡检 V1（信息型，不阻塞 CI）。

    正文（去除 frontmatter / 代码块 / 误区区块后）出现绝对化或时效性措辞
    （SOTA / state-of-the-art / 当前最强 / 最新 等），且该页 frontmatter
    ``updated`` 早于库内共享至少一个 tag 的更晚页面时，提示该断言可能已过时、
    建议复核。属信息型预警，不计入 lint 失败总数。

    命中判定见 :func:`_stale_claim_hit`：否定语境 / 库内页面名引用 / 运行时对象 /
    对照基线类别名四类结构性误报不算断言；「英文缩写速查」区块是词条释义表，
    扫描前整段剥离。
    """
    from wiki_abbrev_section import extract_abbrev_section

    page_stems = {p.stem.lower() for p in pages}
    meta: list[tuple[Path, date | None, set[str], str]] = []
    for page in pages:
        content = page.read_text(encoding="utf-8")
        body = re.sub(r"^---\n.*?\n---", "", content, flags=re.DOTALL)
        body = extract_abbrev_section(body)[1]
        body = strip_misconception_sections(strip_code_blocks(body))
        meta.append(
            (
                page,
                _frontmatter_updated_date(page),
                _frontmatter_tags(_frontmatter_block(content)),
                body,
            )
        )

    for page, upd, tags, body in meta:
        if upd is None or not tags:
            continue
        hit = _stale_claim_hit(body, page_stems)
        if hit is None:
            continue
        newer = next(
            (
                (other, o_upd)
                for other, o_upd, o_tags, _ in meta
                if other != page and o_upd is not None and o_upd > upd and tags & o_tags
            ),
            None,
        )
        if newer is not None:
            results["stale_claims"].append(
                f"{page.relative_to(REPO_ROOT)}（含绝对化措辞「{hit}」，updated={upd.isoformat()}；"
                f"同主题更新页 {newer[0].relative_to(REPO_ROOT)} updated={newer[1].isoformat()}）"
            )


def _wiki_reviewed_on_or_after_source(wiki_target: Path, src_mtime: float) -> bool:
    """frontmatter updated 日历日 ≥ source 提交日 → 视为已 review，不报 stale。"""
    reviewed = _frontmatter_updated_date(wiki_target)
    if reviewed is None:
        return False
    return reviewed >= date.fromtimestamp(src_mtime)


def _check_sources_health(results: dict[str, Any]) -> None:
    """Sources 孤儿检测 + 陈旧页面检测。"""
    sources_papers_dir = REPO_ROOT / "sources" / "papers"
    if not sources_papers_dir.exists():
        return

    for src_file in sorted(sources_papers_dir.glob("*.md")):
        src_content = src_file.read_text(encoding="utf-8")
        for m in re.finditer(r"\]\(([^)]*wiki/[^)]+\.md)\)", src_content):
            href = m.group(1).split("#")[0]
            target = (src_file.parent / href).resolve()
            if not target.exists():
                results["sources_orphans"].append(f"sources/papers/{src_file.name} → {href}")

    if os.environ.get("GITHUB_ACTIONS") == "true":
        return
    git_mtime_map = _build_git_mtime_map()
    seen_stale: set[Path] = set()
    for src_file in sorted(sources_papers_dir.glob("*.md")):
        src_content = src_file.read_text(encoding="utf-8")
        src_mtime = _effective_mtime(src_file, git_mtime_map)
        for m in re.finditer(r"\]\(([^)]*wiki/[^)]+\.md)\)", src_content):
            href = m.group(1).split("#")[0]
            wiki_target = (src_file.parent / href).resolve()
            if wiki_target.exists() and wiki_target not in seen_stale:
                if _wiki_reviewed_on_or_after_source(wiki_target, src_mtime):
                    continue
                wiki_mtime = _effective_mtime(wiki_target, git_mtime_map)
                if src_mtime > wiki_mtime + 86400:
                    seen_stale.add(wiki_target)
                    rel_wiki = wiki_target.relative_to(REPO_ROOT)
                    src_date = date.fromtimestamp(src_mtime).isoformat()
                    wiki_date = date.fromtimestamp(wiki_mtime).isoformat()
                    results["stale_pages"].append(
                        f"{rel_wiki} (wiki:{wiki_date} < sources/{src_file.name}:{src_date})"
                    )


def _check_contradictions(pages: list[Path], results: dict[str, Any]) -> None:
    """矛盾检测 — 检查同一概念在不同页面是否有相反的定性描述。"""
    canonical_facts = load_canonical_facts()
    all_pages_content = {
        p: strip_misconception_sections(p.read_text(encoding="utf-8")) for p in pages
    }
    for fact_id, fact in canonical_facts.items():
        pos_pages, neg_pages = [], []
        for page, content in all_pages_content.items():
            if not all(re.search(t, content, re.IGNORECASE) for t in fact["terms"]):
                continue
            has_pos = any(re.search(p, content, re.IGNORECASE) for p in fact["pos_claims"])
            has_neg = any(re.search(p, content, re.IGNORECASE) for p in fact["neg_claims"])
            if has_pos and has_neg:
                continue
            if has_pos:
                pos_pages.append(page.stem)
            if has_neg:
                neg_pages.append(page.stem)
        if pos_pages and neg_pages:
            results["contradictions"].append(
                f"「{fact_id}」正面描述({', '.join(pos_pages)}) vs 负面描述({', '.join(neg_pages)})"
            )


def _check_frontmatter(pages: list[Path], results: dict[str, Any]) -> None:
    """Frontmatter 检查：type 字段、updated 过期、summary 字段。"""
    fm_exempt_dirs = {"references", "roadmaps", "tech-map", "overview", "schema", "queries"}
    today_for_stale = date.today()

    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)

        if page.name.lower() not in ("readme.md", "index.md") and not any(
            d in parts for d in fm_exempt_dirs
        ):
            if fm_match:
                if not re.search(r"^type\s*:", fm_match.group(1), re.MULTILINE):
                    results["missing_type"].append(str(rel))
            else:
                results["missing_type"].append(str(rel))

        if fm_match:
            upd_m = re.search(r"^updated:\s*(\d{4}-\d{2}-\d{2})", fm_match.group(1), re.MULTILINE)
            if upd_m:
                try:
                    days_old = (today_for_stale - date.fromisoformat(upd_m.group(1))).days
                    if days_old > 180:
                        results["outdated_pages"].append(
                            f"{rel} （updated: {upd_m.group(1)}，已 {days_old} 天）"
                        )
                except ValueError:
                    pass

        if len(parts) >= 2 and parts[0] == "wiki" and parts[1] in ("concepts", "methods", "tasks"):
            if not fm_match:
                results["missing_summary"].append(str(rel))
            elif not re.search(r"^(summary|description)\s*:", fm_match.group(1), re.MULTILINE):
                results["missing_summary"].append(str(rel))


def _check_log_activity(results: dict[str, Any]) -> None:
    """知识库活跃度：优先 wiki/roadmap 近 30 天 git 提交，否则回退 log.md。"""
    try:
        git_result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "log",
                "-1",
                "--since=30 days ago",
                "--format=%cs",
                "--",
                "wiki",
                "roadmap",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if git_result.returncode == 0:
            latest_git = git_result.stdout.strip()
            if latest_git:
                return
            results["log_inactive"].append(
                "wiki/ 与 roadmap/ 最近 30 天无提交（知识库可能停止维护）"
            )
            return
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    log_path = REPO_ROOT / "log.md"
    if not log_path.exists():
        results["log_inactive"].append("log.md 文件不存在，且无法用 git 检查知识库活跃度")
        return

    log_content = log_path.read_text(encoding="utf-8")
    date_matches = re.findall(r"^## \[(\d{4}-\d{2}-\d{2})\]", log_content, re.MULTILINE)
    if not date_matches:
        results["log_inactive"].append(
            "log.md 中未找到符合格式的操作记录（格式：## [YYYY-MM-DD] ...）"
        )
        return

    latest = max(date_matches)
    try:
        days_since = (date.today() - date.fromisoformat(latest)).days
        if days_since > 30:
            results["log_inactive"].append(
                f"log.md 最后操作于 {latest}（已 {days_since} 天未更新，知识库可能停止维护）"
            )
    except ValueError:
        pass


def _check_query_format(pages: list[Path], results: dict[str, Any]) -> None:
    """queries/ 页面必须包含 Query 产物说明 + 参考来源 + 关联页面。"""
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if (
            len(parts) < 2
            or parts[0] != "wiki"
            or parts[1] != "queries"
            or page.name == "README.md"
        ):
            continue
        content = page.read_text(encoding="utf-8")
        missing_parts = []
        if "**Query 产物**" not in content:
            missing_parts.append("缺 'Query 产物' 说明")
        if "## 参考来源" not in content:
            missing_parts.append("缺 '## 参考来源' 区块")
        if "## 关联页面" not in content:
            missing_parts.append("缺 '## 关联页面' 区块")
        if missing_parts:
            results["query_format"].append(f"{rel}（{', '.join(missing_parts)}）")


def _check_formalizations(pages: list[Path], results: dict[str, Any]) -> None:
    """formalizations/ 公式块和变量解释检查。"""
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "formalizations":
            continue
        content = page.read_text(encoding="utf-8")

        if "$$" not in content and "$`" not in content and "`$" not in content:
            results["formalization_no_formula"].append(str(rel))
            continue

        _check_formalization_vars(rel, content, results)


def _var_is_explained(var: str, content: str) -> bool:
    """检查单个大写变量是否在正文中有物理含义解释。"""
    esc = re.escape(var)
    patterns = [
        rf"-\s*\$\\?{esc}[^$]*\$[^|\n]*?[:：]",
        rf"\$\\?{esc}[^$]*\$\s*[:：]\s*\S",
        rf"\|\s*\$\\?{esc}[^$]*\$\s*\|",
        rf"\$\\?{esc}[^$]*\$[^.\n]{{0,80}}(?:是|为|表示|代表|指|denote|denotes|含义|定义)",
        rf"(?:其中|where).{{0,100}}\$\\?{esc}[^$]*\$",
        rf"\$\\?{esc}\s*(?:=|\\in|\\succeq|\\succ|\\equiv|\\triangleq)",
        rf"\*\*[^*]*\$\\?{esc}[^$]*\$[^*]*\*\*",
    ]
    return any(re.search(p, content, re.MULTILINE | re.IGNORECASE) for p in patterns)


def _check_formalization_vars(rel: Path, content: str, results: dict[str, Any]) -> None:
    """检查 formalization 页面中 $$...$$ 公式变量是否有正文解释。"""
    display_blocks = re.findall(r"\$\$(.*?)\$\$", content, re.DOTALL)
    if not display_blocks:
        return
    candidate_vars: set[str] = set()
    for block in display_blocks:
        for v in re.findall(r"(?<![A-Za-z\\_^{])([A-Z])(?![A-Za-z_(])", block):
            candidate_vars.add(v)
    if not candidate_vars:
        return
    unexplained: list[str] = []
    for v in sorted(candidate_vars):
        if not _var_is_explained(v, content):
            unexplained.append(v)
    if unexplained:
        results["formalization_unexplained_vars"].append(
            f"{rel}（变量缺解释：{', '.join(unexplained)}）"
        )


def _checklist_version_num(path: Path) -> int:
    m = re.search(r"v(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _check_checklist_badge(readme_content: str, results: dict[str, Any]) -> None:
    """检查 README 中 checklist 链接版本是否与最新文件一致。"""
    checklist_files = sorted(
        (REPO_ROOT / "docs" / "checklists").glob("tech-stack-next-phase-checklist-v*.md")
    )
    if not checklist_files:
        return
    latest_checklist = max(checklist_files, key=_checklist_version_num)
    latest_ver = _checklist_version_num(latest_checklist)

    main_link_versions = re.findall(r"\[技术栈项目执行清单 v(\d+)\]", readme_content)
    if main_link_versions and int(main_link_versions[0]) < latest_ver:
        results["readme_badge"].append(
            f"README 主执行清单标题仍是 v{main_link_versions[0]}，但最新为 v{latest_ver}，请更新"
        )

    source_badge_match = re.search(
        r"\[!\[Sources Coverage\]\([^)]+\)\]\(([^)]+tech-stack-next-phase-checklist-v(\d+)\.md)\)",
        readme_content,
    )
    if not source_badge_match:
        results["readme_badge"].append("README 缺少 Sources Coverage badge 或链接格式异常")
    else:
        badge_link = source_badge_match.group(1)
        badge_ver = int(source_badge_match.group(2))
        expected_link = str(latest_checklist.relative_to(REPO_ROOT))
        if badge_ver < latest_ver or badge_link != expected_link:
            results["readme_badge"].append(
                f"README Sources badge 指向 {badge_link}，但最新应为 {expected_link}"
            )


def _check_graph_badge(readme_content: str, results: dict[str, Any]) -> None:
    """检查 README 中 Knowledge Graph badge 数据是否与 graph-stats.json 一致。"""
    graph_stats_path = REPO_ROOT / "exports" / "graph-stats.json"
    if not graph_stats_path.exists():
        return
    graph_stats = json.loads(graph_stats_path.read_text(encoding="utf-8"))
    node_count = graph_stats.get("node_count")
    edge_count = graph_stats.get("edge_count")
    graph_badge_match = re.search(
        r"\[!\[Knowledge Graph\]\(https://img\.shields\.io/badge/知识图谱-(\d+)节点_(\d+)边-blue\?logo=d3\.js\)\]\([^)]+\)",
        readme_content,
    )
    if not graph_badge_match:
        results["readme_badge"].append("README 缺少 Knowledge Graph badge 或格式异常")
    else:
        badge_nodes = int(graph_badge_match.group(1))
        badge_edges = int(graph_badge_match.group(2))
        if badge_nodes != node_count or badge_edges != edge_count:
            results["readme_badge"].append(
                f"README Knowledge Graph badge 为 {badge_nodes}节点/{badge_edges}边，但实际为 {node_count}节点/{edge_count}边"
            )


def _check_readme_badges(results: dict[str, Any]) -> None:
    """README.md 中 badges / checklist 链接应与当前仓库状态一致。"""
    readme_path = REPO_ROOT / "README.md"
    if not readme_path.exists():
        return
    readme_content = readme_path.read_text(encoding="utf-8")
    _check_checklist_badge(readme_content, results)
    _check_graph_badge(readme_content, results)


def _check_graph_orphans(results: dict[str, Any]) -> None:
    """graph-stats.json 中孤儿节点预警。"""
    graph_stats_path = REPO_ROOT / "exports" / "graph-stats.json"
    if not graph_stats_path.exists():
        return
    graph_stats = json.loads(graph_stats_path.read_text(encoding="utf-8"))
    orphan_nodes = graph_stats.get("orphan_nodes", [])
    if orphan_nodes:
        results["orphan_count"].append(
            f"发现 {len(orphan_nodes)} 个孤儿节点（无入链）：{orphan_nodes}"
        )


def _check_method_page(page: Path, rel: Path, content: str, results: dict[str, Any]) -> None:
    """单个 methods/ 页面检查：必须链接到 formalizations/concepts，必须含主要路线区块。"""
    links = extract_internal_links(content, page)
    has_required_link = False
    for target in links:
        if not target.is_relative_to(REPO_ROOT):
            continue
        target_parts = target.relative_to(REPO_ROOT).parts
        if (
            len(target_parts) >= 2
            and target_parts[0] == "wiki"
            and target_parts[1] in ("formalizations", "concepts")
        ):
            has_required_link = True
            break
    if not has_required_link:
        results["method_missing_link"].append(str(rel))

    if not re.search(r"##\s+(主要方法路线|核心技术路线|主要分类|主要技术路线)", content):
        results["method_missing_sections"].append(str(rel))


def _check_entity_page(page: Path, rel: Path, content: str, results: dict[str, Any]) -> None:
    """单个 entities/ 页面检查：必须含至少 2 个指向 methods/tasks 的出边。"""
    links = extract_internal_links(content, page)
    out_count = 0
    for target in links:
        if not target.is_relative_to(REPO_ROOT):
            continue
        t_parts = target.relative_to(REPO_ROOT).parts
        if len(t_parts) >= 2 and t_parts[0] == "wiki" and t_parts[1] in ("methods", "tasks"):
            out_count += 1
    if out_count < 2:
        results["entity_missing_outgoing"].append(f"{rel} (当前出边: {out_count})")


def _check_methods_without_practitioner_query(
    pages: list[Path],
    inbound: dict[Path, list[Path]],
    results: dict[str, Any],
) -> None:
    """高频引用的 methods/ 页面应有对应 queries/ 操作指南或 comparisons/ 对比页落地。

    判定：被超过 ``METHOD_PRACTITIONER_INBOUND_THRESHOLD`` 个其他 wiki 页面链接
    （排除自链）的 methods/ 页，若其入链来源中没有任何一个属于 queries/ 或
    comparisons/ 目录，则视为"待落地"——理论介绍已扩散，但缺少可操作的选型/
    对比/Query 指南。属信息型预警，不计入 lint 失败总数。
    """
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "methods":
            continue
        if page.name.lower() == "readme.md":
            continue

        resolved = page.resolve()
        sources = [src for src in inbound.get(resolved, []) if src != resolved]
        if len(sources) <= METHOD_PRACTITIONER_INBOUND_THRESHOLD:
            continue

        has_practitioner = False
        for src in sources:
            if not src.is_relative_to(REPO_ROOT):
                continue
            src_parts = src.relative_to(REPO_ROOT).parts
            if (
                len(src_parts) >= 2
                and src_parts[0] == "wiki"
                and src_parts[1] in ("queries", "comparisons")
            ):
                has_practitioner = True
                break

        if not has_practitioner:
            results["methods_without_practitioner_query"].append(
                f"{rel}（被 {len(sources)} 个页面引用，无 queries/comparisons 落地）"
            )


def _check_duplicate_arxiv(pages: list[Path], results: dict[str, Any]) -> None:
    """V30: 同一 frontmatter arxiv ID 只允许出现在一个 wiki 页面（阻塞 CI）。

    一篇论文（一个 arXiv ID）在站点/图谱中只应有一个 canonical 节点。
    历史上按策展来源批量建节点导致同一论文出现多个同名页面（如 SONIC
    双节点），读者困惑且互链分裂。此检查以 frontmatter ``arxiv:`` 为唯一键
    （正文引用不计），发现重复即报错；合并方法见 log.md 中 structural 合并
    记录与 schema/page-aliases.json。
    """
    arxiv_pages: dict[str, list[str]] = {}
    for page in pages:
        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            continue
        m = re.search(r'^arxiv:\s*"?(\d{4}\.\d{4,5})"?\s*$', fm_match.group(1), re.MULTILINE)
        if not m:
            continue
        arxiv_pages.setdefault(m.group(1), []).append(str(page.relative_to(REPO_ROOT)))
    for arxiv_id, page_list in sorted(arxiv_pages.items()):
        if len(page_list) > 1:
            results["duplicate_arxiv"].append(
                f"arXiv:{arxiv_id} → {' 与 '.join(sorted(page_list))}"
            )


def _check_paper_entity_metadata(pages: list[Path], results: dict[str, Any]) -> None:
    """V23/V31: paper-* 实体页元数据基线检查（信息型，不阻塞 CI）。

    针对 ``wiki/entities/paper-*.md``：
      - frontmatter 至少包含 ``arxiv`` / ``venue`` / ``code`` 三类来源键之一；
      - 正文至少存在「方法栈 / 评测 / 与其他工作对比」三段式；
      - 正文须存在 ``## 结论``（政策：后续 ingest / 大幅改写必做）。

    用于 ingest 工作流自检入口，缺失项作为基线快照写入 lint 报告。
    """
    method_patterns = ["方法栈", "流程总览", "流程", "核心机制", "核心信息", "pipeline", "方法"]
    eval_patterns = ["评测", "实验", "量化", "结果", "benchmark"]
    compare_patterns = ["与其他工作", "与其他页面", "对比", "比较"]
    conclusion_patterns = ["结论"]
    source_keys = ("arxiv", "venue", "code")

    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 3 or parts[0] != "wiki" or parts[1] != "entities":
            continue
        if not page.name.startswith("paper-"):
            continue

        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        fm_block = fm_match.group(1) if fm_match else ""
        has_source = any(re.search(rf"^{key}\s*:", fm_block, re.MULTILINE) for key in source_keys)
        if not has_source:
            results["paper_missing_source_meta"].append(str(rel))

        missing_sections = []
        if not has_section(content, method_patterns):
            missing_sections.append("方法")
        if not has_section(content, eval_patterns):
            missing_sections.append("评测")
        if not has_section(content, compare_patterns):
            missing_sections.append("对比")
        if missing_sections:
            results["paper_missing_three_sections"].append(
                f"{rel}（缺 {' / '.join(missing_sections)}）"
            )

        if not has_section(content, conclusion_patterns):
            results["paper_missing_conclusions"].append(str(rel))


def _check_dataset_entity_metadata(pages: list[Path], results: dict[str, Any]) -> None:
    """V25: 数据集实体页元数据巡检 V1（信息型，不阻塞 CI）。

    针对 frontmatter ``tags`` 含 ``dataset`` 的 ``wiki/entities/*.md``：检查正文是否
    覆盖「规模 / 模态 / 许可证 / 重定向就绪度」四类标准化速查维度（以关键词命中近似）。
    缺失维度作为 INFO 级提示写入 lint 报告，沉淀数据层 ingest 的元数据基线，
    不计入 lint 失败总数。
    """
    dimension_patterns: list[tuple[str, list[str]]] = [
        # 规模：时长 / 条目 / 序列 / 被试 / 帧 等量级描述
        (
            "规模",
            [
                "规模",
                "小时",
                "hours?",
                "条",
                "序列",
                "帧",
                "frames?",
                "samples?",
                "被试",
                "subjects?",
                "轨迹",
                "episodes?",
                "数量级",
            ],
        ),
        # 模态：动捕 / 视频 / 深度 / 点云 / SMPL / 文本 等数据形态
        (
            "模态",
            [
                "模态",
                "modality",
                "mocap",
                "动捕",
                "视频",
                "video",
                "rgb",
                "深度",
                "depth",
                "点云",
                "point cloud",
                "smpl",
                "文本",
                "imu",
            ],
        ),
        # 许可证：开源协议 / 注册 / 商用约束
        (
            "许可证",
            [
                "许可证",
                "license",
                "协议",
                "授权",
                "注册",
                "cc[- ]?by",
                "\\bmit\\b",
                "apache",
                "开源",
                "non-commercial",
                "商用",
            ],
        ),
        # 重定向就绪度：能否直接喂给目标形态 / 是否需重定向 / 训练输入
        (
            "重定向就绪度",
            [
                "重定向",
                "retarget",
                "形态",
                "骨架",
                "适配",
                "morphology",
                "训练输入",
                "策略输入",
                "物理可行",
                "可部署",
            ],
        ),
    ]

    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 3 or parts[0] != "wiki" or parts[1] != "entities":
            continue

        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        fm_block = fm_match.group(1) if fm_match else ""
        tags = _frontmatter_tags(fm_block)
        # tags 既支持列表式（- dataset），也支持内联式 [dataset, ...]
        if "dataset" not in tags and not re.search(
            r"^tags:\s*\[[^\]]*\bdataset\b", fm_block, re.MULTILINE
        ):
            continue

        body = re.sub(r"^---\n.*?\n---", "", content, flags=re.DOTALL).lower()
        missing = [
            name for name, pats in dimension_patterns if not any(re.search(p, body) for p in pats)
        ]
        if missing:
            results["dataset_missing_metadata"].append(f"{rel}（缺 {' / '.join(missing)}）")


# 前缀匹配的已知假阳 token：``focal``（focal-loss，检测损失函数）会被 ``foc``
# （Field-Oriented Control，磁场定向控制）前缀命中，但二者毫无关系。
TAG_KEYWORD_FALSE_POSITIVE_TOKENS: frozenset[str] = frozenset({"focal"})


def _tag_keyword_match(tags: set[str], keywords: tuple[str, ...]) -> bool:
    """按连字符 token 前缀匹配 tag 与关键词，覆盖 ``eda-tool`` / ``foc-driver`` /
    ``actuator-network`` / ``actuators`` 等派生标签，同时避免裸子串匹配把
    ``impedance`` / ``bipedal`` / ``pedagogy`` / ``bytedance`` 误判为含 ``eda``。
    """
    for tag in tags:
        for token in tag.split("-"):
            if token in TAG_KEYWORD_FALSE_POSITIVE_TOKENS:
                continue
            if any(token.startswith(kw) for kw in keywords):
                return True
    return False


# 物理保真度知识链枢纽页：动力学/仿真/物理概念页应回链至少一个，形成保真度链路闭环
PHYSICS_FIDELITY_HUBS: tuple[str, ...] = (
    "simulation-physics-fidelity",
    "physics-fidelity-sim2real-gap",
    "hub-physics-fidelity",
)


def _check_physics_concept_crosslink(pages: list[Path], results: dict[str, Any]) -> None:
    """V26: 动力学/仿真概念页交叉链路巡检 V1（信息型，不阻塞 CI）。

    对 frontmatter ``tags`` 含 ``dynamics`` / ``simulation`` / ``physics`` 的
    ``wiki/concepts/*`` 与 ``wiki/formalizations/*`` 概念页，检查正文是否回链到
    「仿真物理保真度」知识链枢纽页（``simulation-physics-fidelity`` /
    ``physics-fidelity-sim2real-gap``）。缺失回链作为 INFO 级提示写入 lint 报告，
    沉淀物理保真度知识链的交叉链路基线，不计入 lint 失败总数。枢纽页自身豁免。
    """
    target_tags = {"dynamics", "simulation", "physics"}
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 3 or parts[0] != "wiki" or parts[1] not in ("concepts", "formalizations"):
            continue
        if page.stem in PHYSICS_FIDELITY_HUBS:
            continue

        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        fm_block = fm_match.group(1) if fm_match else ""
        # tags 既支持列表式（- dynamics），也支持内联式 [dynamics, ...]
        tags = _frontmatter_tags(fm_block)
        inline = re.search(r"^tags:\s*\[([^\]]*)\]", fm_block, re.MULTILINE)
        if inline:
            tags |= {t.strip().lower() for t in inline.group(1).split(",")}
        if not (tags & target_tags):
            continue

        if not any(hub in content for hub in PHYSICS_FIDELITY_HUBS):
            results["physics_concept_crosslink"].append(str(rel))


# 接触力控知识链枢纽页：接触/力控/阻抗/操作/触觉概念页应回链至少一个，形成力旋量闭环链路
CONTACT_FORCE_CONTROL_HUBS: tuple[str, ...] = (
    "contact-wrench-closed-loop",
    "hub-contact-force-control",
)

# 关键词以子串方式匹配 tag，覆盖 force-control / impedance-control / tactile-sensing /
# contact-rich 等派生标签，避免精确匹配漏掉常见变体
CONTACT_CONTROL_TAG_KEYWORDS: tuple[str, ...] = (
    "contact",
    "force-control",
    "impedance",
    "manipulation",
    "tactile",
)


def _check_contact_control_crosslink(pages: list[Path], results: dict[str, Any]) -> None:
    """V27: 接触/力控/操作概念页交叉链路巡检 V1（信息型，不阻塞 CI）。

    对 frontmatter ``tags`` 含 ``contact`` / ``force-control`` / ``impedance`` /
    ``manipulation`` / ``tactile``（以子串方式匹配派生标签）的 ``wiki/concepts/*``
    概念页，检查正文是否回链到「接触力旋量闭环」知识链枢纽页
    （``contact-wrench-closed-loop`` / ``hub-contact-force-control``）。缺失回链
    作为 INFO 级提示写入 lint 报告，沉淀接触力旋量闭环知识链的交叉链路基线，不计入
    lint 失败总数。枢纽页自身豁免。
    """
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 3 or parts[0] != "wiki" or parts[1] != "concepts":
            continue
        if page.stem in CONTACT_FORCE_CONTROL_HUBS:
            continue

        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        fm_block = fm_match.group(1) if fm_match else ""
        # tags 既支持列表式（- contact），也支持内联式 [contact, ...]
        tags = _frontmatter_tags(fm_block)
        inline = re.search(r"^tags:\s*\[([^\]]*)\]", fm_block, re.MULTILINE)
        if inline:
            tags |= {t.strip().lower() for t in inline.group(1).split(",")}
        if not _tag_keyword_match(tags, CONTACT_CONTROL_TAG_KEYWORDS):
            continue

        if not any(hub in content for hub in CONTACT_FORCE_CONTROL_HUBS):
            results["contact_control_crosslink"].append(str(rel))


# 具身大模型分类学选型闭环知识链枢纽页：VLM/VLN/VLA/VLX/World-Model 家族概念/对比页
# 应回链至少一个，形成「感知 → 导航 → 执行 → 扩展 → 推演」五层选型闭环链路
EMBODIED_FM_HUBS: tuple[str, ...] = (
    "embodied-fm-taxonomy-loop",
    "hub-embodied-foundation-model",
)

# 关键词以子串方式匹配 tag，覆盖 vla-model / vision-language-navigation /
# world-model-* 等派生标签，避免精确匹配漏掉常见变体
EMBODIED_FM_TAG_KEYWORDS: tuple[str, ...] = (
    "vlm",
    "vln",
    "vla",
    "vlx",
    "world-model",
)


def _check_embodied_fm_crosslink(pages: list[Path], results: dict[str, Any]) -> None:
    """V28: 具身大模型家族概念页交叉链路巡检 V1（信息型，不阻塞 CI）。

    对 frontmatter ``tags`` 含 ``vlm`` / ``vln`` / ``vla`` / ``vlx`` /
    ``world-model``（以子串方式匹配派生标签）的 ``wiki/concepts/*`` 与
    ``wiki/comparisons/*`` 页，检查正文是否回链到「具身大模型分类学选型闭环」
    知识链枢纽页（``embodied-fm-taxonomy-loop`` / ``hub-embodied-foundation-model``）。
    缺失回链作为 INFO 级提示写入 lint 报告，沉淀具身大模型选型闭环知识链的交叉
    链路基线，不计入 lint 失败总数。枢纽页自身豁免。
    """
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 3 or parts[0] != "wiki" or parts[1] not in ("concepts", "comparisons"):
            continue
        if page.stem in EMBODIED_FM_HUBS:
            continue

        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        fm_block = fm_match.group(1) if fm_match else ""
        # tags 既支持列表式（- vla），也支持内联式 [vla, ...]
        tags = _frontmatter_tags(fm_block)
        inline = re.search(r"^tags:\s*\[([^\]]*)\]", fm_block, re.MULTILINE)
        if inline:
            tags |= {t.strip().lower() for t in inline.group(1).split(",")}
        if not _tag_keyword_match(tags, EMBODIED_FM_TAG_KEYWORDS):
            continue

        if not any(hub in content for hub in EMBODIED_FM_HUBS):
            results["embodied_fm_crosslink"].append(str(rel))


EVAL_BENCHMARK_HUBS: tuple[str, ...] = (
    "embodied-eval-benchmark-selection-loop",
    "hub-embodied-eval-benchmark",
)

# 关键词以子串方式匹配 tag，覆盖 benchmark-suite / policy-evaluation /
# eval-* 等派生标签，避免精确匹配漏掉常见变体
EVAL_BENCHMARK_TAG_KEYWORDS: tuple[str, ...] = (
    "benchmark",
    "evaluation",
)


def _check_eval_benchmark_crosslink(pages: list[Path], results: dict[str, Any]) -> None:
    """V29: 具身大模型评测基准页交叉链路巡检 V1（信息型，不阻塞 CI）。

    对 frontmatter ``tags`` 含 ``benchmark`` / ``evaluation``（以子串方式匹配
    派生标签）的 ``wiki/entities/*`` / ``wiki/comparisons/*`` / ``wiki/concepts/*``
    页，检查正文是否回链到「具身大模型评测基准选型闭环」知识链枢纽页
    （``embodied-eval-benchmark-selection-loop`` / ``hub-embodied-eval-benchmark``）。
    缺失回链作为 INFO 级提示写入 lint 报告，沉淀评测基准选型闭环知识链的交叉
    链路基线，不计入 lint 失败总数。枢纽页自身豁免。
    """
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if (
            len(parts) < 3
            or parts[0] != "wiki"
            or parts[1] not in ("entities", "comparisons", "concepts")
        ):
            continue
        if page.stem in EVAL_BENCHMARK_HUBS:
            continue

        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        fm_block = fm_match.group(1) if fm_match else ""
        # tags 既支持列表式（- benchmark），也支持内联式 [benchmark, ...]
        tags = _frontmatter_tags(fm_block)
        inline = re.search(r"^tags:\s*\[([^\]]*)\]", fm_block, re.MULTILINE)
        if inline:
            tags |= {t.strip().lower() for t in inline.group(1).split(",")}
        if not _tag_keyword_match(tags, EVAL_BENCHMARK_TAG_KEYWORDS):
            continue

        if not any(hub in content for hub in EVAL_BENCHMARK_HUBS):
            results["eval_benchmark_crosslink"].append(str(rel))


ACTUATOR_DRIVE_CHAIN_HUBS: tuple[str, ...] = (
    "actuator-drive-chain-selection-loop",
    "hub-actuator-drive-chain",
)

# 关键词以子串方式匹配 tag，覆盖 actuator-modeling / eda-tool / foc-driver
# 等派生标签，避免精确匹配漏掉常见变体
ACTUATOR_DRIVE_CHAIN_TAG_KEYWORDS: tuple[str, ...] = (
    "actuator",
    "eda",
    "foc",
)


def _check_actuator_drive_chain_crosslink(pages: list[Path], results: dict[str, Any]) -> None:
    """V30: 执行器驱动链页交叉链路巡检 V1（信息型，不阻塞 CI）。

    对 frontmatter ``tags`` 含 ``actuator`` / ``eda`` / ``foc``（以子串方式匹配
    派生标签）的 ``wiki/entities/*`` / ``wiki/comparisons/*`` / ``wiki/concepts/*``
    页，检查正文是否回链到「执行器驱动链选型闭环」知识链枢纽页
    （``actuator-drive-chain-selection-loop`` / ``hub-actuator-drive-chain``）。
    缺失回链作为 INFO 级提示写入 lint 报告，沉淀执行器驱动链选型闭环知识链的
    交叉链路基线，不计入 lint 失败总数。枢纽页自身豁免。
    """
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if (
            len(parts) < 3
            or parts[0] != "wiki"
            or parts[1] not in ("entities", "comparisons", "concepts")
        ):
            continue
        if page.stem in ACTUATOR_DRIVE_CHAIN_HUBS:
            continue

        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        fm_block = fm_match.group(1) if fm_match else ""
        # tags 既支持列表式（- actuator），也支持内联式 [actuator, ...]
        tags = _frontmatter_tags(fm_block)
        inline = re.search(r"^tags:\s*\[([^\]]*)\]", fm_block, re.MULTILINE)
        if inline:
            tags |= {t.strip().lower() for t in inline.group(1).split(",")}
        if not _tag_keyword_match(tags, ACTUATOR_DRIVE_CHAIN_TAG_KEYWORDS):
            continue

        if not any(hub in content for hub in ACTUATOR_DRIVE_CHAIN_HUBS):
            results["actuator_drive_chain_crosslink"].append(str(rel))


PERCEPTION_STACK_HUBS: tuple[str, ...] = (
    "robot-perception-stack-selection-loop",
    "hub-perception-stack",
)

# 关键词以连字符 token 前缀方式匹配 tag，覆盖 object-detection / instance-segmentation
# / semantic-mapping / promptable-segmentation 等派生标签，避免裸子串误判
PERCEPTION_STACK_TAG_KEYWORDS: tuple[str, ...] = (
    "detection",
    "segmentation",
    "perception",
    "semantic",
)


def _check_perception_stack_crosslink(pages: list[Path], results: dict[str, Any]) -> None:
    """V31: 感知栈页交叉链路巡检 V1（信息型，不阻塞 CI）。

    对 frontmatter ``tags`` 含 ``detection`` / ``segmentation`` / ``perception`` /
    ``semantic``(-mapping)（以连字符 token 前缀方式匹配派生标签）的
    ``wiki/entities/*`` / ``wiki/comparisons/*`` / ``wiki/concepts/*`` /
    ``wiki/methods/*`` 页，检查正文是否回链到「机器人视觉感知栈选型闭环」知识链枢纽页
    （``robot-perception-stack-selection-loop`` / ``hub-perception-stack``）。
    缺失回链作为 INFO 级提示写入 lint 报告，沉淀机器人视觉感知栈选型闭环知识链的
    交叉链路基线，不计入 lint 失败总数。枢纽页自身豁免。
    """
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if (
            len(parts) < 3
            or parts[0] != "wiki"
            or parts[1] not in ("entities", "comparisons", "concepts", "methods")
        ):
            continue
        if page.stem in PERCEPTION_STACK_HUBS:
            continue

        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        fm_block = fm_match.group(1) if fm_match else ""
        # tags 既支持列表式（- object-detection），也支持内联式 [object-detection, ...]
        tags = _frontmatter_tags(fm_block)
        inline = re.search(r"^tags:\s*\[([^\]]*)\]", fm_block, re.MULTILINE)
        if inline:
            tags |= {t.strip().lower() for t in inline.group(1).split(",")}
        if not _tag_keyword_match(tags, PERCEPTION_STACK_TAG_KEYWORDS):
            continue

        if not any(hub in content for hub in PERCEPTION_STACK_HUBS):
            results["perception_stack_crosslink"].append(str(rel))


def _check_tool_institutions(pages: list[Path], results: dict[str, Any]) -> None:
    """entities/ 软件工具页须能派生至少一个所属机构（见 schema/institutions.json）。"""
    from bump_institution_tags import (
        INSTITUTIONS_PATH,
        _build_alias_map,
        _derive_institutions,
        _load_registry,
        _parse_frontmatter_list,
        is_tool_entity,
    )

    alias_map = _build_alias_map(_load_registry(INSTITUTIONS_PATH))
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "entities":
            continue
        content = page.read_text(encoding="utf-8")
        tags = _parse_frontmatter_list(content, "tags")
        if not is_tool_entity(rel.as_posix(), tags):
            continue
        if not _derive_institutions(content, alias_map):
            results["tool_missing_institution"].append(str(rel))


def _check_methods_entities(pages: list[Path], results: dict[str, Any]) -> None:
    """methods/ 页面结构检查 + entities/ 出边检查。

    注意：原始实现中 entities/ 检查写在 methods/ 循环内部，
    因 ``continue`` 门控导致 entities/ 页面实际不会被扫描。
    此处保持相同行为——仅在 methods/ 页面上执行两项检查。
    """
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "methods":
            continue
        content = page.read_text(encoding="utf-8")
        _check_method_page(page, rel, content, results)

        if parts[1] == "entities":
            _check_entity_page(page, rel, content, results)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def coverage_stats(results: dict[str, Any]) -> dict[str, int]:
    """从 lint 结果提取 Sources 覆盖率，供 home-stats / preflight 复用。"""
    covered = int(results.get("_ingest_covered", 0))
    total = int(results.get("_ingest_total", 0))
    percent = round(covered / total * 100) if total else 0
    return {"covered": covered, "total": total, "percent": percent}


def save_lint_report(results: dict[str, Any], report_path: Path | None = None) -> Path:
    """将 format_report 写入 exports/lint-report.md。"""
    target = report_path or (REPO_ROOT / "exports" / "lint-report.md")
    target.parent.mkdir(exist_ok=True)
    report = format_report(results)
    target.write_text("# Wiki 健康报告\n\n" + report, encoding="utf-8")
    return target


def lint() -> dict[str, Any]:
    pages = get_wiki_pages()
    page_set = {p.resolve() for p in pages}
    inbound, broken_links = _build_link_index(pages, page_set)
    results = _empty_results()

    _check_per_page(pages, inbound, broken_links, results)
    _check_missing_concepts(pages, results)
    _check_missing_concept_pages(pages, results)
    _check_sources_health(results)
    _check_contradictions(pages, results)
    _check_frontmatter(pages, results)
    _check_log_activity(results)
    _check_query_format(pages, results)
    _check_formalizations(pages, results)
    _check_readme_badges(results)
    _check_graph_orphans(results)
    _check_methods_entities(pages, results)
    _check_methods_without_practitioner_query(pages, inbound, results)
    _check_paper_entity_metadata(pages, results)
    _check_duplicate_arxiv(pages, results)
    _check_dataset_entity_metadata(pages, results)
    _check_stale_claims(pages, results)
    _check_physics_concept_crosslink(pages, results)
    _check_contact_control_crosslink(pages, results)
    _check_embodied_fm_crosslink(pages, results)
    _check_eval_benchmark_crosslink(pages, results)
    _check_actuator_drive_chain_crosslink(pages, results)
    _check_perception_stack_crosslink(pages, results)
    _check_tool_institutions(pages, results)

    return results


def _failing_total(results: dict[str, Any]) -> int:
    """计入 lint 失败的问题总数：排除内部统计键与信息型预警键。"""
    return sum(
        len(v) for k, v in results.items() if not k.startswith("_") and k not in INFO_ONLY_KEYS
    )


def _info_total(results: dict[str, Any]) -> int:
    """信息型预警总数（不阻塞 CI）。

    ``paper_missing_conclusions`` 历史 backlog 可能数百条，按 **1 个桶** 计入总数，
    避免每次 lint 摘要被刷成「另含 800+ 条」；明细仍在报告中截断列出。
    """
    total = 0
    for k in INFO_ONLY_KEYS:
        items = results.get(k, [])
        if k == "paper_missing_conclusions":
            total += 1 if items else 0
        else:
            total += len(items)
    return total


def format_report(results: dict[str, Any]) -> str:
    today = date.today().isoformat()
    lines = [f"## [{today}] lint | health-check | 自动化 wiki 健康检查", ""]

    failing = _failing_total(results)
    info = _info_total(results)
    lines.append(f"共发现 **{failing}** 个问题（另含 **{info}** 条信息型预警）：")
    lines.append("")

    sections = [
        ("orphan_pages", "孤儿页（无入链）", "⚠️"),
        ("missing_related", "缺少关联页面区块", "⚠️"),
        ("missing_sources", "缺少参考来源区块", "⚠️"),
        (
            "missing_abbrev_glossary",
            "缺少英文缩写速查区块（新建/大幅改写页须补齐；全库 backlog 信息型）",
            "💡",
        ),
        (
            "abbrev_glossary_wrong_order",
            "英文缩写速查位置错误（应在「一句话定义」之后、「为什么重要」之前）",
            "❌",
        ),
        ("broken_links", "断链（内链目标不存在）", "❌"),
        ("wikilink_syntax", "禁止的 [[...]] wikilink 写法（请用标准 Markdown）", "❌"),
        (
            "unclosed_autolinks",
            "未闭合的 <https://...> 自动链接（表格/列表行缺少结尾 >）",
            "❌",
        ),
        ("broken_source_refs", "引用了不存在的 sources/ 文件", "❌"),
        (
            "duplicate_arxiv",
            "同一 frontmatter arxiv ID 出现在多个页面（一篇论文只允许一个 canonical 节点）",
            "❌",
        ),
        ("sources_orphans", "Sources 孤儿（sources/papers 死链）", "❌"),
        ("stale_pages", "陈旧页面（sources 比 wiki 新，建议 review）", "⚠️"),
        ("outdated_pages", "可能过期（updated: 距今 > 180 天）", "⚠️"),
        ("contradictions", "潜在矛盾（跨页面相反定性描述）", "⚠️"),
        ("stub_pages", "空壳页面（< 200 字）", "⚠️"),
        ("missing_pages", "频繁提及但缺少 wiki 页面的概念", "💡"),
        (
            "missing_concept_pages",
            "多页以加粗/反引号高频引用但缺独立 concepts/methods/formalizations 页（信息型，不阻塞 CI）",
            "💡",
        ),
        ("missing_type", "Frontmatter 缺少 type 字段", "⚠️"),
        ("log_inactive", "知识库活跃度警告（git / log.md）", "⚠️"),
        ("missing_summary", "缺少摘要字段（summary/description）", "⚠️"),
        ("query_format", "Query 页面格式不完整（缺 Query 产物/参考来源/关联页面）", "⚠️"),
        ("formalization_no_formula", "Formalization 页面缺少公式块", "⚠️"),
        ("formalization_unexplained_vars", "Formalization 公式变量缺少正文物理含义解释", "⚠️"),
        ("readme_badge", "README checklist 链接版本不一致", "⚠️"),
        ("orphan_count", "图谱孤儿节点预警（graph-stats.json）", "⚠️"),
        ("method_missing_link", "Methods 页面缺少 Formalization/Concept 链接", "⚠️"),
        ("method_missing_sections", "Methods 页面缺少主要路线区块", "⚠️"),
        ("entity_missing_outgoing", "Entities 页面缺少 Methods/Tasks 关联出边", "⚠️"),
        ("tool_missing_institution", "工具实体缺少可派生的所属机构", "❌"),
        (
            "methods_without_practitioner_query",
            "高频引用 methods/ 缺 queries/ 或 comparisons/ 落地（信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "paper_missing_source_meta",
            "paper-* 实体 frontmatter 缺 arxiv/venue/code 来源键（信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "paper_missing_three_sections",
            "paper-* 实体正文缺「方法/评测/对比」三段式（信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "paper_missing_conclusions",
            "paper-* 实体缺「结论」章节（信息型；后续 ingest 必做）",
            "💡",
        ),
        (
            "dataset_missing_metadata",
            "dataset 实体正文缺「规模/模态/许可证/重定向就绪度」速查维度（信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "stale_claims",
            "陈旧声明（含绝对化措辞但同主题有更晚更新页，建议复核；信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "physics_concept_crosslink",
            "动力学/仿真/物理概念页缺回链「仿真物理保真度」知识链枢纽（信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "contact_control_crosslink",
            "接触/力控/操作概念页缺回链「接触力旋量闭环」知识链枢纽（信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "embodied_fm_crosslink",
            "VLM/VLN/VLA/VLX/World-Model 家族概念/对比页缺回链「具身大模型分类学选型闭环」知识链枢纽（信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "eval_benchmark_crosslink",
            "benchmark/evaluation 实体/对比/概念页缺回链「具身大模型评测基准选型闭环」知识链枢纽（信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "actuator_drive_chain_crosslink",
            "actuator/eda/foc 实体/对比/概念页缺回链「执行器驱动链选型闭环」知识链枢纽（信息型，不阻塞 CI）",
            "💡",
        ),
        (
            "perception_stack_crosslink",
            "detection/segmentation/perception/semantic-mapping 实体/对比/概念/方法页缺回链「机器人视觉感知栈选型闭环」知识链枢纽（信息型，不阻塞 CI）",
            "💡",
        ),
    ]

    for key, label, icon in sections:
        items = results[key]
        lines.append(f"### {icon} {label}（{len(items)} 个）")
        if items:
            # 历史 paper-* 缺「结论」可能数百条：报告只展示前 15 条，避免刷屏
            display = items
            extra = 0
            if key == "paper_missing_conclusions" and len(items) > 15:
                display = items[:15]
                extra = len(items) - 15
            for item in display:
                lines.append(f"- {item}")
            if extra:
                lines.append(
                    f"- … 另有 {extra} 个（历史 backlog；"
                    "新建 / 大幅改写的 paper-* 必须含 `## 结论`）"
                )
        else:
            lines.append("- 无")
        lines.append("")

    covered = results.get("_ingest_covered", 0)
    total = results.get("_ingest_total", 0)
    pct = round(covered / total * 100) if total else 0
    lines.append(f"📊 Sources 覆盖率：{covered}/{total} ({pct}%) wiki/entity 页有 ingest 来源")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Robotics_Notebooks wiki lint 检查")
    parser.add_argument(
        "--write-log",
        action="store_true",
        help="将结果插入 log.md 顶部（叙事层；与 append_log 一致）",
    )
    parser.add_argument(
        "--report", action="store_true", help="将 markdown 健康报告保存到 exports/lint-report.md"
    )
    args = parser.parse_args()

    print("正在扫描 wiki/ 目录...")
    results = lint()
    report = format_report(results)

    print(report)

    total = _failing_total(results)
    info = _info_total(results)
    if total == 0:
        if info:
            print(f"✅ 所有检查通过！（另含 {info} 条信息型预警，不阻塞 CI）")
        else:
            print("✅ 所有检查通过！")
    else:
        print(f"⚠️  共发现 {total} 个问题，请参考上方报告处理。")

    if args.write_log:
        from log_md import DEFAULT_LOG_PATH, write_log_prepend

        write_log_prepend(report + "\n", DEFAULT_LOG_PATH)
        print(f"\n已将报告插入 {DEFAULT_LOG_PATH} 顶部")

    if args.report:
        exports_dir = REPO_ROOT / "exports"
        exports_dir.mkdir(exist_ok=True)
        report_path = exports_dir / "lint-report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Wiki 健康报告\n\n")
            f.write(report)
        print(f"\n已将健康报告保存到 {report_path}")

    if total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
