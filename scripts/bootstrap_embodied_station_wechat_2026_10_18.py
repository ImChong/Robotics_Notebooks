#!/usr/bin/env python3
"""Bootstrap ingest: 具身智能小站 2026-10-18（26 篇）公众号盘点。

- 抓取方式：`scripts/fetch_wechat_article.py`（Chromium + 微信 UA），正文落盘 `sources/raw/`。
- 26 条论文 → 26 个独立 `wiki/entities/paper-*.md` 节点：新建 19，复用 7（arXiv 已在库内）。
- 开源结论（步骤 2.5）：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取并扫描
  GitHub 链接 / "coming soon" 字样（入库日 2026-10-18）。
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()

ARTICLES: dict[str, dict[str, Any]] = {
    "a1": {
        "key": "26_papers_embodiedswe_2026-10-18",
        "title": "机器人会写控制程序之后，能否自己造训练数据？",
        "url": "https://mp.weixin.qq.com/s/mmnZuhiHkuzKCvdtU_XTNA",
        "date": "2026-10-18",
        "n": 26,
        "map": "embodied-station-2026-10-18-26-papers-technology-map.md",
        "map_title": "编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标",
        "map_summary": "具身智能小站 2026-10-18 盘点：EmbodiedSWE 编程智能体造示范、VLAQuantBench 闭环量化评测、Uranus 交互式视频仿真，以及触觉灵巧、人形控制、世界模型等 26 篇；26/26 独立节点（新建 19 / 复用 7）。",
        "view": "机器人会写控制程序之后，成功程序本身就能变成可验证、可扩增的示范来源；与之配套的是用闭环任务而非数值误差来评测 VLA 的部署改动。本期深读 EmbodiedSWE，跟进 VLAQuantBench 与 Uranus。",
    },
}


def blog_name(a: str) -> str:
    return f"wechat_embodied_station_{ARTICLES[a]['key']}.md"


M = "../methods/"
C = "../concepts/"
T = "../tasks/"
WAM = C + "world-action-models.md"
VLA = M + "vla.md"
RL = M + "reinforcement-learning.md"
MANIP = T + "manipulation.md"
HUM = T + "humanoid-locomotion.md"
LOCOMANIP = T + "loco-manipulation.md"
S2R = C + "sim2real.md"
FM = C + "flow-matching-embodied-policy.md"
GWM = M + "generative-world-models.md"
IL = M + "imitation-learning.md"
SIMEVAL = C + "simulation-evaluation-infrastructure.md"
DEX = C + "dexterous-kinematics.md"
TELEOP = T + "teleoperation.md"
TACT = C + "tactile-sensing.md"
RETARGET = C + "motion-retargeting.md"

OPEN = "已开源"
PART = "部分开源"
TBD = "待发布"
NOLINK = "项目页未列 GitHub"

# 已在库内的 arXiv → 复用节点（不新建，仅补来源与地图关联）
REUSE: dict[str, str] = {
    "2609.27308": "paper-embodiedswe",
    "2609.24815": "paper-uranus",
    "2609.24682": "paper-thaw-vla",
    "2609.24778": "paper-h2rbench",
    "2609.25363": "paper-hotice",
    "2609.24552": "paper-smoothness-constraint-locomotion",
    "2609.23610": "paper-primo-human-motion-odometry",
}

# n 为公众号内序号；reuse=True 的条目只用于索引表
PAPERS: list[dict] = [
    dict(
        n=1,
        a="a1",
        g="数据生成与示范",
        short="EmbodiedSWE",
        arxiv="2609.27308",
        open=OPEN,
        reuse=True,
    ),
    dict(
        n=2,
        a="a1",
        g="VLA 评测与部署",
        slug="vlaquantbench",
        short="VLAQuantBench",
        arxiv="2609.25376",
        title="VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models",
        code="https://github.com/jiuyixu25/VLAQuantBench",
        project=None,
        open=OPEN,
        open_note="仓库可达（git ls-remote）；公众号称显存节省不等于推理加速，复现须自备 LIBERO 与模型权重。",
        tags=["paper", "vla", "quantization", "benchmark", "deployment", "libero"],
        one="统一量化格式与作用范围，在闭环任务上比较 VLA 的训练后量化（独立层组与联合量化），把成功率、校准条件与真实算子延迟一并检查。",
        why="压低数值误差并不保证机器人完成任务；部署团队需要闭环成功率而不是逐层误差来决定量化方案。",
        mech="在统一 RTN 配方下统一量化格式与作用范围，比较独立层组与联合量化，并检查校准与真实算子延迟。",
        metrics="公众号转述：OpenVLA-OFT 的 LIBERO-Goal 成功率由 BF16 的 96% 降至 W8A8 的 13%（每配置 200 轮）；强调格式、层范围与校准耦合，不能推广为所有 8 位量化都失效。",
        concl="VLA 量化应以闭环成功率为准：同一 8 位格式在不同层范围与校准下可从可用到几乎失效，选型前须成套检查。",
        related=[
            VLA,
            SIMEVAL,
            "./paper-sa-2602-10098-vla-jepa-enhancing-vision-language-action-model.md",
        ],
        abbr=[
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("PTQ", "Post-Training Quantization", "训练后量化"),
            ("RTN", "Round-To-Nearest", "最近舍入量化"),
            ("W8A8", "8-bit Weight / 8-bit Activation", "权重与激活均 8 位"),
        ],
    ),
    dict(
        n=3, a="a1", g="世界模型与仿真", short="Uranus", arxiv="2609.24815", open=OPEN, reuse=True
    ),
    dict(
        n=4,
        a="a1",
        g="策略与规划",
        slug="hire-hindsight-reward-editing",
        short="HiRE",
        arxiv="2609.27068",
        title="HiRE: Hindsight Reward Editing for Policy Finetuning",
        code="https://github.com/HiRE-Project/HiRE",
        project="https://hire-project.github.io/",
        open=OPEN,
        open_note="仓库与项目页均可达，项目页链接 GitHub 仓库。",
        tags=[
            "paper",
            "reinforcement-learning",
            "reward",
            "finetuning",
            "visual-reward",
            "manipulation",
        ],
        one="用成功与失败轨迹对照，压低“看似接近目标却会失败”的状态奖励，给策略微调补上控制相关反馈。",
        why="视觉相似不等于任务做对；基于视觉相似度的奖励会把接近但会失败的状态打高分，误导微调。",
        mech="事后（hindsight）对照成功与失败轨迹，编辑视觉奖励：压低失败轨迹中高相似度状态的奖励，使奖励与控制相关。",
        metrics="公众号称在仿真和真机上比较；未给具体数字，本页不引用。",
        concl="视觉奖励需要用结果标签事后校正：成功/失败对照是低成本给奖励补控制相关性的办法。",
        related=[RL, MANIP, "./paper-hire-moma.md"],
        abbr=[
            ("HiRE", "Hindsight Reward Editing", "事后奖励编辑"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    ),
    dict(
        n=5, a="a1", g="VLA 评测与部署", short="THAW-VLA", arxiv="2609.24682", open=OPEN, reuse=True
    ),
    dict(
        n=6,
        a="a1",
        g="策略与规划",
        slug="x-planner-event-structured-task-planning",
        short="X-Planner",
        arxiv="2609.25187",
        title="X-Planner: Event-Structured Task Planning for Embodied Intelligence",
        code="https://github.com/X-Square-Robot/Xplanner",
        project=None,
        open=OPEN,
        open_note="仓库可达（git ls-remote）。",
        tags=["paper", "task-planning", "vla", "event-structure", "long-horizon"],
        one="用事件状态衔接高层指令与下游动作模型，把长任务计划写成可执行事件，并探索连续潜在推理接口。",
        why="长任务计划需要可检查的中间状态；事件结构让规划与动作模型之间的接口显式化。",
        mech="事件状态作为计划与动作模型的接口；另探索连续潜在推理接口。",
        metrics="公众号提示：真机结果来自事件模式的完整系统，离线文本评分和任务进度不能直接当作成功率。",
        concl="评价规划器要区分离线文本评分、任务进度与真机成功率，三者不可互换。",
        related=[VLA, MANIP, LOCOMANIP],
        abbr=[("VLA", "Vision-Language-Action", "视觉–语言–动作策略")],
    ),
    dict(
        n=7, a="a1", g="世界模型与仿真", short="H2RBench", arxiv="2609.24778", open=OPEN, reuse=True
    ),
    dict(
        n=8,
        a="a1",
        g="世界模型与仿真",
        slug="d-jepa-decision-aligned-latent-world-model",
        short="D-JEPA",
        arxiv="2609.24749",
        title="D-JEPA: A Decision-Aligned Latent World Model",
        code="https://github.com/NEBULIS-Lab/D-JEPA",
        project="https://nebulis-lab.com/D-JEPA/",
        open=OPEN,
        open_note="仓库可达（git ls-remote），项目页 HTTP 200。",
        tags=["paper", "world-model", "jepa", "latent-planning", "decision-alignment"],
        one="用实际执行结果学习候选未来之间的关系，修正潜在距离与任务成败不一致的问题，让世界模型选对能成功的动作。",
        why="预测接近目标的潜在状态不等于选对动作；潜在距离与成败可能不一致。",
        mech="以真实执行结果监督候选未来之间的排序关系，使潜在空间与决策对齐。",
        metrics="公众号提示：决策对齐在匹配候选集合下评估，效果须连同动作候选生成方式理解。",
        concl="潜在世界模型的评价要绑定候选动作生成方式；决策对齐的收益不能脱离候选集合解读。",
        related=[GWM, WAM, "./paper-vl-jepa.md"],
        abbr=[("JEPA", "Joint-Embedding Predictive Architecture", "联合嵌入预测架构")],
    ),
    dict(
        n=9,
        a="a1",
        g="VLA 评测与部署",
        slug="gaussian-is-enough-flow-matching-priors",
        short="Gaussian Is Enough",
        arxiv="2609.27070",
        title="The Gaussian Is Enough: Flow-Matching Priors Do Not Help When Fine-Tuning Large Behavior Models",
        code=None,
        project="https://cxu-tri.github.io/non_gaussian_FT/",
        open=OPEN,
        open_note="公众号未给代码链接；项目页列出 TRI-ML/openpi-priors 与 ToyotaResearchInstitute/lbm_eval，二者均 git ls-remote 可达。",
        tags=["paper", "behavior-model", "flow-matching", "finetuning", "prior"],
        one="在多种行为模型上比较高斯与非高斯流匹配先验，发现后者在所测微调设置中未稳定获益。",
        why="微调大型行为模型时，换噪声先验常被当作改进手段；该工作给出否定性证据，提示先检查编码器训练。",
        mech="系统比较高斯与非高斯先验在多种行为模型上的微调表现。",
        metrics="公众号称非高斯先验未稳定获益；极少数据等条件仍留有例外空间，未给具体数字。",
        concl="微调行为模型时不必急着换先验：编码器训练更值得优先检查。",
        related=[FM, IL, VLA],
        abbr=[
            ("LBM", "Large Behavior Model", "大型行为模型"),
            ("FM", "Flow Matching", "流匹配生成"),
        ],
    ),
    dict(
        n=10,
        a="a1",
        g="VLA 评测与部署",
        slug="libero-vpro",
        short="LIBERO-VPro",
        arxiv="2609.24350",
        title="LIBERO-VPro: Benchmarking Closed-Loop Visual Robustness of Robotic Foundation Models",
        code=None,
        project="https://huiqiongli.github.io/LIBERO-VPro/",
        open=NOLINK,
        open_note="项目页 HTTP 200，仅见 “Dataset (soon)” 按钮，未列 GitHub 代码链接。",
        tags=["paper", "benchmark", "vla", "visual-robustness", "libero"],
        one="在执行时扰动视觉证据（缺失、延迟、多视角矛盾、场景变化），做 LIBERO 上的闭环视觉鲁棒性基准。",
        why="只看干净环境成功率会高估鲁棒性；遮住物体仍成功也可能是依赖位置记忆。",
        mech="执行期注入视觉扰动：缺失、延迟、多视角矛盾和场景变化，配合诊断实验。",
        metrics="公众号提示：诊断实验须结合对照阅读；未给具体数字。",
        concl="视觉鲁棒性应在闭环中用扰动测，并用对照实验区分“看见”与“记住位置”。",
        related=[VLA, SIMEVAL, "./libero-benchmark.md"],
        abbr=[("LIBERO", "Lifelong Robot Learning Benchmark", "操作基准套件")],
    ),
    dict(
        n=11,
        a="a1",
        g="数据生成与示范",
        slug="touch2robot",
        short="Touch2Robot",
        arxiv="2609.24660",
        title="Touch2Robot: Robot Touch in the Human Demonstration Loop",
        code=None,
        project="https://touch2robot.github.io/",
        open=NOLINK,
        open_note="项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。",
        tags=["paper", "teleoperation", "tactile", "demonstration", "retargeting"],
        one="把手套触觉、人体动作和物体跟踪用于实时重定向，让示范者在采集时看到目标机器人的接触反馈。",
        why="人与机器人形态差异要在采集阶段就纠正，事后重定向难以补回接触信息。",
        mech="手套触觉 + 人体动作 + 物体跟踪 → 实时重定向，并把目标机器人的接触反馈回显给示范者。",
        metrics="公众号称采集阶段纠正形态差异改善真机回放；任务完成度不能混称成功率。",
        concl="示范采集回路里加入机器人侧接触反馈，是比事后重定向更早的形态对齐手段。",
        related=[TELEOP, TACT, RETARGET],
        abbr=[("Retargeting", "Motion Retargeting", "跨形态动作重定向")],
    ),
    dict(
        n=12,
        a="a1",
        g="数据生成与示范",
        slug="arstag",
        short="ARSTAG",
        arxiv="2609.24563",
        title="ARSTAG: An Agentic Real2Sim2Real System for Task-Specific Robot Data Generation",
        code="https://github.com/boweili666/ARSTAG",
        project="https://boweili666.github.io/ARSTAG/",
        open=OPEN,
        open_note="项目页链接的仓库可达（git ls-remote）。",
        tags=["paper", "real2sim2real", "data-generation", "agentic", "manipulation"],
        one="协调场景重建、专家生成与数据扩增，把单张图像和任务指令转成仿真训练示范。",
        why="围绕任务构建仿真可减少逐任务的工程量，是单图到示范的自动化路径。",
        mech="单张图像 + 指令 → 场景重建 → 专家示范生成 → 数据扩增，由智能体协调。",
        metrics="公众号称真机验证覆盖抓取、放置和堆叠，仍有初始姿态限制；未给具体数字。",
        concl="Real2Sim2Real 数据生成适合任务窄而重复的场景；初始姿态范围是主要边界。",
        related=[MANIP, S2R, IL],
        abbr=[("Real2Sim2Real", "Real-to-Sim-to-Real", "真实→仿真→真实流程")],
    ),
    dict(
        n=13,
        a="a1",
        g="灵巧与触觉",
        slug="contact-anchored-retargeting-residual-dexterous",
        short="Contact-Anchored Retargeting",
        arxiv="2609.24093",
        title="Dexterous Robot Manipulation from Human Demonstrations via Contact-Anchored Retargeting and Residual Policy Learning",
        code="https://github.com/DexGEM-Lab/real2sim2real",
        project=None,
        open=OPEN,
        open_note="仓库可达（git ls-remote）。",
        tags=["paper", "dexterous", "retargeting", "residual-policy", "human-demonstration"],
        one="先用物理残差学习补全人类示范接触，再按接触位置和顺序重定向到机器人手，最后修复动力学可行性。",
        why="跨手形态迁移要先保留接触结构，而不是匹配关节。",
        mech="残差策略补全接触 → 按接触位置与顺序重定向 → 动力学可行性修复。",
        metrics="公众号称覆盖四种手形态与真机双手任务；训练仍依赖仿真。",
        concl="接触锚定是跨手形态迁移的稳定中间表示。",
        related=[DEX, RETARGET, MANIP],
        abbr=[("Retargeting", "Motion Retargeting", "跨形态动作重定向")],
    ),
    dict(
        n=14,
        a="a1",
        g="策略与规划",
        slug="packlab",
        short="PackLab",
        arxiv="2609.23784",
        title="PackLab: A Comprehensive Framework for Developing, Training, and Evaluating MLLMs in Robotic Bin Packing",
        code="https://github.com/Correr-Zhou/PackLab",
        project=None,
        open=OPEN,
        open_note="仓库可达（git ls-remote）。",
        tags=["paper", "bin-packing", "mllm", "benchmark", "simulation"],
        one="把仿真平台、专用多模态模型和分难度评测连在一起，让模型随箱内状态更新选择物体与落点。",
        why="装箱不仅看单次落点，还要看后续空间；装入体积与紧凑度共同评价布局。",
        mech="仿真平台 + 专用 MLLM + 分难度评测，随箱内状态更新决策。",
        metrics="公众号称主要证据来自物理仿真；未给具体数字。",
        concl="序贯装箱评测应同时看体积与紧凑度，且证据目前停留在仿真。",
        related=[MANIP, SIMEVAL],
        abbr=[("MLLM", "Multimodal Large Language Model", "多模态大语言模型")],
    ),
    dict(
        n=15,
        a="a1",
        g="VLA 评测与部署",
        slug="vla-simd",
        short="vla.simd",
        arxiv="2609.24274",
        title="vla.simd: Efficient CPU Inference for Language-Conditioned Manipulation",
        code=None,
        project="https://vla-simd.github.io/",
        open=NOLINK,
        open_note="项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。",
        tags=["paper", "vla", "cpu-inference", "simd", "deployment"],
        one="用 SIMD 内核与计算复用加速 CPU 推理，并把查询延迟和动作块长度一起分析。",
        why="CPU 部署先算动作供给，再看反馈频率；动作供给率不等于闭环反馈频率。",
        mech="SIMD 内核 + 计算复用；联合分析查询延迟与动作块长度。",
        metrics="公众号提示：树莓派部署数据需连同热负载和执行方式阅读；未给具体数字。",
        concl="评估 CPU 端 VLA 要把动作供给率与闭环反馈频率分开。",
        related=[VLA, "./paper-vlaquantbench.md"],
        abbr=[("SIMD", "Single Instruction Multiple Data", "单指令多数据向量化")],
    ),
    dict(
        n=16,
        a="a1",
        g="策略与规划",
        slug="visuomotor-robotic-pruning-orchards",
        short="Visuomotor Pruning",
        arxiv="2609.24906",
        title="Visuomotor Robotic Pruning in Planar Orchards Using Hybrid Reinforcement Learning",
        code="https://github.com/OSUrobotics/lpy_treesim",
        project=None,
        open=OPEN,
        open_note="仓库可达（git ls-remote）；仓库名为 lpy_treesim（树木仿真），是否含完整训练代码未核实。",
        tags=["paper", "agriculture", "reinforcement-learning", "visuomotor", "pruning"],
        one="用腕部相机光流驱动闭环控制，混合离线示范与在线仿真学习，在平面整形果树上做修枝。",
        why="不必完整三维重建也能闭环引导修枝；纯仿真训练后可迁移实地。",
        mech="腕部相机光流 → 闭环控制；离线示范 + 在线仿真混合 RL。",
        metrics="公众号提示：范围是平面整形果树与预先指定的切点；未给具体数字。",
        concl="光流驱动闭环可替代完整重建，但适用范围限于平面整形果树。",
        related=[RL, MANIP, S2R],
        abbr=[("RL", "Reinforcement Learning", "强化学习")],
    ),
    dict(
        n=17,
        a="a1",
        g="数据生成与示范",
        slug="glide-beyond-human-demonstration",
        short="GLIDE",
        arxiv="2609.24996",
        title="Learning Beyond What Humans Can Demonstrate",
        code=None,
        project="https://guardrail-policy.github.io/",
        open=NOLINK,
        open_note="项目页 HTTP 200；页面仅含 nerfies 模板作者链接，无项目代码。",
        tags=["paper", "teleoperation", "guardrail", "failure-modes", "imitation-learning"],
        one="从任务描述和遥操作代码推断失败模式，把规则护栏插入采集与部署过程，帮助完成难以直接示范的动作。",
        why="有些动作人难以演示；执行护栏可约束采集与部署，弥补示范能力缺口。",
        mech="任务描述 + 遥操作代码 → 推断失败模式 → 规则护栏。",
        metrics="公众号称证据来自三个具体任务，应连同护栏约束一起理解策略表现。",
        concl="护栏让“人演示不了”的任务也能采集，但策略表现不应脱离护栏单独评价。",
        related=[TELEOP, IL, MANIP],
        abbr=[("Guardrail", "Execution Guardrail", "执行护栏：采集与部署期的规则约束")],
    ),
    dict(
        n=18,
        a="a1",
        g="策略与规划",
        slug="migu-multimodal-instruction-grounding",
        short="MIGU",
        arxiv="2609.24995",
        title="MIGU: Multimodal Instruction Grounding under Uncertainty for Manipulation Planning",
        code=None,
        project="https://multimodal-instruction.github.io/",
        open=TBD,
        open_note="项目页 HTTP 200，含 coming soon 字样，未列 GitHub。",
        tags=[
            "paper",
            "instruction-grounding",
            "pointing-gesture",
            "uncertainty",
            "mobile-manipulation",
        ],
        one="融合语言语义与指向手势的三维几何不确定性，先形成目标信念，再决定执行或澄清。",
        why="语言加手势也需要知道何时追问；不确定性显式建模连接移动操作与桌面规划。",
        mech="语言语义 + 手势三维几何不确定性 → 目标信念 → 执行/澄清决策。",
        metrics="公众号提示：当前决策只允许单轮追问；未给具体数字。",
        concl="显式的目标信念让“执行还是追问”成为可评估的决策，当前仅支持单轮澄清。",
        related=[MANIP, LOCOMANIP, VLA],
        abbr=[
            (
                "MIGU",
                "Multimodal Instruction Grounding under Uncertainty",
                "不确定性下的多模态指令落地",
            )
        ],
    ),
    dict(
        n=19,
        a="a1",
        g="VLA 评测与部署",
        slug="robomp-dinov2",
        short="RoboMP-DINOv2",
        arxiv="2609.25506",
        title="RoboMP-DINOv2: Prompts, Not Filters for Robust Robot Manipulation",
        code="https://github.com/han20192019/RoboMP_DINOv2",
        project=None,
        open=OPEN,
        open_note="仓库可达（git ls-remote）。",
        tags=["paper", "dinov2", "visual-prompt", "robust-manipulation", "mask"],
        one="在完整图像特征上加入区域提示，而不是用硬掩码过滤，避免丢掉障碍与背景线索。",
        why="硬掩码会丢掉障碍和背景线索；区域提示保留完整场景。",
        mech="完整图像特征 + 区域提示（空间提示），替代硬掩码。",
        metrics="公众号称空间、杂乱和颜色迁移实验均在仿真中完成，颜色增强版本应单独看待。",
        concl="用提示而非过滤注入空间先验，但证据目前全在仿真。",
        related=[MANIP, VLA, "./paper-dinov2.md"],
        abbr=[("DINOv2", "Self-DIstillation with NO labels v2", "自监督视觉特征")],
    ),
    dict(
        n=20,
        a="a1",
        g="灵巧与触觉",
        slug="berkeley-quad-hand",
        short="Berkeley QUAD Hand",
        arxiv="2609.27240",
        title="A Quasi-Direct-Drive Underactuated Asymmetric Hand for Dexterous and Efficient Grasping and Manipulation",
        code=None,
        project="https://benudavis.github.io/berkeley-quadhand/",
        open=NOLINK,
        open_note="项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。",
        tags=["paper", "dexterous-hand", "hardware", "underactuated", "quasi-direct-drive"],
        one="用非对称驱动和柔顺耦合让不同手指分工，兼顾灵巧与持续握力，并控制体积与发热。",
        why="灵巧与持续握力难以由同一指实现，指分工是硬件层面的折中。",
        mech="准直驱 + 欠驱动 + 非对称手指分工 + 柔顺耦合。",
        metrics="公众号称硬件实现了 Feix 分类中 29 种抓型，分母为 33 种。",
        concl="29/33 的抓型覆盖说明该硬件范围广但并非全覆盖。",
        related=[DEX, MANIP],
        abbr=[("QDD", "Quasi-Direct-Drive", "准直驱")],
    ),
    dict(n=21, a="a1", g="人形控制", short="HOTICE", arxiv="2609.25363", open=TBD, reuse=True),
    dict(n=22, a="a1", g="人形控制", short="DeCap", arxiv="2609.24552", open=OPEN, reuse=True),
    dict(
        n=23,
        a="a1",
        g="VLA 评测与部署",
        slug="topology-informed-visual-prompting-vla",
        short="Topology-VLA",
        arxiv="2609.23944",
        title="Topology-Informed Visual Prompting For Vision Language Action Policies",
        code=None,
        project="https://topology-vla.github.io/",
        open=NOLINK,
        open_note="项目页 HTTP 200；页面仅含学术模板链接，无项目代码。",
        tags=["paper", "vla", "topology", "visual-prompt", "obstacle-avoidance"],
        one="用拓扑签名区分外观相似却需不同绕行的状态，借仿真规划扩增示范，再让 VLM 生成视觉路径提示。",
        why="外观相似的状态可能要求不同绕行方向，拓扑信息补上了 VLA 缺的路线语义。",
        mech="拓扑签名 → 仿真规划扩增示范 → VLM 生成视觉路径提示。",
        metrics="公众号提示：训练时几何特权信息与部署时相机引导应分开理解。",
        concl="拓扑提示让 VLA 的避障路线可显式表达，需区分训练期特权信息与部署期输入。",
        related=[VLA, MANIP],
        abbr=[("VLM", "Vision-Language Model", "视觉语言模型")],
    ),
    dict(n=24, a="a1", g="人形控制", short="PRIMO", arxiv="2609.23610", open=OPEN, reuse=True),
    dict(
        n=25,
        a="a1",
        g="灵巧与触觉",
        slug="spectrobot-single-point-tactile",
        short="SpectRobot",
        arxiv="2609.24621",
        title="Learning tactile perception from high-bandwidth single-point sensing",
        code=None,
        resource="https://github.com/spectrobot-project",
        project="https://huggingface.co/jogarulfop",
        open=NOLINK,
        open_note="公众号给的 GitHub 为组织主页（本环境访问返回 403，无法核实具体仓库）；项目页为 Hugging Face 个人主页，未见具体项目链接。",
        tags=["paper", "tactile", "spectrogram", "single-point-sensing", "vision-encoder"],
        one="把高带宽单点传感信号转成时频图，让视觉编码器读取接触历史。",
        why="时间与频率信息能帮助识别遮挡盒内物体；用现成视觉编码器读取触觉降低了建模成本。",
        mech="高带宽单点信号 → 时频图 → 视觉编码器。",
        metrics="公众号称实验也比较不同传感技术，尚不代表通用灵巧操作；未给具体数字。",
        concl="单点高带宽触觉经时频图可复用视觉编码器，但目前不是通用灵巧操作方案。",
        related=[TACT, DEX],
        abbr=[("Spectrogram", "Time-Frequency Map", "时频图")],
    ),
    dict(
        n=26,
        a="a1",
        g="灵巧与触觉",
        slug="norm2tex",
        short="Norm2Tex",
        arxiv="2609.25398",
        title="Norm2Tex: Augmenting Visuo-Tactile Simulations with Texture",
        code="https://github.com/utn-air/norm2tex",
        project="https://utn-air.github.io/norm2tex",
        open=TBD,
        open_note="项目页链接的 github.com/utn-air/norm2tex 经 git ls-remote 不可读（仓库不存在或未公开），暂按待发布处理。",
        tags=["paper", "visuo-tactile", "simulation", "texture", "sim2real"],
        one="把法线纹理转成高度信息，在触觉渲染前修改深度图，为现有模拟器增加高频表面细节。",
        why="现有视触觉模拟器缺高频表面纹理，影响材料分类等任务的迁移。",
        mech="法线纹理 → 高度信息 → 修改深度图 → 触觉渲染。",
        metrics="公众号称通过材料分类与控制任务检查纹理信息能否迁移到真机；未给具体数字。",
        concl="在深度图层面补纹理是不改模拟器内核的低成本增强。",
        related=[TACT, S2R],
        abbr=[("Visuo-tactile", "Vision-based Tactile", "视觉型触觉传感")],
    ),
]


def _yaml_list(items: list[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _ax_file(p: dict) -> str:
    return f"{p['slug'].replace('-', '_')}_arxiv_{p['arxiv'].replace('.', '_')}.md"


def _repo_file(p: dict) -> str:
    return f"{p['slug'].replace('-', '_')}.md"


def _site_file(p: dict) -> str:
    return f"{p['slug']}.md"


def _paper_source(p: dict) -> str:
    a = ARTICLES[p["a"]]
    lines = [
        f"# {p['short']}（arXiv:{p['arxiv']}）",
        "",
        "> 来源归档（paper）",
        "",
        f"- **标题：** {p['title']}",
        "- **类型：** paper",
        f"- **arXiv：** <https://arxiv.org/abs/{p['arxiv']}>",
        f"- **PDF：** <https://arxiv.org/pdf/{p['arxiv']}>",
    ]
    if p.get("project"):
        lines.append(f"- **项目页：** <{p['project']}>")
    if p.get("code"):
        lines.append(f"- **代码：** <{p['code']}>")
    if p.get("resource"):
        lines.append(f"- **资源：** <{p['resource']}>")
    lines += [
        f"- **入库日期：** {TODAY}",
        f"- **线索来源：** [{a['title']}](../blogs/{blog_name(p['a'])})（具身智能小站，{a['date']}）",
        f"- **一句话说明：** {p['one']}",
        "",
        "## 开源状态",
        "",
        f"- **{p['open']}**（步骤 2.5 核查，{TODAY}）：{p['open_note']}",
        "",
        "## 核心摘录（公众号转述，未核验原文表格）",
        "",
        f"- {p['why']}",
        f"- {p['mech']}",
        f"- 指标：{p['metrics']}",
        "",
        "## 对 wiki 的映射",
        "",
        f"- [paper-{p['slug']}](../../wiki/entities/paper-{p['slug']}.md)",
        f"- [{a['map_title']}](../../wiki/overview/{a['map']})",
    ]
    return "\n".join(lines) + "\n"


def _repo(p: dict) -> str:
    return f"""# {p["short"]} 官方仓库

> 来源归档（repo）

- **标题：** {p["title"]}
- **类型：** repo
- **链接：** {p["code"]}
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **入库日期：** {TODAY}
- **一句话说明：** {p["one"]}
- **沉淀到 wiki：** [`wiki/entities/paper-{p["slug"]}.md`](../../wiki/entities/paper-{p["slug"]}.md)

## 开源状态

- **{p["open"]}**：{p["open_note"]} 仓库内容以 README 与 release 为准（本次仅用 `git ls-remote` 核实可达，未逐个阅读 README）。
"""


def _site(p: dict) -> str:
    code = f"- **代码：** <{p['code']}>\n" if p.get("code") else ""
    return f"""# {p["short"]} 项目页

> 来源归档（site）

- **标题：** {p["title"]}
- **类型：** site
- **链接：** {p["project"]}
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
{code}- **入库日期：** {TODAY}
- **一句话说明：** {p["one"]}
- **沉淀到 wiki：** [`wiki/entities/paper-{p["slug"]}.md`](../../wiki/entities/paper-{p["slug"]}.md)

## 开源状态

- **{p["open"]}**（步骤 2.5 核查，{TODAY}）：{p["open_note"]}
"""


def _seq(p: dict) -> str:
    if p["open"] in (OPEN, PART) and p.get("code"):
        repo = p["code"].split("github.com/")[-1]
        return f"""## 源码运行时序图

以下为按仓库入口（`{repo}`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as {repo}
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 {p["short"]} 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```
"""
    if p["open"] == PART:
        return f"""## 源码运行时序图

**部分开源** — {p["open_note"]} 待可运行代码公开后补图。
"""
    return f"""## 源码运行时序图

**不适用（{p["open"]}）** — {p["open_note"]} 不编造运行脚本；代码公开后再补。
"""


def _entity(p: dict) -> str:
    ax = p["arxiv"]
    slug = p["slug"]
    a = ARTICLES[p["a"]]
    srcs = [f"../../sources/papers/{_ax_file(p)}"]
    if p.get("code"):
        srcs.append(f"../../sources/repos/{_repo_file(p)}")
    if p.get("project"):
        srcs.append(f"../../sources/sites/{_site_file(p)}")
    srcs.append(f"../../sources/blogs/{blog_name(p['a'])}")
    fm_extra = ""
    if p.get("project"):
        fm_extra += f"project: {p['project']}\n"
    if p.get("code"):
        fm_extra += f"code: {p['code']}\n"
    abbrev = "\n".join(f"| {x} | {y} | {z} |" for x, y, z in p["abbr"])
    links = []
    if p.get("project"):
        links.append(f"[项目页]({p['project']})")
    if p.get("code"):
        links.append(f"[代码]({p['code']})")
    if p.get("resource"):
        links.append(f"[资源]({p['resource']})")
    link_txt = ("，" + "，".join(links)) if links else ""
    related = p["related"] + [f"../overview/{a['map']}"]
    rel_lines = "\n".join(f"- [{r.split('/')[-1].replace('.md', '')}]({r})" for r in related)
    reads = [f"- [arXiv PDF](https://arxiv.org/pdf/{ax})"]
    if p.get("project"):
        reads.append(f"- [项目页]({p['project']})")
    if p.get("code"):
        reads.append(f"- [{p['code']}]({p['code']})")
    if p.get("resource"):
        reads.append(f"- [{p['resource']}]({p['resource']})")
    summary = f"{p['short']}（arXiv:{ax}）：{p['one']}".replace('"', "'")
    return f"""---
type: entity
project_id: {slug}
tags:
{_yaml_list(p["tags"], 2)}
status: complete
updated: {TODAY}
arxiv: "{ax}"
{fm_extra}related:
{_yaml_list(related, 2)}
sources:
{_yaml_list(srcs, 2)}
summary: "{summary}"
---

# {p["short"]}（arXiv:{ax}）

**{p["short"]}**（*{p["title"]}*，[arXiv:{ax}](https://arxiv.org/abs/{ax}){link_txt}）来自 [具身智能小站 {a["date"]} 盘点](../../sources/blogs/{blog_name(p["a"])})，属「{p["g"]}」组。

## 一句话定义

**{p["one"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

- {p["why"]}
- 开源结论：**{p["open"]}**（步骤 2.5，{TODAY}）。{p["open_note"]}
- 同期同组工作见 [{a["map_title"]}](../overview/{a["map"]})。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [{ax}](https://arxiv.org/abs/{ax}) |
| **开源** | **{p["open"]}** |
| **要点** | {p["mech"]} |
| **文内指标** | {p["metrics"]} |

{_seq(p)}
## 实验与评测

- {p["metrics"]}
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [{a["map_title"]}](../overview/{a["map"]})；同 arXiv 不重复造页。

## 结论

**{p["concl"]}**

1. 开源边界：**{p["open"]}** — {p["open_note"]}
2. 核心机制：{p["mech"]}
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

{rel_lines}

## 参考来源

- [{_ax_file(p)}](../../sources/papers/{_ax_file(p)})
- [{blog_name(p["a"])}](../../sources/blogs/{blog_name(p["a"])})
- [arXiv:{ax}](https://arxiv.org/abs/{ax})

## 推荐继续阅读

{chr(10).join(reads)}
"""


def _real(a: str) -> list[dict]:
    return [p for p in PAPERS if p["a"] == a]


def _wiki_of(p: dict) -> str:
    return REUSE[p["arxiv"]] if p.get("reuse") else f"paper-{p['slug']}"


def _rows(a: str) -> list[tuple[str, str, str, str, str, str, bool]]:
    rows = []
    for p in _real(a):
        w = _wiki_of(p)
        reuse = bool(p.get("reuse"))
        tag = "（**复用**）" if reuse else ""
        tag2 = "（复用）" if reuse else ""
        rows.append(
            (
                p["g"],
                p["short"],
                p["arxiv"],
                p["open"],
                f"[{w}](../../wiki/entities/{w}.md){tag}",
                f"[{w}](../entities/{w}.md){tag2}",
                reuse,
            )
        )
    return rows


def _blog(a: str) -> str:
    A = ARTICLES[a]
    rows = _rows(a)
    new = sum(1 for r in rows if not r[6])
    body = "\n".join(
        f"| {i:02d} | {r[0]} | {r[1]} | [{r[2]}](https://arxiv.org/abs/{r[2]}) | **{r[3]}** | {r[4]} |"
        for i, r in enumerate(rows, 1)
    )
    return f"""# {A["title"]}

> 来源归档（blog / 微信公众号）

- **标题：** {A["title"]}
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** {A["url"]}
- **发表日期：** {A["date"]}
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_{A["key"]}.md`](../raw/wechat_embodied_station_{A["key"]}.md)
- **一句话说明：** {A["view"]} 共 {A["n"]} 篇；**{A["n"]}/{A["n"]} 独立详情节点**（本 ingest **新建 {new}**、**复用 {len(rows) - new}**）。

## {A["n"]} 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
{body}

## 对 wiki 的映射

- **{A["n"]}/{A["n"]} 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[{A["map_title"]}](../../wiki/overview/{A["map"]})
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（{TODAY}）
- [x] {A["n"]} 篇独立节点（{new} 新建 / {len(rows) - new} 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
"""


def _map(a: str) -> str:
    A = ARTICLES[a]
    rows = _rows(a)
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r[0], []).append(r)
    sections = []
    for g, rs in groups.items():
        tbl = "\n".join(f"| {r[1]} | {r[5]} | {r[3]} |" for r in rs)
        sections.append(
            f"### {g}（{len(rs)} 篇）\n\n| 论文 | 节点 | 开源 |\n|------|------|------|\n{tbl}\n"
        )
    related = [f"../entities/{_wiki_of(p)}.md" for p in _real(a)[:3]] + [VLA, WAM]
    flow_nodes = "\n".join(
        f'  G{i}["{g}（{len(rs)}）"] --> GOAL' for i, (g, rs) in enumerate(groups.items())
    )
    return f"""---
type: overview
tags: [overview, survey, embodied-ai, vla, data-generation, technology-map, wechat]
status: complete
updated: {TODAY}
related:
{_yaml_list(related, 2)}
sources:
  - ../../sources/blogs/{blog_name(a)}
summary: "{A["map_summary"]}"
---

# {A["map_title"]}

> **本页定位**：为 [具身智能小站 · {A["title"]}]({A["url"]})（{A["date"]}）提供按研究方向组织的阅读坐标；{A["n"]} 篇全部有独立详情节点。

## 一句话观点

**{A["view"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WAM | World Action Model | 世界–动作联合模型 |
| PTQ | Post-Training Quantization | 训练后量化 |
| JEPA | Joint-Embedding Predictive Architecture | 联合嵌入预测架构 |

## 为什么单独做这张地图

- 公众号一次列出 {A["n"]} 篇（1 深读 + 2 跟进 + 23 扫读）、方向跨度大，需要横切面索引；每篇只保留一个判断，细节在各节点。
- 节点内的数字为公众号转述，未核验原文表格；开源结论为入库日核查结果，见各节点「为什么重要」。

## 流程总览

```mermaid
flowchart TB
{flow_nodes}
  GOAL[可部署具身系统]
```

## 分组索引

{chr(10).join(sections)}
## 关联页面

- [VLA](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)

## 参考来源

- [{blog_name(a)}](../../sources/blogs/{blog_name(a)})

## 推荐继续阅读

- [{rows[0][1]}]({rows[0][5].split("](")[1].split(")")[0]})
- [{rows[1][1]}]({rows[1][5].split("](")[1].split(")")[0]})
"""


def _patch_reuse(a: str) -> None:
    """复用节点：补本期博客来源与地图关联（幂等）。"""
    blog = f"  - ../../sources/blogs/{blog_name(a)}\n"
    mapl = f"  - ../overview/{ARTICLES[a]['map']}\n"
    for w in REUSE.values():
        path = ROOT / "wiki/entities" / f"{w}.md"
        text = path.read_text(encoding="utf-8")
        head, sep, body = text.partition("\n---\n")
        if not sep:
            raise SystemExit(f"frontmatter 缺失: {path}")
        lines = head.split("\n")

        def _append(key: str, item: str) -> None:
            if item.strip() in (l.strip() for l in lines):
                return
            try:
                i = next(i for i, l in enumerate(lines) if l.startswith(f"{key}:"))
            except StopIteration:
                lines.append(f"{key}:")
                lines.append(item.rstrip("\n"))
                return
            if lines[i].strip() != f"{key}:":
                raise SystemExit(f"{path}: {key} 为行内写法，需手工补")
            j = i + 1
            while j < len(lines) and lines[j].startswith("  "):
                j += 1
            lines.insert(j, item.rstrip("\n"))

        _append("sources", blog)
        _append("related", mapl)
        lines = [f"updated: {TODAY}" if l.startswith("updated:") else l for l in lines]
        path.write_text("\n".join(lines) + sep + body, encoding="utf-8")


def main() -> None:
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    new = [p for p in PAPERS if not p.get("reuse")]
    seen = set()
    for p in PAPERS:
        if p["arxiv"] in seen:
            raise SystemExit(f"重复 arXiv: {p['arxiv']}")
        seen.add(p["arxiv"])
        if p.get("reuse"):
            if not (ROOT / "wiki/entities" / f"{REUSE[p['arxiv']]}.md").exists():
                raise SystemExit(f"复用节点不存在: {p['arxiv']}")
            continue
        if p["arxiv"] in REUSE:
            raise SystemExit(f"应标 reuse: {p['arxiv']}")
        ent = ROOT / "wiki/entities" / f"paper-{p['slug']}.md"
        if ent.exists():
            raise SystemExit(f"节点已存在: {ent}")
        for rel in p["related"]:
            tgt = (ROOT / "wiki/entities" / rel).resolve()
            if not tgt.exists() and not rel.startswith("./paper-"):
                raise SystemExit(f"关联页不存在: {rel} ({p['slug']})")
    for a, A in ARTICLES.items():
        if raw_dir:
            src = raw_dir / f"{a}.txt"
            dst = ROOT / "sources/raw" / f"wechat_embodied_station_{A['key']}.md"
            dst.write_text(
                f"# {A['title']}\n\n> 原始抓取（{TODAY}，`scripts/fetch_wechat_article.py`），仅供溯源；来源：{A['url']}\n\n"
                + src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        (ROOT / "sources/blogs" / blog_name(a)).write_text(_blog(a), encoding="utf-8")
        (ROOT / "wiki/overview" / str(A["map"])).write_text(_map(a), encoding="utf-8")
        _patch_reuse(a)
    for p in new:
        (ROOT / "sources/papers" / _ax_file(p)).write_text(_paper_source(p), encoding="utf-8")
        (ROOT / "wiki/entities" / f"paper-{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
        if p.get("code"):
            (ROOT / "sources/repos" / _repo_file(p)).write_text(_repo(p), encoding="utf-8")
        if p.get("project"):
            (ROOT / "sources/sites" / _site_file(p)).write_text(_site(p), encoding="utf-8")
    print(f"Wrote 1 blog, 1 map, {len(new)} new entities, patched {len(REUSE)} reuse nodes")


if __name__ == "__main__":
    main()
