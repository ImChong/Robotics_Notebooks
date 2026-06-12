"""AMP_mjlab training flow explainer — ManimGL (3b1b/manim).

Based on ccrpRepo/AMP_mjlab source:
  scripts/train.py → AmpOnPolicyRunner.learn()
  src/tasks/amp_loco/config/g1/{__init__,env_cfgs,rl_cfg}.py
  rsl_rl/runners/amp_on_policy_runner.py
"""

from __future__ import annotations

from manimlib import *

CN = "WenQuanYi Micro Hei"
MONO = "DejaVu Sans Mono"
BLUE = "#3B82F6"
GREEN = "#22C55E"
ORANGE = "#F97316"
PURPLE = "#A855F7"
RED = "#EF4444"
GRAY = "#94A3B8"
BG = "#0F172A"

# ManimGL default frame ≈ 14.22 × 8.0; keep content inside safe margins.
MAX_CONTENT_W = 12.0
MAX_CONTENT_H = 5.2
TITLE_BUFF = 0.55
SECTION_FS = 34


def fit_in_frame(mob: Mobject, max_w: float = MAX_CONTENT_W, max_h: float = MAX_CONTENT_H) -> Mobject:
    if mob.get_width() > max_w:
        mob.set_width(max_w)
    if mob.get_height() > max_h:
        mob.set_height(max_h)
    return mob


def section_title(text: str) -> Text:
    t = Text(text, font=CN, font_size=SECTION_FS, color=WHITE)
    t.to_edge(UP, buff=0.45)
    return t


def body_text(text: str, size: int = 20, color=WHITE, mono: bool = False) -> Text:
    return Text(text, font=MONO if mono else CN, font_size=size, color=color)


def code_line(text: str, size: int = 18) -> Text:
    return body_text(text, size=size, color=GRAY, mono=True)


def multiline_text(lines: list[str], size: int = 18, color=WHITE, mono: bool = False) -> VGroup:
    group = VGroup(*[body_text(line, size=size, color=color, mono=mono) for line in lines])
    group.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
    return group


def box_label(
    text: str,
    width: float = 3.0,
    height: float | None = None,
    color: str = BLUE,
    font_size: int = 20,
) -> VGroup:
    lines = text.split("\n")
    label = multiline_text(lines, size=font_size, color=WHITE)
    if height is None:
        height = max(0.72, 0.34 * len(lines) + 0.36)
    rect = RoundedRectangle(
        width=max(width, label.get_width() + 0.5),
        height=max(height, label.get_height() + 0.32),
        corner_radius=0.1,
        stroke_color=color,
        stroke_width=2,
        fill_color=color,
        fill_opacity=0.15,
    )
    label.move_to(rect.get_center())
    return VGroup(rect, label)


def step_box(title: str, detail: str, color: str, width: float = 10.5) -> VGroup:
    lines = detail.split("\n")
    t1 = body_text(title, size=20, color=WHITE, mono=True)
    t2 = multiline_text(lines, size=17, color=GRAY) if len(lines) > 1 else body_text(detail, size=17, color=GRAY)
    inner = VGroup(t1, t2).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
    rect_h = max(0.78, inner.get_height() + 0.28)
    rect = RoundedRectangle(
        width=width,
        height=rect_h,
        corner_radius=0.1,
        stroke_color=color,
        stroke_width=2,
        fill_color=color,
        fill_opacity=0.12,
    )
    inner.move_to(rect.get_center())
    return VGroup(rect, inner)


def place_content_below_title(content: Mobject, title: Mobject, buff: float = 0.38) -> Mobject:
    content.next_to(title, DOWN, buff=buff)
    fit_in_frame(content)
    # If still too low, nudge up slightly.
    if content.get_bottom()[1] < -3.35:
        content.shift(UP * (content.get_bottom()[1] + 3.35))
    return content


class AmpMjlabTrainingFlow(Scene):
    """End-to-end AMP_mjlab training pipeline (code-faithful)."""

    def construct(self) -> None:
        self.camera.background_rgba = list(hex_to_rgb(BG)) + [1.0]
        self.show_intro()
        self.show_entry_and_registration()
        self.show_environment_stack()
        self.show_networks_and_data()
        self.show_single_iteration()
        self.show_rewards_and_amp()
        self.show_training_dynamics()
        self.show_export_and_deploy()
        self.show_outro()

    # ------------------------------------------------------------------
    def show_intro(self) -> None:
        title = body_text("AMP_mjlab 训练流程", size=48)
        subtitle = body_text(
            "Unitree G1 · mjlab + rsl_rl · 统一走跑与跌倒恢复",
            size=24,
            color=GRAY,
        )
        repo = code_line("github.com/ccrpRepo/AMP_mjlab", 22)
        block = VGroup(title, subtitle, repo).arrange(DOWN, buff=0.38)
        fit_in_frame(block, max_w=11.0, max_h=4.0)

        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(subtitle, shift=UP * 0.15), run_time=0.7)
        self.play(FadeIn(repo), run_time=0.5)
        self.wait(1.0)
        self.play(FadeOut(block), run_time=0.7)

    # ------------------------------------------------------------------
    def show_entry_and_registration(self) -> None:
        st = section_title("1. 训练入口与任务注册")
        self.play(Write(st))

        cmd = code_line(
            "python scripts/train.py Unitree-G1-AMP-Flat\n"
            "--env.scene.num-envs=4096",
            17,
        )

        flow = VGroup(
            box_label("main() · tyro 解析任务", 5.5, color=BLUE, font_size=19),
            box_label("launch_training()", 5.0, color=BLUE, font_size=19),
            box_label("run_train()", 4.5, color=BLUE, font_size=19),
            box_label("AMPOnPolicyRunner.learn()", 6.0, color=PURPLE, font_size=19),
        ).arrange(DOWN, buff=0.28)

        reg = multiline_text(
            [
                "register_mjlab_task",
                "Unitree-G1-AMP-Flat / Rough",
                "runner_cls = AMPOnPolicyRunner",
            ],
            size=17,
            color=ORANGE,
            mono=True,
        )

        content = VGroup(cmd, flow, reg).arrange(DOWN, buff=0.38, aligned_edge=LEFT)
        place_content_below_title(content, st, buff=0.42)

        arrows = VGroup(
            *[
                Arrow(flow[i].get_bottom(), flow[i + 1].get_top(), buff=0.06, color=GRAY, stroke_width=2)
                for i in range(len(flow) - 1)
            ]
        )

        self.play(Write(cmd), run_time=0.9)
        self.play(LaggedStartMap(FadeIn, flow, shift=DOWN * 0.1, lag_ratio=0.12), run_time=1.0)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.15), run_time=0.7)
        self.play(FadeIn(reg, shift=UP * 0.1), run_time=0.7)
        self.wait(1.1)
        self.play(FadeOut(VGroup(st, content, arrows)), run_time=0.6)

    # ------------------------------------------------------------------
    def show_environment_stack(self) -> None:
        st = section_title("2. mjlab 并行仿真环境")
        self.play(Write(st))

        center = box_label("ManagerBasedRlEnv\n4096 × G1", 5.0, 1.0, GREEN, 20)

        params = multiline_text(
            [
                "timestep = 0.005 s · decimation = 4 → 50 Hz",
                "episode_length_s = 20",
            ],
            size=17,
            color=GRAY,
            mono=True,
        )

        motions = multiline_text(
            [
                "WalkandRun/ 参考动作 · Recovery/ 起身",
                "delay_reset 40% env · max_delay_steps = 250",
            ],
            size=17,
            color=ORANGE,
        )

        events = VGroup(
            box_label("reset_from_motion", 3.4, 0.72, ORANGE, 17),
            box_label("push_robot", 2.8, 0.72, ORANGE, 17),
            box_label("摩擦/质心 DR", 2.6, 0.72, ORANGE, 17),
        ).arrange(RIGHT, buff=0.22)
        fit_in_frame(events, max_w=11.5, max_h=1.2)

        content = VGroup(center, params, motions, events).arrange(DOWN, buff=0.32)
        place_content_below_title(content, st, buff=0.42)

        self.play(FadeIn(center), run_time=0.6)
        self.play(FadeIn(params), FadeIn(motions), run_time=0.7)
        self.play(FadeIn(events, shift=UP * 0.1), run_time=0.7)
        self.wait(1.2)
        self.play(FadeOut(VGroup(st, content)), run_time=0.6)

    # ------------------------------------------------------------------
    def show_networks_and_data(self) -> None:
        st = section_title("3. 观测、动作与网络")
        self.play(Write(st))

        actor_block = VGroup(
            box_label("Actor-Critic", 4.2, 0.85, BLUE, 19),
            multiline_text(
                ["输入 obs 384 维（4 帧 × 96）", "输出 actions 29 维 · scale=0.25 PD"],
                size=17,
            ),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)

        disc_block = VGroup(
            box_label("AMP Discriminator", 4.6, 0.85, PURPLE, 19),
            multiline_text(
                ["amp obs：13 body 部位", "风格奖励 coef=0.1 · lerp=0.75"],
                size=17,
            ),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)

        obs_terms = multiline_text(
            [
                "base_ang_vel · projected_gravity · twist command",
                "joint_pos/vel · last_action · history_ordering=time",
            ],
            size=15,
            color=GRAY,
            mono=True,
        )

        hidden = code_line("Actor [512,256,128] · Disc [1024,512,256] · AMPPPO", 16)

        content = VGroup(actor_block, disc_block, obs_terms, hidden).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        place_content_below_title(content, st, buff=0.4)

        self.play(FadeIn(actor_block), run_time=0.7)
        self.play(FadeIn(disc_block), run_time=0.7)
        self.play(FadeIn(obs_terms), FadeIn(hidden), run_time=0.7)
        self.wait(1.1)
        self.play(FadeOut(VGroup(st, content)), run_time=0.6)

    # ------------------------------------------------------------------
    def show_single_iteration(self) -> None:
        st = section_title("4. 单次训练迭代")
        self.play(Write(st))

        sub = body_text("AmpOnPolicyRunner.learn()", size=18, color=GRAY, mono=True)
        sub.next_to(st, DOWN, buff=0.18)

        steps = [
            ("Rollout", "24 steps × 4096 envs", BLUE),
            ("alg.act()", "采样 actions", BLUE),
            ("env.step()", "MuJoCo 物理步进", GREEN),
            ("predict_amp_reward", "任务奖励 + 风格奖励", PURPLE),
            ("compute_returns", "GAE λ=0.95 · γ=0.99", BLUE),
            ("alg.update()", "PPO + 判别器 · 5 epochs", PURPLE),
        ]
        boxes = VGroup(*[step_box(t, d, c) for t, d, c in steps]).arrange(DOWN, buff=0.16)
        note = body_text("每 iter 采样 ≈ 98,304 transitions（4096 × 24）", size=18, color=ORANGE)

        content = VGroup(boxes, note).arrange(DOWN, buff=0.28)
        content.next_to(sub, DOWN, buff=0.28)
        fit_in_frame(content, max_w=11.0, max_h=4.8)
        if content.get_bottom()[1] < -3.35:
            content.shift(UP * (content.get_bottom()[1] + 3.35))

        arrows = VGroup(
            *[
                Arrow(boxes[i].get_bottom(), boxes[i + 1].get_top(), buff=0.04, color=GRAY, stroke_width=2)
                for i in range(len(boxes) - 1)
            ]
        )

        for i, box in enumerate(boxes):
            self.play(FadeIn(box, shift=DOWN * 0.08), run_time=0.28)
            if i < len(arrows):
                self.play(GrowArrow(arrows[i]), run_time=0.15)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(1.1)
        self.play(FadeOut(VGroup(st, sub, content, arrows)), run_time=0.6)

    # ------------------------------------------------------------------
    def show_rewards_and_amp(self) -> None:
        st = section_title("5. 奖励组合与 AMP 对抗")
        self.play(Write(st))

        task_lines = [
            ("+ track_anchor_linear / angular_velocity", GREEN),
            ("+ track_root_height（起身关键）", GREEN),
            ("− joint_acc · action_rate · foot_slip", RED),
            ("− is_terminated（weight −200）", RED),
        ]
        task_rewards = VGroup(
            *[body_text(txt, size=17, color=col, mono=True) for txt, col in task_lines]
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)

        amp_panel = VGroup(
            box_label("AMP Discriminator", 5.5, 0.85, PURPLE, 19),
            multiline_text(
                [
                    "expert: WalkandRun + Recovery",
                    "policy: 仿真生成状态转移",
                    "lerp=0.75 混合任务/风格 → step reward",
                ],
                size=16,
                color=GRAY,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        total = box_label("Train/mean_reward", 4.5, 0.8, GREEN, 19)

        content = VGroup(task_rewards, amp_panel, total).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        place_content_below_title(content, st, buff=0.42)

        self.play(Write(task_rewards), run_time=0.9)
        self.play(FadeIn(amp_panel), run_time=0.8)
        self.play(FadeIn(total), run_time=0.6)
        self.wait(1.1)
        self.play(FadeOut(VGroup(st, content)), run_time=0.6)

    # ------------------------------------------------------------------
    def show_training_dynamics(self) -> None:
        st = section_title("6. 训练曲线：Recovery Jump")
        self.play(Write(st))

        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 50, 10],
            width=8.0,
            height=3.2,
            axis_config={"color": GRAY, "stroke_width": 2},
        )
        x_label = body_text("iterations (×1000)", size=15, color=GRAY)
        x_label.next_to(axes, DOWN, buff=0.18)
        y_label = body_text("mean_reward", size=15, color=GRAY, mono=True)
        y_label.next_to(axes, LEFT, buff=0.22).rotate(PI / 2)

        def reward_curve(x: float) -> float:
            if x < 20:
                return 25 + 2 * np.sin(x * 0.5)
            return 25 + 16 * (1 - np.exp(-(x - 20) * 0.15))

        graph = axes.get_graph(reward_curve, color=GREEN, stroke_width=4)
        jump_line = DashedLine(axes.c2p(20, 0), axes.c2p(20, 45), color=ORANGE, stroke_width=2)
        jump_label = multiline_text(["≈20k iter", "Recovery 涌现"], size=16, color=ORANGE)
        jump_label.next_to(axes.c2p(20, 40), RIGHT, buff=0.12)

        chart = VGroup(axes, x_label, y_label, graph, jump_line, jump_label)

        metrics = VGroup(
            body_text("同时观察: episode_length → 1000", size=16),
            body_text("track_root_height 抬升 · is_terminated 衰减", size=16),
            body_text("Loss/amp_policy_pred vs amp_expert_pred 保持间距", size=15, color=GRAY, mono=True),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)

        content = VGroup(chart, metrics).arrange(DOWN, buff=0.35)
        place_content_below_title(content, st, buff=0.38)

        self.play(ShowCreation(axes), FadeIn(x_label), FadeIn(y_label), run_time=0.7)
        self.play(ShowCreation(graph), run_time=1.6)
        self.play(ShowCreation(jump_line), FadeIn(jump_label), run_time=0.7)
        self.play(FadeIn(metrics, shift=UP * 0.08), run_time=0.8)
        self.wait(1.3)
        self.play(FadeOut(VGroup(st, content)), run_time=0.6)

    # ------------------------------------------------------------------
    def show_export_and_deploy(self) -> None:
        st = section_title("7. 导出与真机部署")
        self.play(Write(st))

        pipeline = VGroup(
            box_label("save() 每 100 iter", 5.0, color=BLUE, font_size=18),
            box_label("model_<iter>.pt", 4.8, color=BLUE, font_size=18),
            box_label("policy.onnx\nobs → actions", 4.8, 0.9, PURPLE, 18),
            box_label("play.py 回放验证", 4.8, color=GREEN, font_size=18),
            box_label("Unitree G1 · wbc_fsm 50Hz", 5.2, color=GREEN, font_size=18),
        ).arrange(DOWN, buff=0.22)

        arrows = VGroup(
            *[
                Arrow(pipeline[i].get_bottom(), pipeline[i + 1].get_top(), buff=0.05, color=GRAY, stroke_width=2)
                for i in range(len(pipeline) - 1)
            ]
        )

        deploy_notes = VGroup(
            body_text("ONNX 内置 obs normalizer — 部署端勿重复归一化", size=16, color=ORANGE),
            body_text("动作语义: default_pos + 0.25 × action → PD", size=15, color=GRAY, mono=True),
            body_text("上线顺序: play 回放 → 吊架限速 → 全速", size=16),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)

        content = VGroup(pipeline, deploy_notes).arrange(DOWN, buff=0.38)
        place_content_below_title(content, st, buff=0.4)

        self.play(LaggedStartMap(FadeIn, pipeline, shift=DOWN * 0.1, lag_ratio=0.1), run_time=1.0)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.08), run_time=0.7)
        self.play(FadeIn(deploy_notes), run_time=0.7)
        self.wait(1.2)
        self.play(FadeOut(VGroup(st, content, arrows)), run_time=0.6)

    # ------------------------------------------------------------------
    def show_outro(self) -> None:
        title = body_text("AMP_mjlab 训练流程", size=42)
        cmd = code_line(
            "python scripts/train.py Unitree-G1-AMP-Flat\n--env.scene.num-envs=4096",
            17,
        )
        wiki = code_line("wiki/entities/amp-mjlab.md", 20)
        wiki.set_color(BLUE)
        block = VGroup(title, cmd, wiki).arrange(DOWN, buff=0.32)
        fit_in_frame(block, max_w=10.5, max_h=3.5)

        self.play(Write(title), run_time=0.7)
        self.play(FadeIn(cmd), FadeIn(wiki), run_time=0.7)
        self.wait(1.8)


class AmpMjlabTrainingFlowPreview(AmpMjlabTrainingFlow):
    """Shorter preview for quick renders (-l)."""

    def show_training_dynamics(self) -> None:
        st = section_title("6. Recovery Jump @ ~20k iter")
        note = body_text("mean_reward 阶跃 · episode_length → 1000", size=28, color=ORANGE)
        block = VGroup(st, note).arrange(DOWN, buff=0.5)
        self.play(Write(st), FadeIn(note), run_time=0.9)
        self.wait(0.9)
        self.play(FadeOut(block), run_time=0.4)
