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


def box_label(text: str, width: float = 3.2, height: float = 0.9, color: str = BLUE) -> VGroup:
    rect = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.12,
        stroke_color=color,
        stroke_width=2,
        fill_color=color,
        fill_opacity=0.15,
    )
    label = Text(text, font=CN, font_size=22, color=WHITE)
    label.move_to(rect.get_center())
    return VGroup(rect, label)


def code_line(text: str, size: int = 20) -> Text:
    return Text(text, font=MONO, font_size=size, color=GRAY)


def section_title(text: str) -> Text:
    return Text(text, font=CN, font_size=40, color=WHITE)


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
        title = Text("AMP_mjlab 训练流程", font=CN, font_size=56, color=WHITE)
        subtitle = Text(
            "Unitree G1 · mjlab + rsl_rl · 统一走跑与跌倒恢复",
            font=CN,
            font_size=28,
            color=GRAY,
        )
        subtitle.next_to(title, DOWN, buff=0.4)
        repo = code_line("github.com/ccrpRepo/AMP_mjlab", 24)
        repo.next_to(subtitle, DOWN, buff=0.5)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(repo), run_time=0.6)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, subtitle, repo)), run_time=0.8)

    # ------------------------------------------------------------------
    def show_entry_and_registration(self) -> None:
        st = section_title("1. 训练入口与任务注册")
        st.to_edge(UP, buff=0.5)
        self.play(Write(st))

        cmd = code_line(
            "python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096",
            22,
        )
        cmd.next_to(st, DOWN, buff=0.6)

        flow = VGroup(
            box_label("main() · tyro 解析任务", 4.0, color=BLUE),
            box_label("launch_training()", 3.6, color=BLUE),
            box_label("run_train()", 3.2, color=BLUE),
            box_label("AMPOnPolicyRunner.learn()", 4.4, color=PURPLE),
        )
        flow.arrange(DOWN, buff=0.35)
        flow.next_to(cmd, DOWN, buff=0.5)

        arrows = VGroup()
        for a, b in zip(flow[:-1], flow[1:]):
            arrows.add(Arrow(a.get_bottom(), b.get_top(), buff=0.08, color=GRAY, stroke_width=2))

        reg = VGroup(
            Text("register_mjlab_task", font=MONO, font_size=18, color=ORANGE),
            Text("Unitree-G1-AMP-Flat / Rough", font=CN, font_size=20, color=WHITE),
            Text("runner_cls = AMPOnPolicyRunner", font=MONO, font_size=18, color=ORANGE),
        )
        reg.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        reg.to_edge(RIGHT, buff=0.4).shift(DOWN * 0.3)

        self.play(Write(cmd), run_time=1.0)
        self.play(LaggedStartMap(FadeIn, flow, shift=RIGHT * 0.2, lag_ratio=0.15), run_time=1.2)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.2), run_time=0.8)
        self.play(FadeIn(reg, shift=LEFT * 0.2), run_time=0.8)
        self.wait(1.2)
        self.play(FadeOut(VGroup(st, cmd, flow, arrows, reg)), run_time=0.7)

    # ------------------------------------------------------------------
    def show_environment_stack(self) -> None:
        st = section_title("2. mjlab 并行仿真环境")
        st.to_edge(UP, buff=0.5)
        self.play(Write(st))

        center = box_label("ManagerBasedRlEnv\n4096 × G1", 3.8, 1.2, GREEN)
        center.shift(UP * 0.2)

        left_items = VGroup(
            Text("timestep = 0.005 s", font=MONO, font_size=18, color=GRAY),
            Text("decimation = 4  →  50 Hz 控制", font=MONO, font_size=18, color=GRAY),
            Text("episode_length_s = 20", font=MONO, font_size=18, color=GRAY),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        left_items.next_to(center, LEFT, buff=0.8).align_to(center, UP)

        right_items = VGroup(
            Text("WalkandRun/ 参考动作", font=CN, font_size=20, color=ORANGE),
            Text("Recovery/ 起身动作", font=CN, font_size=20, color=ORANGE),
            Text("delay_reset 40% env", font=MONO, font_size=18, color=GRAY),
            Text("max_delay_steps = 250", font=MONO, font_size=18, color=GRAY),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        right_items.next_to(center, RIGHT, buff=0.6).align_to(center, UP)

        events = VGroup(
            box_label("reset_from_motion", 3.0, 0.75, ORANGE),
            box_label("push_robot 推扰", 2.6, 0.75, ORANGE),
            box_label("摩擦/质心 DR", 2.4, 0.75, ORANGE),
        ).arrange(RIGHT, buff=0.25)
        events.next_to(center, DOWN, buff=0.7)

        self.play(FadeIn(center, scale=0.9), run_time=0.8)
        self.play(FadeIn(left_items, shift=RIGHT * 0.2), FadeIn(right_items, shift=LEFT * 0.2), run_time=0.9)
        self.play(LaggedStartMap(FadeIn, events, shift=UP * 0.15, lag_ratio=0.12), run_time=0.9)
        self.wait(1.3)
        self.play(FadeOut(VGroup(st, center, left_items, right_items, events)), run_time=0.7)

    # ------------------------------------------------------------------
    def show_networks_and_data(self) -> None:
        st = section_title("3. 观测、动作与网络")
        st.to_edge(UP, buff=0.5)
        self.play(Write(st))

        actor = box_label("Actor-Critic", 3.0, 1.0, BLUE)
        disc = box_label("AMP Discriminator", 3.4, 1.0, PURPLE)
        actor.shift(LEFT * 3.2 + UP * 0.3)
        disc.shift(RIGHT * 3.2 + UP * 0.3)

        actor_in = Text("obs 384 维\n4 帧 × 96", font=CN, font_size=20, color=WHITE)
        actor_out = Text("actions 29 维\nscale=0.25 PD", font=CN, font_size=20, color=WHITE)
        actor_in.next_to(actor, LEFT, buff=0.5)
        actor_out.next_to(actor, RIGHT, buff=0.5)

        disc_in = Text("amp obs\n13 body 部位", font=CN, font_size=20, color=WHITE)
        disc_out = Text("风格奖励\ncoef=0.1", font=CN, font_size=20, color=WHITE)
        disc_in.next_to(disc, LEFT, buff=0.5)
        disc_out.next_to(disc, RIGHT, buff=0.5)

        obs_terms = VGroup(
            Text("base_ang_vel · projected_gravity", font=MONO, font_size=16, color=GRAY),
            Text("twist command · joint_pos/vel", font=MONO, font_size=16, color=GRAY),
            Text("last_action · history_ordering=time", font=MONO, font_size=16, color=GRAY),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        obs_terms.next_to(actor, DOWN, buff=0.6).align_to(actor, LEFT)

        hidden = Text(
            "Actor [512,256,128]  ·  Disc [1024,512,256]  ·  AMPPPO",
            font=MONO,
            font_size=18,
            color=GRAY,
        )
        hidden.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(actor), FadeIn(disc), run_time=0.8)
        self.play(
            FadeIn(actor_in), FadeIn(actor_out),
            FadeIn(disc_in), FadeIn(disc_out),
            run_time=0.9,
        )
        self.play(Write(obs_terms), FadeIn(hidden), run_time=1.0)
        self.wait(1.2)
        self.play(
            FadeOut(VGroup(st, actor, disc, actor_in, actor_out, disc_in, disc_out, obs_terms, hidden)),
            run_time=0.7,
        )

    # ------------------------------------------------------------------
    def show_single_iteration(self) -> None:
        st = section_title("4. 单次训练迭代 (AmpOnPolicyRunner.learn)")
        st.to_edge(UP, buff=0.5)
        self.play(Write(st))

        steps = [
            ("Rollout", "24 steps × 4096 envs", BLUE),
            ("alg.act()", "采样 actions", BLUE),
            ("env.step()", "MuJoCo 物理步进", GREEN),
            ("predict_amp_reward", "任务奖励 + 风格奖励", PURPLE),
            ("compute_returns", "GAE λ=0.95 γ=0.99", BLUE),
            ("alg.update()", "PPO + 判别器 5 epochs", PURPLE),
        ]

        boxes = VGroup()
        for title, detail, color in steps:
            g = VGroup()
            r = RoundedRectangle(
                width=5.5, height=0.85, corner_radius=0.1,
                stroke_color=color, stroke_width=2,
                fill_color=color, fill_opacity=0.12,
            )
            t1 = Text(title, font=MONO, font_size=22, color=WHITE)
            t2 = Text(detail, font=CN, font_size=18, color=GRAY)
            t1.move_to(r.get_center() + UP * 0.12)
            t2.move_to(r.get_center() + DOWN * 0.18)
            g.add(r, t1, t2)
            boxes.add(g)

        boxes.arrange(DOWN, buff=0.22)
        boxes.next_to(st, DOWN, buff=0.45)

        arrows = VGroup(
            *[Arrow(boxes[i].get_bottom(), boxes[i + 1].get_top(), buff=0.05, color=GRAY, stroke_width=2)
              for i in range(len(boxes) - 1)]
        )

        note = Text(
            "每 iter 采样 ≈ 98,304 transitions  (4096 × 24)",
            font=CN, font_size=22, color=ORANGE,
        )
        note.to_edge(DOWN, buff=0.7)

        for i, box in enumerate(boxes):
            self.play(FadeIn(box, shift=RIGHT * 0.15), run_time=0.35)
            if i < len(arrows):
                self.play(GrowArrow(arrows[i]), run_time=0.2)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.3)
        self.play(FadeOut(VGroup(st, boxes, arrows, note)), run_time=0.7)

    # ------------------------------------------------------------------
    def show_rewards_and_amp(self) -> None:
        st = section_title("5. 奖励组合与 AMP 对抗")
        st.to_edge(UP, buff=0.5)
        self.play(Write(st))

        task_rewards = VGroup(
            Text("+ track_anchor_linear_velocity", font=MONO, font_size=18, color=GREEN),
            Text("+ track_anchor_angular_velocity", font=MONO, font_size=18, color=GREEN),
            Text("+ track_root_height  (起身关键)", font=MONO, font_size=18, color=GREEN),
            Text("− joint_acc / action_rate / foot_slip", font=MONO, font_size=18, color=RED),
            Text("− is_terminated  (weight −200)", font=MONO, font_size=18, color=RED),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        task_rewards.shift(LEFT * 2.8 + UP * 0.2)

        amp_box = RoundedRectangle(
            width=4.8, height=2.8, corner_radius=0.15,
            stroke_color=PURPLE, stroke_width=2,
            fill_color=PURPLE, fill_opacity=0.1,
        )
        amp_box.shift(RIGHT * 2.8 + UP * 0.1)
        amp_text = VGroup(
            Text("Discriminator", font=MONO, font_size=22, color=WHITE),
            Text("expert: WalkandRun + Recovery", font=CN, font_size=17, color=GRAY),
            Text("policy: 仿真生成状态转移", font=CN, font_size=17, color=GRAY),
            Text("lerp = 0.75 混合任务/风格", font=MONO, font_size=17, color=ORANGE),
            Text("→ 写入 step reward", font=CN, font_size=17, color=GRAY),
        ).arrange(DOWN, buff=0.12)
        amp_text.move_to(amp_box.get_center())

        plus = Text("=", font_size=48, color=WHITE)
        plus.move_to(ORIGIN + UP * 0.1)

        total = box_label("Train/mean_reward", 3.6, 0.9, GREEN)
        total.next_to(plus, DOWN, buff=1.0)

        self.play(Write(task_rewards), run_time=1.0)
        self.play(FadeIn(amp_box), Write(amp_text), run_time=1.0)
        self.play(FadeIn(plus), FadeIn(total), run_time=0.7)
        self.wait(1.3)
        self.play(FadeOut(VGroup(st, task_rewards, amp_box, amp_text, plus, total)), run_time=0.7)

    # ------------------------------------------------------------------
    def show_training_dynamics(self) -> None:
        st = section_title("6. 训练曲线：Recovery Jump")
        st.to_edge(UP, buff=0.5)
        self.play(Write(st))

        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 50, 10],
            width=9,
            height=4,
            axis_config={"color": GRAY, "stroke_width": 2},
        )
        axes.shift(DOWN * 0.3)
        x_label = Text("iterations (×1000)", font=CN, font_size=18, color=GRAY)
        x_label.next_to(axes, DOWN, buff=0.25)
        y_label = Text("mean_reward", font=MONO, font_size=18, color=GRAY)
        y_label.next_to(axes, LEFT, buff=0.3).rotate(PI / 2)

        def reward_curve(x: float) -> float:
            if x < 20:
                return 25 + 2 * np.sin(x * 0.5)
            return 25 + 16 * (1 - np.exp(-(x - 20) * 0.15))

        graph = axes.get_graph(reward_curve, color=GREEN, stroke_width=4)
        jump_line = DashedLine(
            axes.c2p(20, 0), axes.c2p(20, 45),
            color=ORANGE, stroke_width=2,
        )
        jump_label = Text("≈20k iter\nRecovery 涌现", font=CN, font_size=20, color=ORANGE)
        jump_label.next_to(jump_line, UP, buff=0.1).shift(RIGHT * 0.3)

        metrics = VGroup(
            Text("同时观察: episode_length → 1000", font=CN, font_size=20, color=WHITE),
            Text("track_root_height 抬升 · is_terminated 衰减", font=CN, font_size=20, color=WHITE),
            Text("Loss/amp_policy_pred vs amp_expert_pred 保持间距", font=MONO, font_size=18, color=GRAY),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        metrics.to_edge(DOWN, buff=0.55)

        self.play(ShowCreation(axes), FadeIn(x_label), FadeIn(y_label), run_time=0.8)
        self.play(ShowCreation(graph), run_time=2.0)
        self.play(ShowCreation(jump_line), FadeIn(jump_label), run_time=0.9)
        self.play(LaggedStartMap(FadeIn, metrics, shift=UP * 0.1, lag_ratio=0.15), run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(VGroup(st, axes, x_label, y_label, graph, jump_line, jump_label, metrics)), run_time=0.7)

    # ------------------------------------------------------------------
    def show_export_and_deploy(self) -> None:
        st = section_title("7. 导出与真机部署")
        st.to_edge(UP, buff=0.5)
        self.play(Write(st))

        nodes = VGroup(
            box_label("save() 每 100 iter", 3.2, color=BLUE),
            box_label("model_<iter>.pt", 3.0, color=BLUE),
            box_label("policy.onnx\nobs→actions", 3.0, 1.0, PURPLE),
            box_label("play.py 回放验证", 3.0, color=GREEN),
            box_label("Unitree G1\nwbc_fsm 50Hz", 3.0, 1.0, GREEN),
        )
        nodes.arrange(RIGHT, buff=0.35)
        nodes.scale(0.92)
        nodes.next_to(st, DOWN, buff=0.8)

        arrows = VGroup(
            *[Arrow(nodes[i].get_right(), nodes[i + 1].get_left(), buff=0.1, color=GRAY, stroke_width=2)
              for i in range(len(nodes) - 1)]
        )

        deploy_notes = VGroup(
            Text("ONNX 内置 obs normalizer — 部署端勿重复归一化", font=CN, font_size=20, color=ORANGE),
            Text("动作语义: default_pos + 0.25 × action → PD", font=MONO, font_size=18, color=GRAY),
            Text("上线顺序: play 回放 → 吊架限速 → 全速", font=CN, font_size=20, color=WHITE),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        deploy_notes.next_to(nodes, DOWN, buff=0.7)

        self.play(LaggedStartMap(FadeIn, nodes, shift=DOWN * 0.15, lag_ratio=0.12), run_time=1.2)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.1), run_time=0.8)
        self.play(FadeIn(deploy_notes, shift=UP * 0.15), run_time=0.9)
        self.wait(1.5)
        self.play(FadeOut(VGroup(st, nodes, arrows, deploy_notes)), run_time=0.7)

    # ------------------------------------------------------------------
    def show_outro(self) -> None:
        title = Text("AMP_mjlab 训练流程", font=CN, font_size=48, color=WHITE)
        cmd = code_line("python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096", 20)
        cmd.next_to(title, DOWN, buff=0.4)
        wiki = Text(
            "wiki/entities/amp-mjlab.md",
            font=MONO,
            font_size=22,
            color=BLUE,
        )
        wiki.next_to(cmd, DOWN, buff=0.35)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(cmd), FadeIn(wiki), run_time=0.8)
        self.wait(2.0)


class AmpMjlabTrainingFlowPreview(AmpMjlabTrainingFlow):
    """Shorter preview for quick renders (-l)."""

    def show_training_dynamics(self) -> None:
        st = section_title("6. Recovery Jump @ ~20k iter")
        st.to_edge(UP, buff=0.5)
        note = Text(
            "mean_reward 阶跃 · episode_length → 1000",
            font=CN, font_size=32, color=ORANGE,
        )
        self.play(Write(st), FadeIn(note), run_time=1.0)
        self.wait(1.0)
        self.play(FadeOut(VGroup(st, note)), run_time=0.5)
