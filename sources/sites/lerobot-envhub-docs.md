# LeRobot EnvHub 官方文档

- **标题：** Loading Environments from the Hub (EnvHub)
- **类型：** site / 官方文档
- **链接：** https://huggingface.co/docs/lerobot/envhub
- **代码仓：** https://github.com/huggingface/lerobot（已开源，Apache 2.0）
- **入库日期：** 2026-09-06
- **一句话说明：** LeRobot 从 Hugging Face Hub 动态加载仿真环境的机制：Hub 仓只需 `env.py` 暴露 `make_env`，策略侧一行 `make_env("user/repo", trust_remote_code=True)` 即可评测，无需 PyPI 打包。
- **沉淀到 wiki：** 是 → [`wiki/concepts/lerobot-envhub.md`](../../wiki/concepts/lerobot-envhub.md)

---

## 核心机制

**EnvHub** 把仿真环境从单体库中拆出，以 **HF Git 仓** 分发：

1. 作者发布含 `env.py` 的 Hub 仓（可用任意仿真后端：Gymnasium、Isaac Lab、MuJoCo 等）
2. 读者用 `lerobot.envs.make_env` 或 CLI `lerobot-eval --env.hub_path=...` 动态拉取
3. Git 语义做版本钉扎（branch / commit / tag）

**解耦价值：** 策略仓不必 `import` 任务仓；避免 N 个 benchmark 之间的依赖冲突；发现新环境 → 跑实验可在秒级完成。

---

## `env.py` 契约（必做）

必须暴露：

```python
def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg: EnvConfig = None):
    ...
```

**返回值（三选一）：**

| 返回类型 | 用途 |
|----------|------|
| `gym.vector.VectorEnv` | 最常见；单任务并行评测 |
| `gym.Env` | 单环境；LeRobot 自动包装 |
| `dict[suite_name, dict[task_id, VectorEnv]]` | 多任务 benchmark（如 LIBERO 式） |

环境须实现标准 `gym.vector.VectorEnv` 接口（`reset` / `step` 等）。可选通过 `EnvConfig` 传入任务名、相机键、episode 长度、控制模式等。

---

## 加载 API

```python
from lerobot.envs import make_env

envs_dict = make_env("lerobot/cartpole-env", n_envs=4, trust_remote_code=True)
```

**URL 格式：**

| 模式 | 示例 |
|------|------|
| `user/repo` | `make_env("lerobot/pusht-env")` → main 分支 `env.py` |
| `user/repo@revision` | `make_env("user/env@abc123")` → 钉 commit（论文复现推荐） |
| `user/repo:path` | `make_env("user/env:custom_env.py")` |
| `user/repo@rev:path` | 组合 |

**安全：** `trust_remote_code=True` 会执行第三方 Python；须审阅 `env.py`、钉 commit、查 `requirements.txt`，不可盲信随机用户仓。

---

## 仓结构（最小）

```
my-environment-repo/
├── env.py              # 必需
├── requirements.txt    # 可选；用户手动安装
├── README.md           # 推荐：观测/动作空间、奖励、示例
└── assets/             # 可选
```

**本地测试（文档推荐）：**

```python
from lerobot.envs.utils import _load_module_from_path, _call_make_env, _normalize_hub_result
module = _load_module_from_path("./env.py")
result = _call_make_env(module, n_envs=2, use_async_envs=False)
normalized = _normalize_hub_result(result)
```

---

## CLI 评测路径（README 交叉）

内置 benchmark：`lerobot-eval --env.type=libero|metaworld|...`

**Hub 环境：** 见各集成文档；Isaac Lab-Arena 示例：

```bash
lerobot-eval \
  --policy.path=nvidia/smolvla-arena-gr1-microwave \
  --env.type=isaaclab_arena \
  --env.hub_path=nvidia/isaaclab-arena-envs \
  --env.environment=gr1_microwave \
  --trust_remote_code=True
```

光轮厨房：`--env.hub_path=LightwheelAI/lw_benchhub_env`（见 [lw_benchhub_tour](../repos/lw_benchhub_tour.md)）。

---

## 参考示例仓

- `lerobot/cartpole-env` — 文档 CartPole 参考实现
- `nvidia/isaaclab-arena-envs` — Isaac Lab-Arena GPU 环境
- `LightwheelAI/lw_benchhub_env` — 光轮 LW-BenchHub 厨房任务

---

## 开源核查

- 文档与加载器在 **huggingface/lerobot** 主仓 **已开源**
- Hub 上的各 `env.py` 仓为**第三方远程代码**；开放程度因作者而异

---

## 对 wiki 的映射

- [LeRobot EnvHub](../../wiki/concepts/lerobot-envhub.md)
- [LeRobot](../../wiki/entities/lerobot.md)
- [Isaac Lab-Arena](../../wiki/entities/isaac-lab-arena.md)
- [LW BENCHHUB TOUR](../../wiki/entities/lw-benchhub-tour.md)
