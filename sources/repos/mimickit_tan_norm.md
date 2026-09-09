# Source: MimicKit `tan_norm` 旋转观测编码（xbpeng/MimicKit）

- **Title**: MimicKit `quat_to_tan_norm` / `tan_norm_to_quat` 实现
- **URL**: https://github.com/xbpeng/MimicKit/blob/main/mimickit/util/torch_util.py
- **相关调用**: https://github.com/xbpeng/MimicKit/blob/main/mimickit/envs/char_env.py (`compute_char_obs`)
- **Author**: Xue Bin Peng 团队
- **Year**: 2025（MimicKit 仓库；函数沿 Isaac Gym / 运动模仿栈惯例）
- **Type**: Code / API 摘录
- **License**: Apache-2.0
- **入库日期**: 2026-09-09
- **一句话说明**: 将 SO(3) 四元数编码为 6 维 `tan_norm`（旋转后的参考切向 + 法向），作为 DeepMimic / AMP 等环境的 `root_rot_obs` 与 `joint_rot_obs`，避免四元数 $q \equiv -q$ 双覆盖。

## 开源状态（步骤 2.5）

- **已开源**：MimicKit 主仓 `xbpeng/MimicKit`，Apache-2.0；观测编码在 `mimickit/util/torch_util.py` 与 `mimickit/envs/char_env.py` 可直接核对。
- **姊妹实现**：NVlabs [ProtoMotions](https://github.com/NVlabs/ProtoMotions) 的 `protomotions/utils/rotations.py` 提供同名 `quat_to_tan_norm` / `tan_norm_to_quat`（接口与语义对齐 MimicKit）。

## 摘录 1：编码定义（`quat_to_tan_norm`）

固定参考方向 $\mathbf{t}_0=[1,0,0]^\top$、$\mathbf{n}_0=[0,0,1]^\top$，用四元数 $q$ 旋转后拼接为 6 维：

$$
\text{tan\_norm}(q) = \big[\, R(q)\,\mathbf{t}_0 \;\|\; R(q)\,\mathbf{n}_0 \,\big] \in \mathbb{R}^6
$$

仓库实现（`@torch.jit.script`，节选）：

```python
def quat_to_tan_norm(q):
    ref_tan = torch.zeros_like(q[..., 0:3]); ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)
    ref_norm = torch.zeros_like(q[..., 0:3]); ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)
    return torch.cat([tan, norm], dim=-1)
```

**对 wiki 的映射**：[`wiki/formalizations/tan-norm-rotation.md`](../../wiki/formalizations/tan-norm-rotation.md) — 与 Zhou 6D（矩阵前两列 + Gram–Schmidt）对照，强调「固定体轴参考方向」这一工程变体。

## 摘录 2：解码（`tan_norm_to_matrix` → 四元数）

前 3 维为旋转后的 **x 轴**，后 3 维为旋转后的 **z 轴**；y 轴由叉积重建，再转四元数：

```python
def tan_norm_to_matrix(tan_norm):
    tan = tan_norm[..., 0:3]   # x-axis
    norm = tan_norm[..., 3:6]  # z-axis
    col2 = torch.cross(norm, tan, dim=-1)  # y = z × x
    mat3 = torch.stack([tan, col2, norm], dim=-2)
    return mat3.transpose(-1, -2)

def tan_norm_to_quat(tan_norm):
    return matrix_to_quat(tan_norm_to_matrix(tan_norm))
```

**注意**：网络输出的 6 维在解码前通常**未**强制单位正交；与 Zhou 6D 一样，推理端需 normalize + 重正交（ProtoMotions 的 `tan_norm_to_quat` 显式做了这一步）。

**对 wiki 的映射**：工程实践节写清「观测侧用 `quat_to_tan_norm`，SMP 扩散先验等下游用 `tan_norm_to_quat` 还原」。

## 摘录 3：DeepMimic 观测中的用法（`compute_char_obs`）

`char_env.py` 中根节点与每个关节四元数均转为 `tan_norm` 再拼进策略观测：

```python
root_rot_obs = torch_util.quat_to_tan_norm(root_rot)          # 6D
joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)  # J×6
obs = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel, ...]
```

`global_obs=True` 时用世界系四元数；否则根朝向先乘 `heading_inv` 再编码。G1 DeepMimic 配置下 `root_rot_obs` + 28 关节 × 6 维约占 char_obs 中旋转块主体（见组内笔记 `resources/train/MimicKit/MimicKit 05 DeepMimic.md`）。

**对 wiki 的映射**：[`wiki/entities/mimickit.md`](../../wiki/entities/mimickit.md)、[`wiki/methods/deepmimic.md`](../../wiki/methods/deepmimic.md) — 观测维度与原版 DeepMimic（四元数）差异。

## 摘录 4：SMP 扩散先验中的逆变换

`tools/diffusion_model/motion_prior_dataset.py` 将采样窗口中的 `root_rot_obs` / `joint_rot_obs` 经 `tan_norm_to_quat` 还原后再转 exp map 写回 motion frame——说明 **tan_norm 是 MimicKit 全栈统一的旋转观测 lingua franca**，不限于 PPO 观测。

**对 wiki 的映射**：[`wiki/methods/smp.md`](../../wiki/methods/smp.md) — 先验特征空间与 motion 表示衔接。

## 建议 wiki 动作

- 新建 **`wiki/formalizations/tan-norm-rotation.md`**
- 更新 **`wiki/formalizations/se3-representation.md`** 交叉引用
- 在 **`wiki/entities/mimickit.md`** 观测/表示小节链到本页
