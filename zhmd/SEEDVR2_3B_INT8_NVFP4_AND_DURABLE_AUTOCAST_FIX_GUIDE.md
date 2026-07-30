# SeedVR2 — 3B INT8 ConvRot / NVFP4 Registry + Durable NVFP4 Autocast 跳过

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../md/SEEDVR2_3B_INT8_NVFP4_AND_DURABLE_AUTOCAST_FIX_GUIDE.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

目标自定义节点：`ComfyUI/custom_nodes/seedvr2_videoupscaler`  
规范提交：`2f90466cc78312f21677012eeabfd0ddcb7259d9`  
（`Support 3B NVFP4/INT8 registry and durable NVFP4 autocast skip`）

本指南记录 **2026-07-31** 修复，内容包括：

1. 将 **3B** HSWQ INT8 ConvRot 与 NVFP4 权重包按与 7B 相同的方式登记到 registry
2. 增加 **durable** 的 NVFP4 autocast 跳过标志，使 materialize 清空 checkpoint 路径之后，推理仍能跳过 autocast

相关既有指南：

- `md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md` — NVFP4 原生 ops 与 torch.compile（中文：`zhmd/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md`）
- `md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md` — INT8 构造期 `comfy.ops`（若存在；中文：`zhmd/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md`）

模型权重包（本地示例）：

- `models/SEEDVR2/seedvr2_3b_nvfp4.safetensors`
- `models/SEEDVR2/seedvr2_3b_int8_convrot.safetensors`

---

## 1. 增加 3B ConvRot INT8 / NVFP4

### 本提交之前已可用的部分

HSWQ INT8 / NVFP4 的原生显存路径 **并不** 仅依赖文件名。检测是 **基于内容** 的：

- 扫描 checkpoint 中的 `*.comfy_quant` 元数据
- 在 DiT **构造期**（meta）通过 `create_object(..., operations=...)` 注入 `comfy.ops.mixed_precision_ops`
- 加载时保持 `QuantizedTensor` 存储（打包的 INT8 / NVFP4），而不是展开为 FP16/BF16

该路径位于：

- `src/optimization/nvfp4_native_ops.py`
- `src/optimization/int8_native_ops.py`
- `src/core/model_loader.py`（`_dit_comfy_quant_ops`、`_dit_needs_comfy_quant_prep`）
- DiT `dit_3b` / `dit_7b` 已同样传递 `operations=`

在 registry 修复之前的冒烟测试已表明：只要原生 ops 路径跑通，两个 3B 权重包都能以 `QuantizedTensor` 加载（约 1957 MB NVFP4 / 约 3342 MB INT8 CUDA DiT 权重，约 210 个量化 Linear）。

### 当时缺失的部分

`MODEL_REGISTRY` 已列出 7B 条目：

- `seedvr2_7b_int8_convrot.safetensors`
- `seedvr2_7b_nvfp4.safetensors`
- sharp 变体

但 **没有** 列出对应的 3B 条目。缺少 registry 时：

- UI / 下载 / 默认列表可能省略这些权重包
- `resolve_dit_config_folder()` 无法使用 registry 的 `size="3B"`，只能回退到文件名启发式（`"3b" in name`）
- Runner 配置选择历史上使用原始的 `"7b" in dit_model`，对不含 `7b`/`3b` 标记的文件名较脆弱

### 本提交为 3B 增加的内容

按与 7B 相同的 precision 标签登记两个 3B 权重包：

| 文件名 | `ModelInfo` |
|---|---|
| `seedvr2_3b_int8_convrot.safetensors` | `size="3B"`, `precision="int8_tensorwise_convrot"` |
| `seedvr2_3b_nvfp4.safetensors` | `size="3B"`, `precision="nvfp4"` |

同时将 `_create_new_runner` 改为使用 `resolve_dit_config_folder(os.path.basename(dit_model))`，使 3B / 7B 配置目录来自 registry 规格（或文件名标记），而不再仅依赖硬编码的 `"7b" in dit_model` 子串。

**重要：** 仅登记 registry **不会**实现量化。原生显存仍来自内容检测 + 构造期 ops。Registry 使 3B 权重包在列表、配置目录解析以及与 7B 的文档对等性上成为一等公民。

---

## 2. 新增 / 修改的文件名

提交 `2f90466` 改动了 **四个** 源文件（无新模块）：

| 文件 | 作用 |
|---|---|
| `src/utils/model_registry.py` | 增加 3B INT8 / NVFP4 的 `MODEL_REGISTRY` 条目；托管 `resolve_dit_config_folder`（既有辅助函数，现由 runner 创建使用） |
| `src/core/model_loader.py` | 在 `prepare_model_structure` 中设置 durable 的 `runner._dit_is_nvfp4` |
| `src/core/generation_phases.py` | autocast 跳过只读 `_dit_is_nvfp4`；去掉实时的 `checkpoint_is_nvfp4(_dit_checkpoint)` |
| `src/core/model_configuration.py` | 导入辅助函数；在 `_create_new_runner` 中使用 `resolve_dit_config_folder`；在 DiT 缓存复用时恢复 `_dit_is_nvfp4` / `_dit_comfy_quant_native` |

不属于 `2f90466`（后续独立提交）：

- `8848760` — 删除仓库根目录的 `seedvr2_int8_bench.py`
- `a4a5a96` — 跟踪 `benchmark/seedvr2_int8_bench.py`、`benchmark/seedvr2_nvfp4_bench.py`；停止忽略 `benchmark/`

---

## 3. 新增 / 修改代码全文

### 3.1 `src/utils/model_registry.py` — 新增 registry 行

```python
    # HSWQ INT8 / NVFP4 (native VRAM path; same as 7B)
    "seedvr2_3b_int8_convrot.safetensors": ModelInfo(size="3B", precision="int8_tensorwise_convrot"),
    "seedvr2_3b_nvfp4.safetensors": ModelInfo(size="3B", precision="nvfp4"),
```

### 3.2 `src/utils/model_registry.py` — `resolve_dit_config_folder`（本修复使用）

```python
def resolve_dit_config_folder(dit_model: str) -> str:
    """
    Resolve configs_7b vs configs_3b from registry size and/or filename.

    Filename substring \"7b\"/\"3b\" is the historical rule. Registry size is used
    when the model is registered (including HSWQ INT8 names). Prefer explicit
    7b/3b tokens in the basename so untagged temp names do not silently pick 3B.
    """
    info = MODEL_REGISTRY.get(dit_model)
    if info is not None and info.category == "dit":
        size = (info.size or "").upper()
        if size == "7B":
            return "configs_7b"
        if size == "3B":
            return "configs_3b"

    name = dit_model.lower()
    if "7b" in name:
        return "configs_7b"
    if "3b" in name:
        return "configs_3b"
    return "configs_3b"
```

### 3.3 `src/core/model_configuration.py` — 导入

```python
from ..utils.model_registry import resolve_dit_config_folder
from ..optimization.nvfp4_native_ops import checkpoint_is_nvfp4
from ..optimization.int8_native_ops import checkpoint_is_hswq_int8
```

### 3.4 `src/core/model_configuration.py` — `_create_new_runner` 配置路径

**之前：**

```python
    config_path = os.path.join(script_directory, 
                              './configs_7b' if "7b" in dit_model else './configs_3b', 
                              'main.yaml')
```

**之后：**

```python
    config_folder = resolve_dit_config_folder(os.path.basename(dit_model))
    config_path = os.path.join(script_directory, config_folder, 'main.yaml')
```

### 3.5 `src/core/model_configuration.py` — DiT 缓存复用标志

```python
        # Durable native-quant flags (survive _dit_checkpoint clear after materialize)
        cp = runner._dit_checkpoint
        runner._dit_is_nvfp4 = checkpoint_is_nvfp4(cp)
        runner._dit_comfy_quant_native = runner._dit_is_nvfp4 or checkpoint_is_hswq_int8(cp)
```

### 3.6 `src/core/model_loader.py` — 结构准备时的 durable 标志

```python
        runner._dit_comfy_quant_native = bool(create_kwargs)
        # Durable after materialize clears _dit_checkpoint (generation_phases autocast skip).
        runner._dit_is_nvfp4 = bool(create_kwargs) and checkpoint_is_nvfp4(checkpoint_path)
```

### 3.7 `src/core/generation_phases.py` — autocast 跳过（修复后）

```python
            # Use durable _dit_is_nvfp4: materialize_model clears _dit_checkpoint.
            nvfp4_native = bool(getattr(runner, "_dit_is_nvfp4", False))
            debug.start_timer(f"dit_inference_{upscale_idx+1}")
            with torch.no_grad():
                use_autocast = (
                    not nvfp4_native
                    and dit_dtype != ctx['compute_dtype']
                    and ctx['dit_device'].type != 'mps'
                )
```

移除的导入：

```python
from ..optimization.nvfp4_native_ops import checkpoint_is_nvfp4
```

---

## 4. 3B / registry / 配置变更的含义

### Registry 条目

- 使 `seedvr2_3b_int8_convrot.safetensors` 与 `seedvr2_3b_nvfp4.safetensors` 成为带 `size="3B"` 的正式 DiT 条目。
- 使 precision 字符串与 7B 对齐（`int8_tensorwise_convrot`、`nvfp4`），便于工具与文档一致。
- 使 `resolve_dit_config_folder` 可在未来文件名省略 `3b` 标记时，仍能通过 registry 规格返回 `configs_3b`（只要该名称已登记）。

### `_create_new_runner` 中的 `resolve_dit_config_folder`

- 旧规则：`"7b" in dit_model` → `configs_7b`，否则 `configs_3b`。
- 新规则：先看 registry 规格，再看文件名中的 `7b`/`3b`，默认 `configs_3b`。
- 避免对不含 `"7b"` 但登记为 7B 的名称误选 7B YAML，并使 3B 量化名称可通过 `size="3B"` 干净解析。

### 缓存复用时的标志恢复

- 复用缓存 DiT 时，会再次把 `_dit_checkpoint` 设为磁盘路径。
- 立即从该路径重新计算 `_dit_is_nvfp4` 与 `_dit_comfy_quant_native`，避免后续阶段继承上一次运行的陈旧或缺失标志。

### 这些变更 **不是** 什么

- 不是第二套量化实现。
- 不替代基于内容的 `*.comfy_quant` 检测。
- 本身不能单独修复 float32 → `quantize_nvfp4` 失败（那是第 5–7 节）。

---

## 5. NVFP4 缺陷概述

### 症状

原生 NVFP4 DiT（7B **与** 3B）在放大推理时可能因 kitchen / CUDA 错误失败：当 `quantize_nvfp4` 收到 **float32** 激活时。Comfy Kitchen 的 CUDA 分发仅接受 FP16/BF16（`DISPATCH_HALF_DTYPE`）；float32 会被拒绝（dtype 代码 0）。

### 既定缓解（已在 NVFP4 指南中记录）

在 DiT 放大阶段，对原生 NVFP4 **跳过 `torch.autocast`**，避免 LayerNorm / RMSNorm 在进入 NVFP4 Linear 的 `from_float` / quantize 之前，在 autocast 下把激活提升为 float32。

ComfyUI UNet/Flux 风格的 NVFP4 路径通常保持 FP16/BF16 激活，而不用 autocast 包裹整个 forward。SeedVR2 先前试图通过在检测到 NVFP4 原生时跳过 autocast 来镜像该行为。

### materialize 之后检测失效

`materialize_model` 在权重加载后清空 checkpoint 路径：

```python
    if is_dit:
        runner._dit_checkpoint = None
        runner._dit_dtype_override = None
```

**旧的** autocast 跳过条件为：

```python
            nvfp4_native = (
                bool(getattr(runner, "_dit_comfy_quant_native", False))
                and checkpoint_is_nvfp4(getattr(runner, "_dit_checkpoint", None))
            )
```

时间线：

1. `prepare_model_structure` 将 `_dit_checkpoint` 设为真实路径，并可能设置 `_dit_comfy_quant_native = True`。
2. `materialize_model` 加载权重，然后将 `_dit_checkpoint = None`。
3. 放大阶段运行。`checkpoint_is_nvfp4(None)` 为 **False**。
4. 即使当前 DiT 已是 NVFP4 `QuantizedTensor`，`nvfp4_native` 仍变为 **False**。
5. autocast 可能启用 → LayerNorm/RMSNorm → float32 → NVFP4 Linear → kitchen 拒绝。

因此该缺陷 **不是**「3B 缺少 NVFP4 内核」，而是 **标志生命周期**：跳过逻辑依赖一条在 materialize 后被有意清空的路径。这同时影响 **3B 与 7B** NVFP4。

仅用 `_dit_comfy_quant_native` 作为跳过信号也不够：INT8 原生同样设置该构造期 ops 标志；跳过 autocast 是 NVFP4 kitchen quantize 的特定需求，不能作为所有量化权重包的笼统规则。

---

## 6. Durable 修复全文（autocast 跳过的持久化）

本节是针对「checkpoint 被清空后 autocast 跳过失效」的 **durable / 永久** 修复（`_dit_is_nvfp4`）。

### 6.1 结构准备时设置标志 — `model_loader.py`

```python
        runner.dit = model
        runner._dit_checkpoint = checkpoint_path
        runner._dit_block_swap_config = block_swap_config
        runner._dit_comfy_quant_native = bool(create_kwargs)
        # Durable after materialize clears _dit_checkpoint (generation_phases autocast skip).
        runner._dit_is_nvfp4 = bool(create_kwargs) and checkpoint_is_nvfp4(checkpoint_path)
```

### 6.2 Materialize 仍清空路径（不变；有意为之）

```python
    if is_dit:
        runner._dit_checkpoint = None
        runner._dit_dtype_override = None
```

此处 **不会** 清空 `_dit_is_nvfp4`。

### 6.3 推理仅使用 durable 标志 — `generation_phases.py`

**之前（materialize 后失效）：**

```python
            nvfp4_native = (
                bool(getattr(runner, "_dit_comfy_quant_native", False))
                and checkpoint_is_nvfp4(getattr(runner, "_dit_checkpoint", None))
            )
```

**之后：**

```python
            # Use durable _dit_is_nvfp4: materialize_model clears _dit_checkpoint.
            nvfp4_native = bool(getattr(runner, "_dit_is_nvfp4", False))
            debug.start_timer(f"dit_inference_{upscale_idx+1}")
            with torch.no_grad():
                use_autocast = (
                    not nvfp4_native
                    and dit_dtype != ctx['compute_dtype']
                    and ctx['dit_device'].type != 'mps'
                )
                if use_autocast:
                    with torch.autocast(ctx['dit_device'].type, ctx['compute_dtype'], enabled=True):
                        upscaled_latents = runner.inference(
```

### 6.4 缓存复用时恢复 — `model_configuration.py`

```python
        runner.dit = cache_context['cached_dit']
        runner._dit_checkpoint = find_model_file(dit_model, base_cache_dir)
        runner._dit_model_name = dit_model
        # Durable native-quant flags (survive _dit_checkpoint clear after materialize)
        cp = runner._dit_checkpoint
        runner._dit_is_nvfp4 = checkpoint_is_nvfp4(cp)
        runner._dit_comfy_quant_native = runner._dit_is_nvfp4 or checkpoint_is_hswq_int8(cp)
```

---

## 7. Durable 修复的含义

### 设计

| 属性 | 生命周期 | 用途 |
|---|---|---|
| `_dit_checkpoint` | materialize 后清空 | 仅作加载时的临时路径；推理时不得依赖 |
| `_dit_comfy_quant_native` | 保留 | 「构造使用了 mixed_precision ops」（INT8 **或** NVFP4） |
| `_dit_is_nvfp4` | 跨 materialize 保留；缓存复用时重算 | 「此 DiT 为 NVFP4 原生；放大阶段跳过 autocast」 |

### 为何需要专用 bool

1. **在路径清空后仍存活** — 推理不再调用 `checkpoint_is_nvfp4(None)`。
2. **NVFP4 专用** — autocast 跳过针对 kitchen NVFP4 拒绝 float32，而非每一个量化 DiT。
3. **开销低** — runner 上一个 `bool`；每个放大 batch 不必重新扫描 safetensors。
4. **缓存安全** — 复用路径在下一次 materialize 清空之前，从恢复的 checkpoint 路径重新推导标志。

### 成功表现

对 NVFP4 权重包执行 `prepare_model_structure` 之后：

- `_dit_is_nvfp4 is True`
- `_dit_comfy_quant_native is True`

`materialize_model` 之后：

- `_dit_checkpoint is None`
- `_dit_is_nvfp4` 仍为 `True`

放大期间：

- `nvfp4_native is True` → `use_autocast is False`
- 激活保持在 DiT 计算 dtype（FP16/BF16 路径），不会经 autocast 提升为 float32 再进入 NVFP4 Linear

### 范围提醒

- 该 durable 标志修复的是 **autocast 跳过的持久性**。
- 打包显存仍依赖构造期 `operations=` 与 `*.comfy_quant` 内容检测（架构不变）。
- 3B registry 条目修复的是 `seedvr2_3b_*` 权重包的 **产品对等 / 配置解析**；不能替代 durable 标志。

---

## 审计锚点

| 项目 | 值 |
|---|---|
| 修复提交 | `2f90466cc78312f21677012eeabfd0ddcb7259d9` |
| 文件 | `model_registry.py`、`model_loader.py`、`generation_phases.py`、`model_configuration.py` |
| 新增 runner 属性 | `runner._dit_is_nvfp4` |
| materialize 后清空 | `runner._dit_checkpoint` |
| 3B INT8 权重包 | `seedvr2_3b_int8_convrot.safetensors` |
| 3B NVFP4 权重包 | `seedvr2_3b_nvfp4.safetensors` |

---

## 与请求大纲的对应

| 请求章节 | 本指南 |
|---|---|
| ① 3B convrot INT8 / NVFP4 追加 | §1 |
| ② 新增 / 修改文件名 | §2 |
| ③ 新增 / 修改代码全文 | §3 |
| ④ 含义 | §4 |
| ⑤ NVFP4 缺陷概述 | §5 |
| ⑥ Durable 修复全文 | §6 |
| ⑦ 该修复的含义 | §7 |
