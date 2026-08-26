# MIM_PyTorch 深度优化实施规划方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 MIM_PyTorch(CVPR2019 Memory In Memory 时空预测模型的 PyTorch 复现)进行系统级深度优化,覆盖 I/O 流水线、CPU 线程、GPU 推理加速、安全漏洞、Bug 修复、逻辑审计与代码质量七个维度。

**Architecture:** 模型核心为 ST-LSTM + MIM Block 堆叠的递归时空网络,时间步串行依赖强(无法跨时间步并行),优化重心放在:数据加载流水线异步化、混合精度 + torch.compile 算子融合、推理期 CUDA Graphs/内存复用、以及安全与数值正确性加固。

**Tech Stack:** PyTorch 2.5.1 (torch.compile / torch.autocast / GradScaler / CUDA Graphs), torch.utils.data 多进程 DataLoader, pytest 测试框架。

---

## 第 0 部分:审计结论(已完成)

### 0.1 已发现并修复的关键缺陷

| # | 严重度 | 位置 | 问题 | 状态 |
|---|--------|------|------|------|
| 1 | 🔴 安全漏洞 | `train.py` `torch.load()` | 缺少 `weights_only=True`,加载恶意 checkpoint 可执行任意代码 (CVE-2022-45907 类风险) | ✅ 已修复 |
| 2 | 🔴 数值 Bug | `metrics.py` `batch_psnr` | 5D 输入 `[B,T,C,H,W]` 上 `dim=[1,2,3]` 漏约减 W 维,PSNR 按 `[B,W]` 求均值,结果错误 | ✅ 已修复(动态 `range(1, x.dim())`) |
| 3 | 🔴 数值 Bug | `metrics.py` `batch_ssim` | `dim=[2,3]` 约减的是 C/H 维而非空间维,SSIM 统计完全错误 | ✅ 已修复 |
| 4 | 🟡 性能 | `train.py` | `.to(device)` 缺少 `non_blocking=True`,与 `pin_memory` 配合失效,H2D 传输同步阻塞 | ✅ 已修复 |
| 5 | 🟡 性能 | `train.py` | `optimizer.zero_grad()` 默认 `set_to_none=False`,多余的零填充 kernel | ✅ 已修复 |
| 6 | 🟡 性能 | `dataset.py` | 缺少 `persistent_workers` / `prefetch_factor`,每个 epoch 重启 worker 进程 | ✅ 已修复 |
| 7 | 🟢 功能 | `train.py` | 无 AMP、无 torch.compile、无随机种子、无 scaler 状态保存 | ✅ 已补齐 |

### 0.2 历史轮次已修复(模型核心)

| # | 问题 | 修复 |
|---|------|------|
| 8 | `MIMN.oc_weight` 未注册为 `nn.Parameter`(不训练/不上 GPU/不入 state_dict) | 已改为 `nn.Parameter` |
| 9 | 全部 LSTM 门缺少 `forget_bias=1.0`(与 TF 原版不一致,训练不稳) | 已添加 |
| 10 | `ss_bool=None` 时 forward 崩溃 | 自动创建全零张量 |
| 11 | `init_state` 使用固定 `self.batch`,不支持动态 batch | 改为运行时 `x.shape[0]` |
| 12 | 缺少论文关键 trick: Tensor Layer Normalization | 已实现 `TensorLayerNorm` |
| 13 | 权重初始化与 TF 原版 Xavier uniform 不一致 | 已对齐 |

### 0.3 逻辑审计遗留观察项(不阻塞,列入规划)

- `MIM.forward` 中 `ts==0` 分支调用 `stlstm_layer_diff` 但丢弃返回值——与 TF 原版行为一致(仅为初始化副作用),保留但需注释说明。
- `evaluate()` 中 `pred = gen_imgs[:, input_length-1:]` 与 `target = frames[:, input_length:]` 的对齐关系依赖"输出第 t 帧预测第 t+1 帧"的约定,正确但脆弱,建议加断言。
- `batch_ssim` 为全局均值 SSIM(非滑窗),与 `skimage.compare_ssim` 数值不可直接对比,论文表格复现时需注意。

---

## 第 1 部分:I/O 进程优化

### Task 1: 数据加载内存映射与零拷贝

**Files:**
- Modify: `dataset.py`
- Test: `tests/test_dataset.py`

**背景:** 当前 `MovingMNIST.__init__` 用 `np.load` 将整个 `.npz` 载入内存。Moving MNIST 训练集约 2GB,雷达回波数据可达数十 GB。改用 `mmap_mode='r'` 内存映射 + 按需读取,主进程内存占用从 O(数据集大小) 降为 O(batch)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_dataset.py
import numpy as np
import torch
from dataset import MovingMNIST


def make_fake_npz(path, n=16, length=20):
    data = np.random.randint(0, 256, (length, n, 64, 64), dtype=np.uint8)
    np.savez_compressed(path, train=data, test=data)


def test_mmap_lazy_load(tmp_path):
    p = tmp_path / "fake.npz"
    make_fake_npz(p)
    ds = MovingMNIST(str(p), total_length=20, input_length=10, is_train=True)
    assert len(ds) == 16
    sample = ds[0]
    assert sample.shape == (20, 1, 64, 64)
    assert sample.dtype == torch.float32
    assert sample.max() <= 1.0
```

- [ ] **Step 2: 运行确认失败** — `pytest tests/test_dataset.py -v`(当前无 tests 目录,先建)

- [ ] **Step 3: 实现 mmap 加载**

```python
# dataset.py MovingMNIST.__init__ 中替换 np.load
raw = np.load(data_path, mmap_mode='r')
self.data = raw['train'] if is_train else raw['test']
```

注意:`np.savez` 的 mmap 支持要求 numpy ≥ 1.26 且文件为未压缩 `.npy` 结构;对 `.npz` 压缩格式需先解包为独立 `.npy`。实现中检测格式并回退:

```python
def _load_array(data_path, key):
    if data_path.endswith('.npy'):
        return np.load(data_path, mmap_mode='r')
    raw = np.load(data_path)
    return raw[key]
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: Commit** `git commit -m "perf(data): memory-mapped dataset loading"`

### Task 2: DataLoader 多进程流水线调优

**Files:**
- Modify: `dataset.py`, `train.py`

**已完成基线:** `persistent_workers=True`、`prefetch_factor` 可配、`pin_memory=True`、`non_blocking=True`。

**追加优化:**

- [ ] **Step 1: worker 数自动探测** — `num_workers=0` 时按 `min(8, os.cpu_count())` 自动设置(仅当数据在慢速存储时):

```python
import os
def auto_num_workers(explicit):
    if explicit > 0:
        return explicit
    return min(8, os.cpu_count() or 1)
```

- [ ] **Step 2: `__getitem__` 内避免重复 `.astype` 分配** — 预分配输出缓冲,`np.divide(..., out=)` 原地归一化:

```python
def __getitem__(self, idx):
    seq = np.asarray(self.data[:, idx], dtype=np.float32)
    seq = seq[:self.total_length]
    np.divide(seq, 255.0, out=seq)
    return torch.from_numpy(seq).unsqueeze(1)
```

- [ ] **Step 3: 验证** — 用 `python -m torch.utils.bottleneck` 或简单计时对比 `num_workers=0` vs `4` 的每 epoch 耗时。
- [ ] **Step 4: Commit** `git commit -m "perf(data): worker auto-tuning and zero-copy normalization"`

### Task 3: Checkpoint 异步保存(消除保存阻塞)

**Files:**
- Modify: `train.py`

**背景:** `torch.save` 8M 参数模型约 32MB,同步保存阻塞训练循环。用后台线程异步写盘。

- [ ] **Step 1: 实现异步保存器**

```python
import threading

class AsyncSaver:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None

    def save(self, obj, path):
        if self._thread is not None:
            self._thread.join()
        self._thread = threading.Thread(
            target=self._save, args=(obj, path), daemon=True)
        self._thread.start()

    def _save(self, obj, path):
        with self._lock:
            torch.save(obj, path)

    def join(self):
        if self._thread is not None:
            self._thread.join()
```

- [ ] **Step 2: 在 `main()` 中替换 `torch.save(checkpoint, save_path)` 为 `saver.save(checkpoint, save_path)`,训练结束前 `saver.join()`**
- [ ] **Step 3: 验证** — 保存期间训练循环不中断,文件完整可 `torch.load`。
- [ ] **Step 4: Commit** `git commit -m "perf(io): async checkpoint saving"`

---

## 第 2 部分:CPU 线程优化

### Task 4: 线程池配置与 GIL 规避

**Files:**
- Modify: `train.py`

**背景:** PyTorch CPU 端涉及两类线程:① intra-op 并行(单个算子内部,`torch.set_num_threads`);② DataLoader worker 进程。两者争抢物理核会导致 oversubscription。

- [ ] **Step 1: 按物理核数划分**

```python
def setup_cpu_threads(num_workers):
    cpu_count = os.cpu_count() or 1
    intra = max(1, cpu_count // max(1, num_workers))
    torch.set_num_threads(intra)
    torch.set_num_interop_threads(1)
```

在 `main()` 创建 DataLoader 前调用。

- [ ] **Step 2: 验证** — `python -c "import torch; print(torch.get_num_threads())"` 与任务管理器确认 CPU 占用无 oversubscription。
- [ ] **Step 3: Commit** `git commit -m "perf(cpu): thread pool partitioning for dataloader workers"`

### Task 5: 评估阶段 CPU 指标计算卸载

**Files:**
- Modify: `metrics.py`, `train.py`

**背景:** `evaluate()` 中 PSNR/SSIM 等指标在 GPU 上计算后 `.item()` 同步回传,每个 batch 触发一次 device 同步。改为累积张量、epoch 末一次性回传。

- [ ] **Step 1: 指标函数增加 `reduce=False` 模式返回 per-batch 张量**

```python
def batch_psnr(gen_frames, gt_frames, reduce=True):
    ...
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item() if reduce else psnr.mean()
```

- [ ] **Step 2: `evaluate()` 累积到列表,循环结束后 `torch.stack(...).mean().item()` 一次同步**
- [ ] **Step 3: 验证** — 评估耗时下降,数值与逐步 `.item()` 一致(误差 < 1e-6)。
- [ ] **Step 4: Commit** `git commit -m "perf(eval): batch metric aggregation to reduce device sync"`

---

## 第 3 部分:GPU 推理加速

### Task 6: AMP 混合精度(已集成,需数值验证)

**Files:**
- Modify: `train.py`(已加 `--amp` 开关)
- Test: `tests/test_amp.py`

**风险点:** MIM 中大量 `sigmoid`/`tanh` 门控运算在 fp16 下可能下溢;`TensorLayerNorm` 的 `var` 计算建议保持 fp32。

- [ ] **Step 1: 写数值稳定性测试**

```python
# tests/test_amp.py
import torch
from mim import MIM

def test_amp_no_nan():
    if not torch.cuda.is_available():
        return
    model = MIM(1, 1, [2,1,64,64], hidden_dim=[32,32],
                total_length=8, input_length=4).cuda()
    frames = torch.randn(2, 8, 1, 64, 64, device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        out = model(frames)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
```

- [ ] **Step 2: 若出现 NaN,将 `TensorLayerNorm.forward` 强制 fp32:**

```python
def forward(self, x):
    orig_dtype = x.dtype
    x = x.float()
    ...
    return (self.gamma * x_norm + self.beta).to(orig_dtype)
```

- [ ] **Step 3: 基准对比** — fp32 vs amp 的 step 时间与 loss 曲线偏差(< 1%)。
- [ ] **Step 4: Commit** `git commit -m "perf(gpu): AMP numerical safety for layer norm"`

### Task 7: torch.compile 算子融合(已集成,需图断点治理)

**Files:**
- Modify: `mim.py`, `train.py`

**背景:** MIM 的递归循环含 Python 控制流(`if ts > 0`、`if i == 1`),会导致 dynamo 图断点(graph breaks)。`torch.compile` 仍可通过逐算子融合获益,但需验证无重编译风暴。

- [ ] **Step 1: 编译诊断**

```bash
TORCH_LOGS="graph_breaks,recompiles" python train.py --compile --epochs 1 ...
```

- [ ] **Step 2: 将 `ss_bool` 生成为固定形状张量(已满足),确保 batch 尺寸恒定(`drop_last=True` 已设置)避免重编译**
- [ ] **Step 3: 基准** — 记录 eager vs compile 的每 step 毫秒数(预期 1.2–1.8× 提速)。
- [ ] **Step 4: Commit** `git commit -m "perf(gpu): torch.compile integration and benchmark"`

### Task 8: 推理专用路径 + CUDA Graphs

**Files:**
- Create: `inference.py`

**背景:** 推理时无 scheduled sampling、无梯度,可构建静态图回放。MIM 每序列 19 个时间步,每步 4 层递归,CUDA Graphs 可消除 kernel launch 开销(约 19×4×N 次 launch)。

- [ ] **Step 1: 实现推理脚本**

```python
# inference.py
import torch
from mim import MIM

@torch.inference_mode()
def predict(model, frames, use_cuda_graph=False):
    model.eval()
    if use_cuda_graph and frames.is_cuda:
        static_input = frames.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = model(static_input)
        g.replay()
        return static_out
    return model(frames)
```

注意:递归网络内部每步分配新状态张量,首次 capture 前需 3 次 warmup 使 CUDA caching allocator 进入稳定池(参考 PyTorch CUDA Graphs 文档的 warmup 模式)。

- [ ] **Step 2: 若 capture 失败(动态分配),退化为 `torch.compile(mode='reduce-overhead')`(内部自动管理 CUDA Graphs)**
- [ ] **Step 3: 基准** — 单序列推理延迟 eager / compile / graph 三档对比。
- [ ] **Step 4: Commit** `git commit -m "feat(infer): dedicated inference path with CUDA Graphs"`

### Task 9: 显存优化(梯度检查点可选开关)

**Files:**
- Modify: `mim.py`

**背景:** 20 帧 × 4 层的展开图保留全部中间激活。对长序列(雷达回波 30+ 帧),用 `torch.utils.checkpoint` 按时间步分段重算,显存 O(T) → O(√T),代价约 30% 时间。

- [ ] **Step 1: 在 `MIM` 增加 `checkpoint_steps: int = 0` 参数,>0 时每 N 步包一层 `checkpoint`**
- [ ] **Step 2: 验证显存** — `torch.cuda.max_memory_allocated()` 对比。
- [ ] **Step 3: Commit** `git commit -m "perf(mem): optional temporal gradient checkpointing"`

---

## 第 4 部分:漏洞测试与安全加固

### Task 10: 安全测试套件

**Files:**
- Create: `tests/test_security.py`

- [ ] **Step 1: 恶意 checkpoint 拦截测试**

```python
# tests/test_security.py
import torch
import pytest

def test_malicious_checkpoint_blocked(tmp_path):
    class Evil:
        def __reduce__(self):
            return (eval, ("__import__('os').system('echo pwned')",))
    p = tmp_path / "evil.pth"
    torch.save({'model_state_dict': Evil()}, p)
    with pytest.raises(Exception):
        torch.load(p, weights_only=True)
```

- [ ] **Step 2: 输入形状校验测试** — 错误通道数/帧数的输入应抛出清晰错误而非 CUDA 断言:

```python
def test_input_validation():
    model = MIM(1, 1, [2,1,32,32], hidden_dim=[8,8],
                total_length=6, input_length=3)
    bad = torch.randn(2, 6, 3, 32, 32)
    with pytest.raises((RuntimeError, AssertionError)):
        model(bad)
```

- [ ] **Step 3: 在 `MIM.forward` 入口添加校验**

```python
assert frames.dim() == 5 and frames.shape[1] >= self.total_length \
    and frames.shape[2] == self.input_dims, \
    f"expect [B>={self.total_length},{self.input_dims},H,W], got {tuple(frames.shape)}"
```

- [ ] **Step 4: 运行** `pytest tests/test_security.py -v`
- [ ] **Step 5: Commit** `git commit -m "security: weights_only enforcement and input validation"`

### Task 11: 依赖与供应链检查

- [ ] **Step 1: 创建 `requirements.txt` 锁定最低安全版本**

```
torch>=2.5.1
numpy>=1.26
```

- [ ] **Step 2: `pip-audit`(如可用)扫描已知 CVE**
- [ ] **Step 3: Commit** `git commit -m "chore: pin dependency versions"`

---

## 第 5 部分:Bug 修复(遗留项)

### Task 12: 评估对齐断言与 SSIM 文档化

**Files:**
- Modify: `train.py`, `metrics.py`

- [ ] **Step 1: `evaluate()` 添加形状断言**

```python
assert pred.shape == target.shape, \
    f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
```

- [ ] **Step 2: `batch_ssim` docstring 注明"全局均值 SSIM,与滑窗 SSIM 数值不可直接对比"**
- [ ] **Step 3: Commit** `git commit -m "fix(eval): alignment assertion and ssim semantics doc"`

### Task 13: MIM.forward ts==0 分支语义注释

**Files:**
- Modify: `mim.py`

- [ ] **Step 1: 在 `ts==0` 的 `stlstm_layer_diff` 调用处添加说明** —— 该调用与 TF 原版一致,目的是让 diff 层在首步执行一次零输入前向以初始化内部状态,返回值按设计丢弃。
- [ ] **Step 2: Commit** `git commit -m "docs: clarify ts==0 diff-layer warmup semantics"`

---

## 第 6 部分:逻辑审计与代码 Review 机制

### Task 14: 单元测试全覆盖

**Files:**
- Create: `tests/test_mim.py`, `tests/test_metrics.py`

- [ ] **Step 1: 模型测试** — 覆盖:输出形状、`ss_bool=None`/有值两条路径、动态 batch、梯度流、state_dict 往返、fp64 数值冒烟。
- [ ] **Step 2: 指标测试** — 相同图像 PSNR≈100 / SSIM=1.0、CSI 边界(全命中=1,全误报→0)。
- [ ] **Step 3: 运行** `pytest tests/ -v --tb=short`,全绿。
- [ ] **Step 4: Commit** `git commit -m "test: unit tests for model and metrics"`

### Task 15: 静态检查与 CI

**Files:**
- Create: `.github/workflows/ci.yml`(如仓库启用 CI)

- [ ] **Step 1: ruff lint + pytest CPU 冒烟**

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install pytest ruff numpy
      - run: ruff check .
      - run: pytest tests/ -v
```

- [ ] **Step 2: Commit** `git commit -m "ci: lint and cpu smoke tests"`

---

## 优先级与执行顺序

| 阶段 | 任务 | 收益 | 风险 |
|------|------|------|------|
| P0(已完成) | 审计修复 #1–#13 | 正确性 + 安全 | — |
| P1 | Task 10(安全测试)、Task 14(单测) | 防回归 | 低 |
| P2 | Task 6(AMP 数值验证)、Task 7(compile 基准) | 训练提速 1.5–3× | 中(需 GPU) |
| P3 | Task 1–3(I/O)、Task 4–5(CPU) | 数据吞吐 | 低 |
| P4 | Task 8–9(推理/显存)、Task 15(CI) | 部署与工程化 | 中 |

## 验证命令汇总

```bash
pytest tests/ -v                                          # 全部单测
python train.py --dataset mnist --data_path <path> --amp --compile --num_workers 4
TORCH_LOGS="graph_breaks,recompiles" python train.py --compile --epochs 1 ...
python -m torch.utils.bottleneck train.py ...             # 性能剖析
```
