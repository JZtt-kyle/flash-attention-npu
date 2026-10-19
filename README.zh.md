# flash-attention-npu

<div align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-README.md-blue?style=flat-square" alt="English"></a> <a href="#"><img src="https://img.shields.io/badge/中文-README.zh.md-green?style=flat-square" alt="中文"></a>
</div>

## 介绍
FlashAttention 通过分块计算和内存感知算法提升训练和推理效率。当前其主流实现为面向 NVIDIA GPU 架构的 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。在昇腾平台迁移过程中，我们发现缺少与[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)接口兼容的实现，增加了适配难度。为此，本仓库参照 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 的核心设计，基于 [CANN/CATLASS](https://gitcode.com/cann/catlass) 框架及其样例代码，实现了适配昇腾 NPU 的 FlashAttention 算法。我们提供与 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 一致的调用接口，便于模型迁移，并支持后续面向昇腾 NPU 的大模型注意力算法改进和优化。

**本项目正在活跃开发中，欢迎参与讨论与贡献！**


## 准备

### 环境要求

- 硬件: 昇腾 910B / 910C / 950 NPU
- 系统: Linux
- 软件:
  - CANN >= 8.5.0
  - PyTorch >= 2.1.0
  - torch_npu >= 2.1.0 (与Pytorch版本相同)
- Python 依赖包
```bash
pip install packaging psutil
```
- 设置 CANN 环境变量
```bash
source [PATH_TO_ASCEND_HOME]/cann/set_env.sh
# 例如:
# source /usr/local/Ascend/cann/set_env.sh
```

### 安装步骤

#### 快速安装

```bash
pip install flash-attn-npu --no-build-isolation
```

#### 从源码编译

1. 拉取源码：
```bash
git clone https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git submodule update --init --recursive
```

2. 编译安装：

```bash
python setup.py install
```

  编译特定版本：

```bash
# 仅编译 v2
FLASH_ATTN_BUILD_VERSION=v2 python setup.py install

# 仅编译 v3
FLASH_ATTN_BUILD_VERSION=v3 python setup.py install

# 仅编译 v4
FLASH_ATTN_BUILD_VERSION=v4 python setup.py install
```

  编译面向特定NPU的版本：

```bash
# 仅编译 910 版本
FLASH_ATTN_BUILD_NPU=910 python setup.py install

# 仅编译 950 版本
FLASH_ATTN_BUILD_NPU=950 python setup.py install
```

## 测试

运行测试脚本：

```bash
# 测试 FlashAttention v2
pytest -q -s tests/test_flash_attn_npu_2.py

# 测试 FlashAttention v3
pytest -q -s tests/test_flash_attn_npu_v3.py

# 测试 FlashAttention v4
pytest -q -s tests/test_flash_attn_npu_v4.py
```

## 使用 msSanitizer 进行内存检测

编译时设置环境变量 `FLASH_ATTN_ENABLE_MSSANITIZER=TRUE` 即可使能 Ascend msSanitizer 内存异常检测：

```bash
# 使能内存检测编译
FLASH_ATTN_ENABLE_MSSANITIZER=TRUE FLASH_ATTN_BUILD_VERSION=v3 python setup.py install

# Ascend 910：静态插桩已编译进算子，直接运行测试即可
pytest -q -s tests/test_flash_attn_npu_v3.py

# Ascend 950：通过 msSanitizer 运行时注入检测内存（注意 `--` 分隔符）
mssanitizer --tool=memcheck -- python -m pytest -q -s tests/test_flash_attn_npu_v3.py
```

## 使用方法

### FlashAttention v2

#### flash_attn_with_kvcache

```python
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 表示无限上下文窗口
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """
    如果 k 和 v 不为 None，k_cache 和 v_cache 将被 *原地更新* 为 k 和 v 的新值。
    这对于增量解码非常有用：你可以传入上一步缓存的 key/value，
    用当前步骤的新 key/value 更新它们，并在一个内核中完成对更新后缓存的注意力计算。

    如果你传入 k / v，必须确保缓存足够大以容纳新值。
    例如，KV 缓存可以预分配最大序列长度，你可以使用 cache_seqlens 来跟踪批次中每个序列的当前序列长度。

    如果传入了 rotary_cos 和 rotary_sin，还会应用旋转位置编码。
    key @k 将在 cache_seqlens, cache_seqlens + 1 等位置被 rotary_cos 和 rotary_sin 旋转。
    如果是 causal 或 local（即 window_size != (-1, -1)），query @q 将在 cache_seqlens, cache_seqlens + 1 等位置被旋转。
    如果既不是 causal 也不是 local，query @q 将仅在 cache_seqlens 位置被旋转
    （即我们认为 @q 中的所有 token 都在位置 cache_seqlens）。

    支持多查询注意力和分组查询注意力（MQA/GQA），通过传入比 Q 头数少的 KV 来实现。
    注意 Q 的头数必须能被 KV 的头数整除。
    例如，如果 Q 有 6 个头，K、V 有 2 个头，那么 Q 的头 0、1、2 将关注 K、V 的头 0，
    Q 的头 3、4、5 将关注 K、V 的头 1。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。
    例如，如果 seqlen_q = 2 且 seqlen_k = 5，因果掩码（1 = 保留，0 = 掩码）为：
        1 1 1 1 0
        1 1 1 1 1
    如果 seqlen_q = 5 且 seqlen_k = 2，因果掩码为：
        0 0
        0 0
        0 0
        1 0
        1 1
    如果掩码的某一行全为零，输出将为零。

    如果 window_size != (-1, -1)，实现滑动窗口局部注意力。
    位置 i 的 query 只会关注 [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] 范围内的 key。

    注意：不支持反向传播。

    参数：
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: 如果没有 block_table，形状为 (batch_size_cache, seqlen_cache, nheads_k, headdim)；
            如果有 block_table（即分页 KV 缓存），形状为 (num_blocks, page_block_size, nheads_k, headdim)
            page_block_size 必须是 256 的倍数。
        v_cache: 如果没有 block_table，形状为 (batch_size_cache, seqlen_cache, nheads_k, headdim)；
            如果有 block_table（即分页 KV 缓存），形状为 (num_blocks, page_block_size, nheads_k, headdim)
        k [可选]: (batch_size, seqlen_new, nheads_k, headdim)。如果不为 None，我们将 k 从 cache_seqlens 指定的位置开始拼接到 k_cache。
        v [可选]: (batch_size, seqlen_new, nheads_k, headdim)。与 k 类似。
        rotary_cos [可选]: (seqlen_ro, rotary_dim / 2)。如果不为 None，我们对 k 和 q 应用旋转位置编码。
            仅在传入 k 和 v 时适用。rotary_dim 必须能被 16 整除。
        rotary_sin [可选]: (seqlen_ro, rotary_dim / 2)。与 rotary_cos 类似。
        cache_seqlens: int 或 (batch_size,)，dtype 为 torch.int32。KV 缓存的序列长度。
        block_table [可选]: (batch_size, max_num_blocks_per_seq)，dtype 为 torch.int32。
        cache_batch_idx: (batch_size,)，dtype 为 torch.int32。用于索引 KV 缓存的索引。
            如果为 None，我们假设批次索引为 [0, 1, 2, ..., batch_size - 1]。
            如果索引不唯一，且提供了 k 和 v，缓存中更新的值可能来自任何重复索引。
        softmax_scale: float。softmax 前对 QK^T 的缩放。默认为 1 / sqrt(headdim)。
        causal: bool。是否应用因果注意力掩码（例如用于自回归建模）。
        window_size: (left, right)。如果 != (-1, -1)，实现滑动窗口局部注意力。
        rotary_interleaved: bool。仅在传入 rotary_cos 和 rotary_sin 时适用。
            如果为 True，旋转位置编码将组合维度 0 & 1、2 & 3 等。如果为 False，
            旋转位置编码将组合维度 0 & rotary_dim / 2、1 & rotary_dim / 2 + 1
            （即 GPT-NeoX 风格）。
        alibi_slopes: (nheads,) 或 (batch_size, nheads)，fp32。
            将 (-alibi_slope * |i + seqlen_k - seqlen_q - j|) 的偏置加到
            query i 和 key j 的注意力分数上。

    约束：
        - 1 <= headdim <= 256。
        - nheads % nheads_k == 0。
        - dtype 仅支持 float16 / bfloat16；Q、K、V 的 dtype 必须一致。
        - Q、K、V 的最后一维必须连续（stride(-1) == 1）。
        - batch_size > 0。
        - softcap >= 0（0.0 表示关闭；Ascend 950 不支持 softcap）。
        - 不支持 alibi_slopes / rotary_cos / rotary_sin。
        - cache_seqlens / block_table 若传入须为 int32。
        - 不支持反向传播。

    返回：
        out: (batch_size, seqlen, nheads, headdim)。
    """
```

#### flash_attn_func

```python
def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 表示无限上下文窗口
    softcap=0.0,  # <=0.0 表示不启用
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """
    评估时应将 dropout_p 设为 0.0。

    支持多查询注意力和分组查询注意力（MQA/GQA），通过传入比 Q 头数少的 K、V 来实现。
    注意 Q 的头数必须能被 K、V 的头数整除。
    例如，如果 Q 有 6 个头，K、V 有 2 个头，那么 Q 的头 0、1、2 将关注 K、V 的头 0，
    Q 的头 3、4、5 将关注 K、V 的头 1。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。
    例如，如果 seqlen_q = 2 且 seqlen_k = 5，因果掩码（1 = 保留，0 = 掩码）为：
        1 1 1 1 0
        1 1 1 1 1
    如果 seqlen_q = 5 且 seqlen_k = 2，因果掩码为：
        0 0
        0 0
        0 0
        1 0
        1 1
    如果掩码的某一行全为零，输出将为零。

    如果 window_size != (-1, -1)，实现滑动窗口局部注意力。
    位置 i 的 query 只会关注 [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] 范围内的 key。

    参数：
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float。Dropout 概率。
        softmax_scale: float。softmax 前对 QK^T 的缩放。默认为 1 / sqrt(headdim)。
        causal: bool。是否应用因果注意力掩码（例如用于自回归建模）。
        window_size: (left, right)。如果 != (-1, -1)，实现滑动窗口局部注意力。
        softcap: float。大于 0 时激活 softcapping 注意力。
        alibi_slopes: (nheads,) 或 (batch_size, nheads)，fp32。
            将 (-alibi_slope * |i + seqlen_k - seqlen_q - j|) 的偏置加到
            query i 和 key j 的注意力分数上。
        deterministic: bool。是否使用反向传播的确定性实现（稍慢且占用更多内存）。
            前向传播始终是确定性的。
        return_attn_probs: bool。是否返回注意力概率。此选项仅用于测试，
            返回的概率不保证正确（缩放可能不正确）。

    约束：
        - 1 <= headdim <= 256。
        - nheads % nheads_k == 0。
        - dtype 仅支持 float16 / bfloat16；Q、K、V 的 dtype 必须一致。
        - Q、K、V 的最后一维必须连续（stride(-1) == 1）。
        - batch_size > 0。
        - softcap >= 0（0.0 表示关闭；Ascend 950 不支持 softcap）。
        - 0.0 <= dropout_p < 1.0；0.0 表示关闭 dropout。
        - 不支持 alibi_slopes。
        - 反向：headdim 须在 (0, 256]；Q 与 K 的 headdim 须相同。

    返回：
        out: (batch_size, seqlen, nheads, headdim)。
        softmax_lse [可选，return_attn_probs=True 时]: (batch_size, nheads, seqlen)。
            QK^T * scaling 每行的 logsumexp（即 softmax 归一化因子的对数）。
        S_dmask [可选，return_attn_probs=True 时]: (batch_size, nheads, seqlen, seqlen)。
            softmax 的输出（缩放可能不同），同时编码 dropout 模式
            （负值表示该位置被丢弃，非负值表示保留）。
    """
```

#### flash_attn_varlen_func

```python
def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 表示无限上下文窗口
    softcap=0.0,  # <=0.0 表示不启用
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """
    评估时应将 dropout_p 设为 0.0。

    支持变长序列：Q、K、V 按 token 拼接存储，通过 cu_seqlens 索引各序列边界。
    支持多查询注意力和分组查询注意力（MQA/GQA），通过传入比 Q 头数少的 K、V 来实现。
    注意 Q 的头数必须能被 K、V 的头数整除。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。
    例如，如果 seqlen_q = 2 且 seqlen_k = 5，因果掩码（1 = 保留，0 = 掩码）为：
        1 1 1 1 0
        1 1 1 1 1
    如果 seqlen_q = 5 且 seqlen_k = 2，因果掩码为：
        0 0
        0 0
        0 0
        1 0
        1 1
    如果掩码的某一行全为零，输出将为零。

    如果 window_size != (-1, -1)，实现滑动窗口局部注意力。
    位置 i 的 query 只会关注 [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] 范围内的 key。

    参数：
        q: (total_q, nheads, headdim)，total_q 为批次中 query token 总数。
        k: (total_k, nheads_k, headdim)，total_k 为批次中 key token 总数。
        v: (total_k, nheads_k, headdim)，total_k 为批次中 value token 总数。
        cu_seqlens_q: (batch_size + 1,)，dtype 为 torch.int32。用于索引 q 的累积序列长度。
        cu_seqlens_k: (batch_size + 1,)，dtype 为 torch.int32。用于索引 k、v 的累积序列长度。
        max_seqlen_q: int。批次中最大 query 序列长度。
        max_seqlen_k: int。批次中最大 key 序列长度。
        dropout_p: float。Dropout 概率。
        softmax_scale: float。softmax 前对 QK^T 的缩放。默认为 1 / sqrt(headdim)。
        causal: bool。是否应用因果注意力掩码（例如用于自回归建模）。
        window_size: (left, right)。如果 != (-1, -1)，实现滑动窗口局部注意力。
        softcap: float。大于 0 时激活 softcapping 注意力。
        alibi_slopes: (nheads,) 或 (batch_size, nheads)，fp32。
            将 (-alibi_slope * |i + seqlen_k - seqlen_q - j|) 的偏置加到
            query i 和 key j 的注意力分数上。
        deterministic: bool。是否使用反向传播的确定性实现（稍慢且占用更多内存）。
            前向传播始终是确定性的。
        return_attn_probs: bool。是否返回注意力概率。此选项仅用于测试。
        block_table [可选]: 分页 KV 缓存的块表。

    约束：
        - 1 <= headdim <= 256。
        - nheads % nheads_k == 0。
        - dtype 仅支持 float16 / bfloat16；Q、K、V 的 dtype 必须一致。
        - Q、K、V 的最后一维必须连续（stride(-1) == 1）。
        - batch_size > 0。
        - total_q > 0；Q 必须至少包含一个 query token。
        - softcap >= 0（0.0 表示关闭；Ascend 950 不支持 softcap）。
        - 0.0 <= dropout_p < 1.0；0.0 表示关闭 dropout。
        - 不支持 alibi_slopes。
        - cu_seqlens_q / cu_seqlens_k / block_table 若传入须为 int32。
        - 反向：headdim 须在 (0, 256]；Q 与 K 的 headdim 须相同。

    返回：
        out: (total_q, nheads, headdim)。
        softmax_lse [可选，return_attn_probs=True 时]: (nheads, total_q)。
            QK^T * scaling 每行的 logsumexp（即 softmax 归一化因子的对数）。
        S_dmask [可选，return_attn_probs=True 时]: (batch_size, nheads, seqlen, seqlen)。
            softmax 的输出，同时编码 dropout 模式。
    """
```

### FlashAttention v3

#### Ascend 910B/C 的 MLA

FP16/BF16 的三个 v3 入口均可传入压缩 MLA：`q/k` 的头维度为 64，`qv/v` 为 512，KV 头数为 1。
`q/k` 表示已经形成的 RoPE 分量，`qv` 是 query 内容，`v` 是共享的内容 K/V。
分数为 `(q @ k.T + qv @ v.T) * softmax_scale`，默认 scale 为 `1 / sqrt(576)`，
输出末维为 512。MLA 仅支持前向，会拒绝需要梯度的输入。

同时支持展开的 MLA prefill：把 128 维内容和 64 维 RoPE 拼为末维 192 的 `q/k`，
传入独立的 128 维 `v`，保持 `qv=None`。默认 scale 为 `1 / sqrt(192)`，
`Hq` 必须能被 `Hkv` 整除。实现按输入表示选择，压缩表示也支持多 query token。

BSND 使用 `flash_attn_func`，紧凑 TND 使用 `flash_attn_varlen_func`；
`flash_attn_with_kvcache` 接收 BSND/TND query 及只读的普通或分页 cache，
拒绝追加 KV、rotary 输入及 cache 重映射。支持 FP16/BF16、causal 和 window。
func/varlen 的 `num_splits` 默认 1，cache 默认 0（自动）；五种布局均接受 0..256，包含分页 TND。
两种表示均支持 BSND/BSND、TND/TND、TND/BSND、BSND/ND 分页 KV、TND/ND 分页 KV
五种布局。支持非负有限 `softcap`，零值关闭；`pack_gqa=None/True/False` 分别选择
默认、打包、不打包的调度，`sm_margin` 预留 AIC 核，范围为 `0 <= sm_margin < physical_AIC`。
要求 `attention_chunk=0`、`rotary_interleaved=True`。任意 mask、外部 NZ cache、
追加 KV 和反向不在本次 MLA 范围内。INT8 使用下方单列的较窄契约。

后端使用 CANN 的 910B 和 `Ascend910_93`（910C）平台资源，并检查 C1V2 与内存容量。
实机验收使用 910B；910C 核对 CANN 平台配置兼容，不宣称已完成 910C 实机验收。
布局、页大小、长度和空值规则见 [MLA 契约](docs/mla/design.md#2-fa3-入口与后端边界)，
实测边界见 [status](docs/mla/status.md)。下文通用约束用于普通 attention，MLA 以此契约为准。

```python
import torch
import torch_npu
from flash_attn_npu_3 import flash_attn_func

q = torch.randn(1, 2, 4, 64, device="npu", dtype=torch.float16)
k = torch.randn(1, 16, 1, 64, device="npu", dtype=torch.float16)
qv = torch.randn(1, 2, 4, 512, device="npu", dtype=torch.float16)
v = torch.randn(1, 16, 1, 512, device="npu", dtype=torch.float16)
out, lse = flash_attn_func(q, k, v, qv=qv, return_attn_probs=True)
assert out.shape == (1, 2, 4, 512) and lse.shape == (1, 4, 2)
```

func/varlen 的 `return_attn_probs=True` 返回 `(out, lse)`；cache 的
`return_softmax_lse=True` 返回 `(out, lse, out_accum, lse_accum)`，后两项在单 split
时为空，在 split 执行时是真实的 FP32 分片结果。不开启返回开关时均只返回 `out`。

`get_scheduler_metadata` 接收连续的设备 int32 KV 长度。记录绑定几何、长度、dtype、
mask、split 和调度选项；这些条件变化后须重新生成，匹配条件的多层可以复用。
ND 分页 cache 为 `[num_pages, page_size, Hkv, D]`，页大小是 `[16, 1024]` 内的 16 倍数，
逻辑容量不超过 16384，页表为 int32。

两种表示的五种布局均支持图捕获和重放。捕获后保持 tensor 的 shape、stride、存储地址
及标量选项不变。重放期间修改设备长度时，应在图内重新生成 metadata，或按正确的 stream
依赖更新图持有的外部记录；旧记录不会自行更新。图正确性与图性能分别验收，最新图性能
实测仍超过默认门槛，见 [当前证据与限制](docs/mla/status.md)。

[可运行示例](tests/mla/support/mla_examples.py) 演示压缩 dense、prefill varlen、分页
metadata 跨调用复用，以及长度变化时的图重放。构建 v3 后在本仓运行：

```bash
python3 -m tests.mla.support.mla_examples --device 0 --dtype fp16
python3 -m tests.mla.support.mla_examples --device 0 --dtype bf16
```

#### Ascend 910B/C 的 INT8 压缩 MLA

`flash_attn_with_kvcache` 和 `get_scheduler_metadata` 同时支持 INT8 共享 512 维内容
与 64 维 RoPE、BSND/TND query 及 ND 分页 cache。这是 FA3 的 NPU 扩展，所有 tensor
必须位于同一 NPU：

| 参数 | dtype 与 shape |
| --- | --- |
| `q` / `qv` | BF16 RoPE / INT8 内容；`[B,Sq,Hq,64/512]` 或 `[Tq,Hq,64/512]` |
| `k_cache` / `v_cache` | BF16 RoPE / 共享 INT8 内容 K/V；`[Np,128,1,64/512]` |
| `mla_q_scale` | FP32 `[B,Sq,Hq]` 或 `[Tq,Hq]`，每个 query token/head 一个 scale |
| `mla_k_scale` / `mla_v_scale` | 各为 FP32 `[1]`，共享内容 K/V 分别使用各自 scale |
| `cache_seqlens` / `page_table` | int32 `[B]` / `[B,max_pages_per_sequence]` |
| TND `cu_seqlens_q` | int32 `[B+1]`，首项 0、末项 `Tq`；同时传入 `max_seqlen_q` |

三个 MLA scale 均须通过关键字传入，不能用 `q_descale/k_descale/v_descale` 替代。
metadata 使用 `qkv_dtype=torch.int8`、`headdim=64`、`headdim_v=512`、
`num_heads_kv=1`、`page_size=128`。更新 scale 值可以复用记录；改变绑定的长度或 TND
前缀须重新生成。O 为末维 512 的 BF16，LSE 为 FP32 `[B,Hq,Sq]` 或 `[Hq,Tq]`，
空 KV 行返回 O=0、LSE=`-inf`。`return_softmax_lse=True` 返回四项；自动 split 的 FP32
accumulator 有 26 个槽，未使用槽为 0 / `-inf`。

`Hq` 支持 1/2/4/8/16/32/64/128，每段 query 不超过 32。单 query 支持无 mask 或等价
causal，多 query 须设置右下对齐的 `causal=True`。选项为 `num_splits=0/1`、
`softcap=0`、`window_size=(-1,-1)`、`pack_gqa=None/True`、`sm_margin=0`。
非 PA、192/128 INT8、其他页大小、显式多 split、有限窗口、softcap、offset、postquant
和混合量化模式均明确拒绝。

数值保留固定 CANN INT8 算法及动态概率量化；源的 Q/K scale 作用于内容与 **RoPE 合并后**
的分数，与仅反量化内容点积的数学公式不同。固定源逐元素一致和独立 FP32/FP64 的精度限制
分别记录，见 [INT8 契约](docs/mla/design.md#34-int8-压缩-mla-扩展) 和
[覆盖映射](docs/mla/design.md#78-int8-收尾映射)。

[INT8 可运行示例](tests/mla/support/mla_int8_examples.py) 演示两种布局、metadata 复用、
scale 更新及 KV 长度变化时的图重放：

```bash
python3 -m tests.mla.support.mla_int8_examples --device 0 --layout bsnd
python3 -m tests.mla.support.mla_int8_examples --device 0 --layout tnd
```

#### flash_attn_with_kvcache

```python
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    *,
    mla_q_scale=None,
    mla_k_scale=None,
    mla_v_scale=None,
):
    """
    v3 版本的 KV 缓存接口，相比 v2 增加了更多功能。

    如果 k 和 v 不为 None，k_cache 和 v_cache 将被 *原地更新* 为 k 和 v 的新值。
    这适用于普通 attention 的增量解码；MLA cache 只读。

    支持多查询注意力和分组查询注意力（MQA/GQA）。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。

    如果 window_size != (-1, -1)，实现滑动窗口局部注意力。

    注意：不支持反向传播。

    参数：
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: 如果没有 page_table，形状为 (batch_size_cache, seqlen_cache, nheads_k, headdim)；
            如果有 page_table（即分页 KV 缓存），形状为 (num_blocks, page_block_size, nheads_k, headdim)
            普通 attention 的 page_block_size 可为任意值；MLA 须满足上方链接的页大小规则。
        v_cache: 如果没有 page_table，形状为 (batch_size_cache, seqlen_cache, nheads_k, headdim_v)；
            如果有 page_table，形状为 (num_blocks, page_block_size, nheads_k, headdim_v)。
        k [可选]: (batch_size, seqlen_new, nheads_k, headdim)。如果不为 None，从 cache_seqlens 指定的位置开始拼接到 k_cache。
        v [可选]: (batch_size, seqlen_new, nheads_k, headdim_v)。与 k 类似。
        qv [可选]: (batch_size, seqlen, nheads, headdim_v)。
        rotary_cos [可选]: (seqlen_ro, rotary_dim / 2)。旋转位置编码的 cos 值。
        rotary_sin [可选]: (seqlen_ro, rotary_dim / 2)。旋转位置编码的 sin 值。
        cache_seqlens: int 或 (batch_size,)，dtype 为 torch.int32。KV 缓存的序列长度。
        cache_batch_idx: (batch_size,)，dtype 为 torch.int32。用于索引 KV 缓存的索引。
        cache_leftpad: (batch_size,)，dtype 为 torch.int32。KV 缓存起始索引。
        page_table [可选]: (batch_size, max_num_blocks_per_seq)，dtype 为 torch.int32。
        cu_seqlens_q [可选]: 变长模式下的 query 累积序列长度。
        cu_seqlens_k_new [可选]: 变长模式下的新 key 累积序列长度。
        max_seqlen_q [可选]: 变长模式下的最大 query 序列长度。
        rotary_seqlens [可选]: 旋转位置编码的序列长度。
        q_descale, k_descale, v_descale: 可选，用于 FP8 量化的反缩放因子。
        softmax_scale: float。有 qv 时缩放 q @ k.T + qv @ v.T，否则缩放 q @ k.T。
            有 qv 时默认为 1 / sqrt(headdim + headdim_v)，否则为 1 / sqrt(headdim)。
        causal: bool。是否应用因果注意力掩码。
        window_size: (left, right)。如果 != (-1, -1)，实现滑动窗口局部注意力。
        attention_chunk: int。注意力分块大小。
        softcap: float。大于 0 时激活 softcapping 注意力。
        rotary_interleaved: bool。旋转位置编码模式。
        scheduler_metadata: 可选，调度器元数据。
        num_splits: int。如果 > 1，将 key/value 在序列维度上分割成这么多块。
            如果 num_splits == 1，不分割。如果 num_splits == 0，自动选择。
        pack_gqa: bool。是否打包 GQA 以提高性能。
        sm_margin: int。SM 边际，用于调优。
        return_softmax_lse: bool。是否返回注意力分数的 logsumexp。

    约束：
        - 1 <= headdim <= 256。
        - nheads % nheads_k == 0。
        - dtype 仅支持 float16 / bfloat16；Q、K、V 的 dtype 必须一致。
        - Q、K、V 的最后一维必须连续（stride(-1) == 1）。
        - batch_size > 0。
        - softcap >= 0（0.0 表示关闭；Ascend 950 不支持 softcap）。
        - 不支持 alibi_slopes / rotary / FP8 descales / attention_chunk / pack_gqa。
        - cache_seqlens / page_table / cu_seqlens_* 若传入须为 int32。
        - 不支持反向传播。

    返回：
        out: (batch_size, seqlen, nheads, headdim_v)。
        return_softmax_lse=True 时返回 (out, softmax_lse, out_accum, softmax_lse_accum)。
        softmax_lse [可选]: (batch_size, nheads, seqlen)。FP32 每行 logsumexp，有 qv 时包含其贡献。
    """
```

#### flash_attn_func

```python
def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """
    v3 版本的标准注意力接口，相比 v2 增加了 FP8 反量化、attention_chunk 等参数。

    支持多查询注意力和分组查询注意力（MQA/GQA），通过传入比 Q 头数少的 K、V 来实现。
    注意 Q 的头数必须能被 K、V 的头数整除。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。
    例如，如果 seqlen_q = 2 且 seqlen_k = 5，因果掩码（1 = 保留，0 = 掩码）为：
        1 1 1 1 0
        1 1 1 1 1
    如果 seqlen_q = 5 且 seqlen_k = 2，因果掩码为：
        0 0
        0 0
        0 0
        1 0
        1 1
    如果掩码的某一行全为零，输出将为零。

    如果 window_size != (-1, -1)，实现滑动窗口局部注意力。
    位置 i 的 query 只会关注 [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] 范围内的 key。

    参数：
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim_v)
        softmax_scale: float。有 qv 时缩放 q @ k.T + qv @ v.T，否则缩放 q @ k.T。
            有 qv 时默认为 1 / sqrt(headdim + headdim_v)，否则为 1 / sqrt(headdim)。
        causal: bool。是否应用因果注意力掩码（例如用于自回归建模）。
        qv [可选]: (batch_size, seqlen, nheads, headdim_v)。
        q_descale, k_descale, v_descale: 可选，用于 FP8 量化的反缩放因子。
        window_size: (left, right)。如果 != (-1, -1)，实现滑动窗口局部注意力。
        attention_chunk: int。注意力分块大小。
        softcap: float。大于 0 时激活 softcapping 注意力。
        num_splits: int。如果 > 1，将 key/value 在序列维度上分割成这么多块。
            如果 num_splits == 1，不分割。如果 num_splits == 0，自动选择。
        pack_gqa: bool。是否打包 GQA 以提高性能。
        deterministic: bool。是否使用反向传播的确定性实现。
        sm_margin: int。SM 边际，用于调优。
        return_attn_probs: bool。为 True 时返回 (out, softmax_lse)。

    约束：
        - 1 <= headdim <= 256。
        - nheads % nheads_k == 0。
        - dtype 仅支持 float16 / bfloat16；Q、K、V 的 dtype 必须一致。
        - Q、K、V 的最后一维必须连续（stride(-1) == 1）。
        - batch_size > 0。
        - softcap >= 0（0.0 表示关闭；Ascend 950 不支持 softcap）。
        - 不支持 alibi_slopes / FP8 descales / attention_chunk / pack_gqa。
        - 反向：headdim 须在 (0, 256]；Q 与 K 的 headdim 须相同；反向暂不支持 seqused_*。

    返回：
        out: (batch_size, seqlen, nheads, headdim_v)。
        softmax_lse [可选，return_attn_probs=True 时]: (batch_size, nheads, seqlen)。
            QK^T * scaling 每行的 logsumexp。
    """
```

#### flash_attn_varlen_func

```python
def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """
    v3 版本的变长序列注意力接口。

    支持变长序列：Q、K、V 按 token 拼接存储，通过 cu_seqlens 索引各序列边界。
    支持多查询注意力和分组查询注意力（MQA/GQA）。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。

    如果 window_size != (-1, -1)，实现滑动窗口局部注意力。

    参数：
        q: (total_q, nheads, headdim)，total_q 为批次中 query token 总数。
        k: (total_k, nheads_k, headdim)，total_k 为批次中 key token 总数。
        v: (total_k, nheads_k, headdim_v)，total_k 为批次中 value token 总数。
        cu_seqlens_q: (batch_size + 1,)，dtype 为 torch.int32。用于索引 q 的累积序列长度。
        cu_seqlens_k: (batch_size + 1,)，dtype 为 torch.int32。用于索引 k、v 的累积序列长度。
        max_seqlen_q: int。批次中最大 query 序列长度。
        max_seqlen_k: int。批次中最大 key 序列长度。
        seqused_q [可选]: 实际使用的 query 序列长度。
        seqused_k [可选]: 实际使用的 key 序列长度。
        softmax_scale: float。有 qv 时缩放 q @ k.T + qv @ v.T，否则缩放 q @ k.T。
            有 qv 时默认为 1 / sqrt(headdim + headdim_v)，否则为 1 / sqrt(headdim)。
        causal: bool。是否应用因果注意力掩码。
        qv [可选]: 额外的 query value 张量。
        q_descale, k_descale, v_descale: 可选，用于 FP8 量化的反缩放因子。
        window_size: (left, right)。如果 != (-1, -1)，实现滑动窗口局部注意力。
        attention_chunk: int。注意力分块大小。
        softcap: float。大于 0 时激活 softcapping 注意力。
        num_splits: int。key/value 序列维度分割块数。
        pack_gqa: bool。是否打包 GQA 以提高性能。
        deterministic: bool。是否使用反向传播的确定性实现。
        sm_margin: int。SM 边际，用于调优。
        return_attn_probs: bool。为 True 时返回 (out, softmax_lse)。

    约束：
        - 1 <= headdim <= 256。
        - nheads % nheads_k == 0。
        - dtype 仅支持 float16 / bfloat16；Q、K、V 的 dtype 必须一致。
        - Q、K、V 的最后一维必须连续（stride(-1) == 1）。
        - batch_size > 0。
        - total_q > 0；Q 必须至少包含一个 query token。
        - softcap >= 0（0.0 表示关闭；Ascend 950 不支持 softcap）。
        - 不支持 alibi_slopes / FP8 descales / attention_chunk / pack_gqa。
        - cu_seqlens_q / cu_seqlens_k 若传入须为 int32。
        - 反向：headdim 须在 (0, 256]；Q 与 K 的 headdim 须相同；反向暂不支持 seqused_*。

    返回：
        out: (total_q, nheads, headdim_v)。
        softmax_lse [可选，return_attn_probs=True 时]: (nheads, total_q)。
            QK^T * scaling 每行的 logsumexp。
    """
```

### FlashAttention v4

#### flash_attn_varlen_func
```python
def flash_attn_varlen_func(
    q,
    k,
    v,
    qv=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size=(-1, -1),  # -1 means infinite context window
    learnable_sink: Optional[torch.Tensor] = None,
    softcap=0.0, # 0.0 means deactivated
    num_splits=0,    # Can be tuned for speed
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors=None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    return_lse: bool = False,
):
    """
    v4 版本的变长序列注意力接口。

    支持变长序列：Q、K、V 按 token 拼接存储，通过 cu_seqlens 索引各序列边界。
    支持多查询注意力和分组查询注意力（MQA/GQA）。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。

    如果 window_size 非 (None, None) / (-1, -1)，实现滑动窗口局部注意力。

    支持可选分页 KV Cache（仅正向）：
    通过 page_table 指定 KV cache 页表，k/v 可采用分页格式存储。

    参数：
        q: (total_q, nheads, headdim)，total_q 为批次中 query token 总数。当不使用变长模式时，为 (batch_size, seqlen, nheads, headdim)。
        k: (total_k, nheads_k, headdim)，total_k 为批次中 key token 总数。当使用 paged KV cache 时，为 (num_pages, page_size, nheads_k, headdim)。
        v: (total_k, nheads_k, headdim_v)，total_k 为批次中 value token 总数。当使用 paged KV cache 时，为 (num_pages, page_size, nheads_k, headdim_v)。
        cu_seqlens_q: (batch_size + 1,)，dtype 为 torch.int32。用于索引 q 的累积序列长度。
        cu_seqlens_k: (batch_size + 1,)，dtype 为 torch.int32。用于索引 k、v 的累积序列长度。
        max_seqlen_q: int。批次中最大 query 序列长度。
        max_seqlen_k: int。批次中最大 key 序列长度。
        min_seqlen_k [可选]: key 的最小序列长度。
        seqused_q [可选]: (batch_size,)，dtype 为 torch.int32。每个 batch 实际使用的 query 序列长度。（反向暂不支持）
        seqused_k [可选]: (batch_size,)，dtype 为 torch.int32。每个 batch 实际使用的 key 序列长度。（反向暂不支持）
        gather_kv_indices [可选]: KV 索引。
        page_table [可选]: (batch_size, max_num_pages_per_seq)，dtype 为 torch.int32。分页 KV Cache 的页表。（仅正向）
        softmax_scale: float。softmax 前对 QK^T 的缩放因子。默认为 1 / sqrt(headdim + (headdim_v if qv is not None else 0))。
        causal: bool。是否应用因果注意力掩码。
        qv [可选]: (batch_size, seqlen, nheads, headdim_v)。用于 cross-attention。
        window_size: (left, right)。如果 != (-1, -1)，实现滑动窗口局部注意力。
        softcap: float。大于 0 时激活 softcapping 注意力。
        num_splits: int。key/value 序列维度分割块数。如果为 0，则根据启发式方法自动确定分割数量。
        pack_gqa: bool。是否打包 GQA 以提高性能。
        deterministic: bool。是否使用反向传播的确定性实现。
        score_mod [可选]: 自定义 score 修改函数。（NPU 暂不支持）
        score_mod_bwd [可选]: 反向传播阶段的自定义 score 修改函数。（NPU 暂不支持）
        mask_mod [可选]: 自定义 attention mask。（NPU 暂不支持）
        block_sparse_tensors [可选]: block sparse tensor。（NPU 暂不支持）
        aux_tensors [可选]: 用于 score_mod 的辅助 tensor。（NPU 暂不支持）
        aux_scalars [可选]: 用于 score_mod/mask_mod 的辅助标量。（NPU 暂不支持）
        return_lse: bool。是否返回 attention scores 的 logsumexp。

    约束：
        - 1 <= headdim <= 256。
        - nheads % nheads_k == 0。
        - dtype 仅支持 float16 / bfloat16；Q、K、V 的 dtype 必须一致。
        - Q、K、V 的最后一维必须连续（stride(-1) == 1）。
        - batch_size > 0。
        - softcap >= 0（0.0 表示关闭；Ascend 950 不支持 softcap）。
        - 不支持 pack_gqa / learnable_sink / score_mod / mask_mod / min_seqlen_k / gather_kv_indices。
        - cu_seqlens_* / seqused_* / page_table 若传入须为 int32。
        - 在 Ascend 950 上，分页 KV Cache 的 page_size 仅支持 128、256、512 或 1024。
        - 反向：headdim 须在 (0, 256]；Q 与 K 的 headdim 须相同；反向暂不支持 seqused_*。

    返回：
        out: (total_q, nheads, headdim_v) 或稠密 (batch_size, seqlen, nheads, headdim_v)。
        softmax_lse [可选，return_lse=True 时]: 变长为 (nheads, total_q)；稠密为 (batch_size, nheads, seqlen)。
            QK^T * scaling 每行的 logsumexp。
    """
```

## 特性

#### flash_attn_with_kvcache

| 特性                | v2 | v3 |
| ------------------- | -- | -- |
| FP16 (float16)      | ✅ | ✅ |
| BF16 (bfloat16)     | ✅ | ✅ |
| 因果注意力 (Causal) | ✅ | ✅ |
| 滑动窗口注意力      | ✅ | ✅ |
| MQA/GQA             | ✅ | ✅ |
| 分页 KV 缓存        | ✅ | ✅ |
| 旋转位置编码 (RoPE) | -  | -  |
| ALiBi               | ✅  | -  |
| Softcapping         | ✅ | ✅ |
| FP8 量化            | -  | -  |
| 变长序列            | ✅ | ✅ |

#### flash_attn_func

| 特性                | v2 | v3 |
| ------------------- | -- | -- |
| FP16 (float16)      | ✅ | ✅ |
| BF16 (bfloat16)     | ✅ | ✅ |
| 因果注意力 (Causal) | ✅ | ✅ |
| 滑动窗口注意力      | ✅ | ✅ |
| MQA/GQA             | ✅ | ✅ |
| 反向传播            | ✅ | ✅ |
| ALiBi               | ✅  | -  |
| Softcapping         | ✅ | ✅ |
| FP8 量化            | -  | -  |
| Dropout             | ✅ | -  |

#### flash_attn_varlen_func

| 特性                | v2 | v3 | v4 |
| ------------------- | -- | -- | -- |
| FP16 (float16)      | ✅ | ✅ | ✅ |
| BF16 (bfloat16)     | ✅ | ✅ | ✅ |
| 因果注意力 (Causal) | ✅ | ✅ | ✅ |
| 滑动窗口注意力      | ✅ | ✅ | ✅ |
| MQA/GQA             | ✅ | ✅ | ✅ |
| 反向传播            | ✅ | ✅ | ✅ |
| 变长序列            | ✅ | ✅ | ✅ |
| 分页 KV 缓存        | ✅ | ✅  | ✅ |
| ALiBi               | ✅  | -  | -  |
| Softcapping         | ✅ | ✅ | ✅ |
| FP8 量化            | -  | -  | -  |
| Dropout             | ✅ | -  | -  |

## 许可证

本项目采用 BSD 3-Clause License 开源协议。详情请参阅 [LICENSE](./LICENSE) 文件。
