# MLA 当前状态

<!-- mla-loop-next: none -->

**R4 测试整理已完成验收，当前无后续已授权步骤。** 基线 `d433904`，分支 `feat/mla-cann-migration`。非量化 M17 与 INT8 M18–M21 已完成；本轮按用户 2026-10-19 的要求，仅保留公开功能 E2E，集中到 `tests/test_flash_attn_npu_v3_mla.py`，参照仓库原有测试风格。

## 当前交付

单个测试文件覆盖两种非量化表示/五布局、INT8 BSND/TND、独立数学 reference、mask/split/softcap、长度/stride/空值、metadata 复用、参数拒绝、fake/compile、图与生命周期。重复的阶段性包装合并为具名参数表；旧日常清单与新完整参数表不是逐节点一一对应，保留/合并/移除依据见 [design 第 7 节](design.md#7-验收口径)。

删除旧 `tests/mla/` 与兼容验收入口的 858 个受版本管理文件，清理公共 conftest 的私有 driver/旧节点映射依赖；README、开发约定、无头脚本及现有全量 CI 改用普通 pytest。非 910B/C 设备跳过 MLA 文件。生产逻辑、C++、构建参数、CATLASS 和 CANN 源未改；生产 Python 仅修正 docstring 的测试路径。

## 最近一次验收

证据根 `/tmp/flash-attention-npu-mla-46928a793876/mla-test-cleanup/`。最终 `acceptance-final.log` 和 `acceptance-final.xml`：**1,954 passed，0 failed/error/skipped，357.30 s**。

| 集合 | 本次实际通过 |
| --- | ---: |
| 新单文件 MLA E2E | 1,814 |
| 原有 metadata 签名/兼容 | 129 |
| 原有 dense/varlen/分页/FD 数值 | 6 |
| 原有 torch.compile | 3 |
| 原有图重放 | 2 |

旧驱动 dry-run/collect 的日常 E2E 为 905 项（31.5 s），见 `before-collect/summary.json`；新文件收集 1,814 项（14.24 s），见 `final-collect.log`；普通回归收集 140 项（21.02 s），见 `ordinary-collect.log`。新节点均实际执行，不继承旧节点通过数。边界组 186 项及契约组 293 项先行通过，最终再次完整执行，不重复累计。

初版夹具曾发现真实/FakeTensor 混用、负窗口归一化、全局空 KV 与 accumulator 的预期错误，均修正后重跑。失败日志保留为 `probe.log`、`public-probe.log`、`acceptance.log`、`boundaries.log` 和 `full-e2e.log`，不记为完整组通过。最终检查使用仓库 pre-commit（含 Ruff、YAML、Actionlint）、相关文档链接及 `git diff --check`；记录在 `final-quality.log`。源码/扩展校验值在 `checksums.txt`，旧源码副本在 `original.tar`，也可从 Git `d433904` 查询。

加载 CANN 环境后可直接复现完整 MLA：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_flash_attn_npu_v3_mla.py
```

本机最终组合命令如下；仓库外的 `run.py` 仅恢复已有 CANN 环境，未启用 native 包或旧测试插件。各普通节点沿用 M21 中未改变的适用回归：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 python3 /tmp/mla-m16-schedule/run.py python3 -m pytest -x -q \
  tests/test_flash_attn_npu_v3_mla.py \
  tests/test_flash_attn_npu_v3_metadata_signature.py \
  tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_func_metadata_bsnd \
  'tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_varlen_func_metadata_tnd[data_type0-1-1-1-512-1024-128-True]' \
  'tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_kvcache_metadata_bsnd[data_type0-1-1-1-1024-1024-128-128-False]' \
  'tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_kvcache_metadata_flash_decode[data_type1-8-1-64-2-False-0.0-0]' \
  tests/test_torch_compile_fa3.py::test_fa3_910_scheduler_metadata_torch_compile_correctness \
  tests/test_torch_compile_fa3.py::test_fa3_910_varlen_torch_compile_correctness \
  tests/test_torch_compile_fa3.py::test_fa3_910_fixed_torch_compile_correctness \
  tests/test_flash_attn_npu_v3_graph.py --durations=10 \
  --junitxml=/tmp/flash-attention-npu-mla-46928a793876/mla-test-cleanup/acceptance-final.xml
```

## 环境与生产基线

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；物理设备 7 为 Ascend910B3，可见逻辑设备 0。实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `1e5ee5b0c7d5aafbd8e0613e917f1040e6bc4fe5f2a6de7a7244d88fa9c6a015`，与 M21 一致；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。沙箱不能访问 NPU，设备检查和测试在宿主进行。生产逻辑未变，未重建扩展。

M21 历史正确性 1,677 项通过，原始证据在工作根 `mla-m21/`；39 个 INT8 配置 eager 平均耗时比 0.707703，通过；graph 1.197126，失败并按用户授权暂缓。INT8 固定源 Q/K scale 对 content/RoPE 合并分数及动态 P 量化的语义、历史独立数学限制仍在；当前可解析输入的数学测试不等价于一般量化固定源验收。固定源、Trap 和性能本轮未重跑，910C 未实测；本轮 E2E 通过不改变这些历史限制。完整历史结论见 `git show d433904:docs/mla/status.md`。

唯一下一步：无已授权开发步骤；本轮提交后停止。
