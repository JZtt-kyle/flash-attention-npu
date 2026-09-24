# MLA 当前交接

## 状态与下一步

- 当前步骤：M04a 已完成，512+64 的连续 BSND 短 KV CATLASS 前向已接通。
- 当前分支：`feat/mla-cann-migration`；接手基点 `2566ed3`。方案已确认，本轮一个提交后停止。
- 本步产物：[MLA host](../../csrc/ascend910/flash_attn_npu_3/mla/mla_host.h)、[Kernel](../../csrc/ascend910/flash_attn_npu_3/mla/mla_short_kernel.hpp)、[softmax/LSE Block](../../csrc/ascend910/flash_attn_npu_3/mla/mla_short_epilogue.hpp)、[精度与边界检查](../../tests/test_flash_attn_npu_v3_mla_short.py)。
- 唯一下一步：M04b，扩展长 KV 在线 softmax 与输出累积，验收 KV 尾块、单/多 query、D01/长 KV 精度及同轮 CANN 性能；其余未实现组合继续明确拒绝。

M04 在开始实现前拆为 M04a（单 KV tile）与 M04b（长 KV 在线累积）。本步不是完整 MLA 迁移验收，也没有生产 CANN 回退。

## 当前入口行为

- `flash_attn_func(q, k, v, qv=...)` 支持 FP16/BF16、`Dq=64/Dv=512/Hkv=1`、连续 BSND、正 batch/query 长度、KV 长度 `1..128`、无 mask、`num_splits=1`。单/多 query 统一按表示分派。
- 默认 scale 为 `1/sqrt(576)`，显式有限 scale 保持 FP32；正值、零及负值均已验证。O 为 `[B,Sq,Hq,512]`、输入 dtype；可选 LSE 为自然对数 FP32 `[B,Hq,Sq]`。不拼接 576 维输入，不展开所有 KV 头，不复制完整输入。
- 复用固定 CATLASS 的 QK+RoPE / PV Block 和 reduction Tile，PV 复用 L1 内容 KV。本仓新增单 tile Kernel、FP32 scale/归一化/LSE Block 和轻量 host 参数；生成器新增两种 dtype 的独立编译单元。
- 真实 qv 调用直接交给 C++，在分配输出前完成全部校验；注册 fake 执行等价元数据检查。去掉重复 Python 校验及普通 FA 参数转换，保持非法参数明确拒绝。连续非零 storage offset（含 offset=1）可用；外层非连续 stride 留待 M06。
- MLA prefill、MLA varlen/kvcache 公共入口、长/空 KV、零 query、mask/window、split 自动/多 split、softcap、pack_gqa/sm_margin、metadata、追加 KV、预分配 out、实际长度/页表/rotary/descale 和梯度请求仍明确拒绝，不返回占位计算结果。
- 短路径使用 host 静态 shape 调度，无长度 D2H、无调度 H2D。MLA metadata 与图执行尚未验收；普通 FA3 的 byte metadata 协议保持兼容。运行时读取 Cube/Vector 资源并按 sub-block 数分工；本次仅实测 910B。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手核对 | PASS：分支/提交与交接一致、工作区干净；M03 checksum 全匹配，原扩展哈希一致；固定 CANN 源和 CATLASS 未修改 |
| 构建溯源 | PASS：仅 v3/910；新增两种 MLA dtype 对象，`flash_api.o` 重编，depfile 含本仓 MLA headers；既有普通 FA3/AICPU 对象保持哈希，最终扩展重新链接 |
| 910B MLA | PASS：`374 passed in 21.41s`，无 skip；其中 M03 契约 233 项、新短路径 141 项，覆盖 O/LSE、单/多 query、头数/KV 尾块、重复执行、输入不改写、连续 storage offset 及真实/fake/C++ 边界 |
| 固定源码 CANN | PASS：12 个 shape/scale × FP16/BF16，共 24 组；CANN 与新实现分别通过原双 reference/comparator，阈值未改；24 次 FIA launch 全部匹配配套 vendor binary |
| 普通 FA3 | PASS：`11 passed in 28.64s`，无 skip；dense/varlen/kvcache/FlashDecode、metadata 和编译图，含 dense/varlen backward |
| 既有 CPU | PASS：reference/profile/provenance 共 `88 passed in 20.68s` |
| 性能 | PASS：最终四例同轮交替测量，kernel 与端到端逐例比值均小于 1.0；详见下表；没有以平均加速掩盖单例退化 |
| 仓库质量 | 本步代码钩子通过；提交前 staged 质量、文档链接/围栏及空白检查见 `quality-staged.log` / `document-check.log` |

最终性能：`B=1, Sk=65, Hq=16, Hkv=1`，Sq 分别为 1（D02-q1）与 7（原 D02）；相同输入、dtype、scale 和物理 7 号卡。时间单位为微秒，中位数按原始逐次记录计算。

| 用例 | CANN kernel | MLA kernel | kernel 比值 | CANN 调用 | MLA 调用 | 调用比值 |
| --- | --- | --- | --- | --- | --- | --- |
| D02-q1 FP16 | 10.7305 | 7.1400 | 0.6654 | 288.965 | 266.795 | 0.9233 |
| D02-q1 BF16 | 10.6900 | 7.1805 | 0.6717 | 287.390 | 257.225 | 0.8950 |
| D02 FP16 | 18.5205 | 12.2600 | 0.6620 | 280.730 | 249.990 | 0.8905 |
| D02 BF16 | 18.7810 | 12.2500 | 0.6523 | 280.855 | 248.555 | 0.8850 |

预热每实现 10 次，普通同步调用为 7 轮 × 50 对 AB/BA 交替；独立 profiler 为 3 轮 × 10 对，240 次调用均对应恰好一个 kernel，无 Slice 转换。调用时间包含输出/workspace 分配、校验和静态调度；MLA metadata 生成/复用及图模式不在本步性能结论内。配对 bootstrap 的比值 95% 区间上界最高 0.933，辅助确认噪声未掩盖退化；验收门槛仍是逐例比值 `<=1.0`。

首次性能测量 kernel 已快，但调用比值为 1.262..1.306，判为失败；主机分析定位重复 Python 校验，修复后的独立两轮调用比值分别为 0.943..0.983、0.885..0.923。首次编译的命名空间/设备地址错误、cache 目录缺失、尾部清零的 UB 对齐与 mask 数组顺序错误均已修复并重验；失败日志保留，不能拿早期产物当作闭环证据。

## 证据与复现

持久证据根目录：

`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04a-validation/`

- `environment-before.json`、`takeover-audit.json`、`previous-checksums.log`：接手环境、源码和旧证据。
- `build-*-command.json` / `build-*.log` / `build-*-exit.json`、`build-provenance-final.json`：增量依赖、对象及扩展构建溯源；`compare-c.json` 同时记录实际加载扩展路径和哈希。
- `mla-final-*` / `mla-final.log` / `mla-final.xml`、`regression-final-*` / `regression-final.log` / `regression-final.xml`：完整最终测试命令和结果。
- `cann-accuracy.py` / `cann-accuracy.json`、`dispatch-accuracy.log`、`verify_dispatch.py` / `dispatch-proof.json`：24 组同输入精度及配套 device binary 来源。
- `compare.py`、`compare-{a,b,c}.json`、`profile-{a,c}/`、`performance-acceptance.json`：失败/修复后的交替计时、原始 profiler 及逐例验收。CANN 对照只用于测试。
- `cpu.log`、`quality-staged.log`、`document-check.log`、`final-integrity.json`、`checksums.sha256`：CPU、仓库质量和证据完整性。

实机执行使用 shell `login=false`、沙箱外、物理 7 号/进程内 `npu:0`。`run_command.py` 记录精确命令及环境：保留 CANN 的 PYTHONPATH，设置线程数、设备及隔离 cache/plog；`compare*` / `dispatch*` 才加载固定 custom FIA vendor。运行前重新检查健康/占用，勿直接覆盖本轮证据文件。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_short.py \
  tests/test_flash_attn_npu_v3_mla_contract.py

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_reference.py \
  tests/test_flash_attn_npu_v3_mla_profile.py tests/test_flash_attn_npu_v3_mla_provenance.py

# 加载配套 vendor 的完整环境见 compare-c-command.json；换用未使用的标签/输出路径。
python build/mla-m04a-validation/run_command.py compare-repro \
  python build/mla-m04a-validation/compare.py \
  --output /tmp/mla-m04a-repro.json --profile-dir /tmp/mla-m04a-repro-profile \
  --iterations 50 --rounds 7
```

普通 FA3 的精确节点见 `regression-final-command.json`，复跑时换掉其 JUnit 输出路径。质量工具沿用 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache`，前者前置 PYTHONPATH，后者设为 PRE_COMMIT_HOME，执行 `ci/code_quality.sh --staged`。

## 环境与继承基线

| 项目 | 本轮结果（2026-09-24） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python`，3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| FA3 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `9bdeb03628e8bc2507243b8a8e781aff15359d132569d152d21dab10b9223923` |
| 芯片 | Ascend910B3 物理 7 号，运行前健康且空闲；故障实验进程退出后重新检查健康，不使用告警设备 |

M02b2 完整基线仍在 `build/mla-m02b2-validation/`，配套源码包在 `build/mla-m02b-validation/opp-dense/`；M03 证据在 `build/mla-m03-validation/`。本轮未修改 reference、comparator、CANN 基线执行器/包、CATLASS 或 CANN 源仓。历史 wall 中位数仍不可跨会话直接作为门槛；M04b 必须继续同轮交替比较。当前闭环提交可由 `git log -1 -- docs/mla/status.md` 定位。
