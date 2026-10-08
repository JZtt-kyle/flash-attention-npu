# MLA 当前状态

## 当前能力与唯一下一步

- 本步 M04c3 的实现和验收完成：FP16/BF16 dense MLA 的 KV 容量为 `1..16384`，含 8193；仍限 910B、512+64 共享 KV、连续 BSND、B>0、Hkv=1、无 mask 和整容量长度。保留 S1G 多块、显式 `num_splits=1` 和 401 字 v1 metadata；真实/fake/C++/调度/metadata 入口一致拒绝 16385 及更长 KV。
- 设备计算、基本块、完整 M 分区、缓冲和同步顺序保持原样。本步补充长 KV 边界与精度检查，并让组件 profiler 支持双 dtype、完整固定矩阵和同调度非 FD 源对照。
- 按用户明确的“性能只测已迁移部分”，同路径对照纳入验收，原生自动 FD 与当前非 FD 的差距只作诊断。撤回前置 FD 的提议；`num_splits=0/>1` 继续拒绝，M10 保留原顺序。
- 分支 `feat/mla-cann-migration`；提交前基点 `f85f2be`，当前闭环提交由 `git log -1 -- docs/mla/status.md` 定位。本步质量检查通过；本次提交后停止，唯一下一步为 [M04c4：dense 全局空容量边界](plan.md)。

## 环境与构建

- 证据位于 `build/mla-m04c3-matched/`，下称新目录；原功能/完整原生性能证据 `build/mla-m04c3-longkv/` 保持封存。新目录 `evidence-audit.json` 核对原 4,956 份文件、当前源码/构建、测试、原始样本及基线来源；旧 `validation.json` 的 `performance_failed_no_commit` 是原跨路径判定，原始失败记录未改写。
- Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`，torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15.0.5；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。固定 CANN 源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CANN 源仓现有状态与 CATLASS 均未修改。
- 功能验证用物理 2 号 Ascend910B3，补充性能用物理 1 号。测量前设备空闲，同进程双方绑定经过空闲预检的本地 CPU；设备快照、profiler 物理设备号、亲和性与原始 CSV 均保留。最终健康 OK、ECC 全零，1/2 号卡无残留进程，见新目录 `final-device.json`。
- 最终实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `cc22a6823c701d65ae89aee6972b1ba6e5a428d4c813128b86b8091623f402fb`。`build-final.log`、`build-provenance.json` 及源码/对象快照保存对应关系。
- 原完整功能与公开 E2E 测量使用扩展 `56f48d827bb7577b650d2bd1e15c8982f0a48edeed1b1f8cd0a222ca6b34b265`。最终构建前只改两处调度范围注释：16 个对象逐字节不变，AICPU 对象唯一差异是生成的 launch wrapper 符号后缀；链接顺序也变化，最终扩展不能声称逐字节相同。最终产物另完成下述最小精度与普通 FA3 回归。

## 功能验证

- **2,217 项去重检查通过，无失败、跳过**：原封存 2,200 项，加本次 profiler/来源回归 17 项。原 CPU/sanitizer 208 项包含 78,345 组固定源调度差分；AICPU 68 项、长 KV 66 项、dispatcher 428 项、契约 582 项、签名/兼容 129 项、普通 FA3 20 项，以及原 KV/S1G/公开入口/边界和设备拒绝回归。各套件有交叉覆盖，数量不直接相加。
- 长 KV 每种 dtype 30 组数值用例：源 Kernel O/LSE 逐元素一致，并通过独立 FP64/低精度 reference；三条公开路径各重复 3 次。覆盖 1025..16384 的块边界、offset、scale、内容/RoPE 扰动、连续最大值变化、任务环、双缓冲、compile/fake 和多流。
- 设备拒绝 42 项通过，真实 Trap 均在 15 秒内，KV 错误用例使用 Sk=16384。
- 最终重建后，FP16/BF16 各抽查 Sk=8193/16384，源对照、双数学 reference 和三条公开路径各重复 3 次均通过；普通 FA3 的 20 项 metadata/编译图/前向回归再次通过。见新目录 `smoke-public*`、`smoke/`、`ordinary-regression*`；重复检查未重复计入去重数。

## 已迁移路径的性能

- 固定矩阵保留原 18 组加长 KV 8 组，两个 dtype 均全部测量。新目录 `profile-{fp16,bf16}/` 对照迁移非 FD Kernel 与固定源非 FD Kernel，双方使用同一输入、基本块、核数和端点；O/LSE 连续 3 次逐元素一致。每路径预热 5 次、5 轮各 10 个 Kernel 样本和 20 个 probe wall 样本，总计 5,200/10,400 个样本。独立 probe 调用时间不算公开 E2E。

| 迁移 / 固定源 Kernel 耗时比 | FP16 | BF16 |
| --- | --- | --- |
| 全部 26 组 | 0.924–1.070 | 0.933–1.003 |
| 本步新增 8 组长 KV | 0.972–1.000 | 0.974–1.001 |

- 新增长 KV 的最大变慢约 0.13%。原短 KV `(1,7,16,65)`、`(1,129,128,33)` 的 FP16 Kernel 分别慢约 4.4%/7.0%，保留为组件诊断；不能表述为所有 Kernel 均不劣化。公开 E2E 门槛的结果如下。
- `comparison-scope.json` 根据固定源自动调度的 FD 字段判定路径，未按耗时筛选：每种 dtype 有 13 组原生 CANN 同样无 FD，另 13 组原生自动 FD。26 组全部保留同路径组件对照；公开 E2E 只对前 13 组应用门槛。
- 公开 E2E 使用原目录 `resume-20261008/wall-recheck.json`：同一健康空闲设备、双方相同本地 CPU 亲和性、1 个 CPU 线程、预热 5 次、5 轮各 20 次交替采样，双 dtype 全 26 组共 36,400 个原始样本。完整精度/profiler 证据保留于原 `performance-{fp16,bf16}/`；新验收复核从原始 wall 样本重新计算中位数。

| 同路径公开 FA3 / 原生 CANN E2E 比值范围（各 13 组） | FP16 | BF16 |
| --- | --- | --- |
| 复用 metadata 前向 | 0.593–0.820 | 0.604–0.798 |
| 生成 metadata + 1 次前向 | 0.827–0.895 | 0.809–0.898 |
| 生成 metadata + 4 次前向 | 0.523–0.942 | 0.508–0.930 |

- 上述公开调用逐例均满足 `<=1.0`，门槛未放宽。自动 FD 跨路径差距仍可在原数据查看，不作为本步非 FD 迁移的阻塞项；其验收留给 M10。

## 质量与复现

- 失败尝试均保留：原目录包含早期环境/脚本错误及跨路径测量；新目录首次 formatter 检查因改写格式退出非零，复跑通过；首次构建溯源错误地要求最终二进制完全相同，后续通过对象逐字节差分和源码快照核清生成符号及链接顺序变化。未将这些失败记为通过。
- 提交只包含 `step-files.json` 中的本步 26 个文件；差异检查、六项适用仓库质量钩子及 `git diff --check` 均通过；暂存后通过 `ci/code_quality.sh --staged` 再核对实际提交范围。最终验收索引为新目录 `acceptance.json`，证据完整性见 `checksums.sha256` 与 `checksums-verification.json`。大日志、源码/产物快照、CSV 和失败数据均保留在忽略目录，不进入提交。
- 每次执行的精确命令、环境和退出码见同名 `*-command.json` / `*-exit.json`。组件 profiler 必须用新的空目录并分别运行两种 dtype；正式测量应先按 `run_matched.py` 选择空闲本地 CPU、固定双方亲和性并确认设备空闲。

```bash
source /usr/local/Ascend/cann-9.0.0-beta.2/set_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_SCHEDULE_SANITIZE=1 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_schedule.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_longkv.py
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_DENSE_SOURCE_ONLY=1 MLA_DENSE_DTYPE=fp16 MLA_DENSE_ARTIFACT_DIR=/tmp/mla-m04c3-matched-fp16-recheck MLA_FORWARD_S1G_MULTIBLOCK=1 MLA_FORWARD_KV_MULTIBLOCK=1 MLA_FORWARD_LONG_KV=1 python3 -m tests.mla_dense_profile
# BF16 同上，同时改为 MLA_DENSE_DTYPE=bf16 和新的输出目录。
git diff --check
bash ci/code_quality.sh --staged
```
