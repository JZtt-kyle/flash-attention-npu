# MLA 当前状态

## 当前能力与唯一下一步

- M04c4 的实现与验收已完成。FP16/BF16、512+64 共享 KV、连续 BSND、Hkv=1、无 mask、显式 `num_splits=1` 的 dense 路径新增 `B=0`、`Sq=0`、`Sk=0` 及其组合；非空 KV 上界仍为 16384。
- 全局空 KV 返回 `O=0、LSE=+inf`；空 batch/query 保留 O/LSE 空 shape。401 字 v1 metadata 保留形状、dtype 和平台，空容量只有校验核，没有 attention 切块、端点和 workspace；生成和消费仍校验完整容量长度及记录，不进入 attention Kernel。非零容量内的样本零长度仍明确拒绝，留给 M06。
- 分支 `feat/mla-cann-migration`，本步基点 `0ed7fe7`，接手时工作区干净。当前闭环提交由 `git log -1 -- docs/mla/status.md` 定位；提交后停止。唯一下一步为 M06，实施前拆分 TND／实际长度／stride。

## 环境与构建

- 本步证据目录为 `build/mla-m04c4-empty/`；上一步 `build/mla-m04c3-matched/`、`build/mla-m04c3-longkv/` 保持封存。命令、环境、退出码见本步各 `*-command.json`、`*-exit.json`，大日志、构建快照及原始样本留在忽略目录。
- Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`，torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15.0.5。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，两者状态保持接手时原样。
- 功能验证使用健康空闲的物理 2/3/4 号 Ascend910B3；各命令记录实际可见设备。接手时加载扩展 SHA256 与上一步一致，`baseline-smoke` 重跑双 dtype 的 KV=8193/16384 源对照、双 reference 和三条公开路径通过。
- `build.log` 核实设备头与 AICPU 增量依赖触发重编译。17 个链接对象中 11 个与上一步逐字节一致，变化为 schedule、FP16/BF16 launcher、forward、flash_api 和 AICPU metadata。源码与对象快照及来源校验见 `build-provenance.json`。
- 最终实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `ef4eb188fc2a193a4f4ecb96e07e350756d605f4796d32b7e3324de1f6be0b7a`。

## 功能验证

- **2177 项去重 pytest 检查通过，无失败、跳过**：CPU/ASan/UBSan 220 项（含 78,345 组固定源调度差分）、空容量功能 180 项、空容量设备拒绝 42 项、入口／AICPU／metadata／签名回归 1715 项、普通 FA3 回归 20 项。
- 空容量覆盖双 dtype、全部 batch/query/KV 零组合、S1G 多块、非空 KV=16384、空指针和偏移视图、NaN 输入、O/LSE 及 accumulator shape/dtype、公开及直接 C++ 入口、有/无及偏移克隆 metadata、只读复用、无 enqueue-path D2H、fake、compile 与跨 stream allocator 生命周期。CPU 消费检查逐字破坏空记录的所有未用 payload，并验证非空记录不能改形状冒充空调度。
- 42 个独立进程验证损坏 metadata 和负值／超容量长度；生成、隐式生成与复用均触发真实 Trap，最大触发时间约 0.307 秒，未超过 15 秒。原始报告见 `runtime-errors/`。
- 最终构建另在物理 4 号重跑 FP16/BF16、KV=8193/16384：源 Kernel O/LSE 逐元素一致，独立 FP64/低精度 reference 和三条公开路径各重复 3 次均通过，见 `final-smoke.log`、`final-smoke/`。普通 FA3 包含 metadata、softcap、BSND/TND/PA/FD 和编译图回归。

## 已迁移路径的性能

- 固定矩阵见 `performance-cases.json`：沿用上一步固定源判定的全部 13 组原生非 FD 用例，双 dtype 各测公开 metadata 复用、生成加一次／四次前向及 Kernel；不以耗时筛选形状。仍要求逐例 E2E 比值 `<=1.0`。
- 初次空闲 CPU 预检未通过，未开始采样。后续双方固定相同本地 CPU 亲和性、1 个计时线程，在同一空闲 2 号 910B 上交替采样；实际 CPU 有背景负载，不能称 CPU 空闲。`affinity.json`、`cpu-preflight-note.json`、`performance-monitor.jsonl` 和 `performance-audit.json` 保留实际条件及逐轮统计。
- 双 dtype 各 13 组的 O/LSE 与固定源 CANN 逐元素一致，并通过独立 FP64/低精度 reference。每条公开路径预热 5 次、5 轮各 20 次，完整矩阵共 18,200 个 wall 样本；Kernel 每路径 5 轮各 10 次，共 2,600 个样本。原始 CSV、kernel 对应关系及加载的固定源库见 `public-{fp16,bf16}/`。

| 公开 FA3 / 固定源 CANN E2E 中位数比值（各 13 组） | FP16 | BF16 |
| --- | --- | --- |
| 复用 metadata 前向 | 0.602–0.775 | 0.627–0.784 |
| 生成 metadata + 1 次前向 | 0.853–0.945 | 0.868–0.934 |
| 生成 metadata + 4 次前向 | 0.498–0.929 | 0.543–0.952 |

- 全部 78 个中位数比值满足 `<=1.0`。保留单轮波动，不声称每轮均更快。按轮与配对样本重采样的 95% 区间中，FP16 `(2,1,8,257), scale=0.23` 的生成加一次前向上界为 1.0098；对这一例另做 20 次预热、20 轮各 50 次交替采样（2,000 个原始样本），精度再次通过，比值 0.9063、95% 区间 0.8402–0.9881，消除不确定性。原矩阵未覆盖或筛除，补测见 `noise-recheck.json`，最终判定见 `performance-verdict.json`。
- 生产 Kernel（含 metadata 消费）的耗时比值为 FP16 0.976–1.068、BF16 0.972–1.043，保留为组件诊断；不能表述为所有 Kernel 均不劣化。公开 E2E 门槛未放宽，自动 FD 仍未接通。
- 空容量另保存 3,200 个 FA3 调用延迟样本，见 `empty-latency.json`。固定 dense CANN 基线的 `CannCall` 明确排除空序列，因此没有同义 CANN E2E 对照；这些数据仅为当前 CPU 负载条件下的延迟诊断。

## 质量与复现

- 已保留失败及修正证据：沙箱无法访问 NPU；ptrace 导致 LeakSanitizer 失败，主机同组 ASan/UBSan 通过；默认 Python 路径无 pre_commit，后使用既有 `/tmp/mla-m01-precommit` 与固定缓存；首次 formatter 改写文件，复跑通过后才构建；初次 CPU 空闲预检失败；统计脚本先因监控窗口过短失败，修正为包围计时段的监控区间；补测准备中的自身 NPU 上下文误判与导入顺序错误均发生在采样前。上述失败未改写为通过。
- 最终 2/3/4 号卡健康 OK、ECC 全零且无残留进程，见 `final-device.json`。本步只提交 `step-files.json` 中的 21 个文件；生成的 `kernel_meta/` 归档到证据目录，CANN 源仓及 CATLASS 状态与接手时一致。
- 提交前差异检查、`git diff --check` 及 `ci/code_quality.sh --staged` 六项适用质量钩子通过。验收索引为 `acceptance.json`，源码／对象见 `build-provenance.json`，证据完整性见 `checksums.sha256` 与 `checksums-verification.json`。
- 最小复现如下。NPU 命令须在可访问设备的主机环境运行；sanitizer 不能在 ptrace 沙箱中执行。各套件的精确参数和设备选择以证据目录中的命令记录为准。

```bash
source /usr/local/Ascend/cann-9.0.0-beta.2/set_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_SCHEDULE_SANITIZE=1 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_schedule.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_empty.py
# 完整普通 FA3 回归：ordinary-command.json；非空双 dtype 源对照：final_smoke.py。
# 性能：run_performance.py / recheck_noise.py；使用全新证据目录并保留实际 CPU 负载。
git diff --check
PYTHONPATH="/tmp/mla-m01-precommit:${PYTHONPATH:-}" PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
```
