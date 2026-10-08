# MLA 当前状态

## 当前能力与唯一下一步

- 本次完成 M06a（纯文档）：核对固定源的长度/调度依赖，将 M06 拆为 BSND 实际 KV 长度、紧凑 TND、`seqused` 和外层 stride 的独立验收步骤，见 [plan](plan.md#m06非分页长度tnd-与-stride)。已确认的范围和接口契约不变，未修改 design、生产代码或测试。
- 当前生产能力仍为 M04c4：FP16/BF16、512+64 共享 KV、连续 BSND、Hkv=1、无 mask、显式 `num_splits=1`，非空 KV 容量最多 16384；支持 `B=0`、`Sq=0`、`Sk=0` 及其组合。全局空 KV 返回 `O=0、LSE=+inf`；空 batch/query 保留空 shape。401 字 v1 metadata 生成/消费仍要求所有样本使用完整容量；样本短长度、TND、非连续外层 stride 等尚未接通。
- 分支 `feat/mla-cann-migration`，本步基点 `85ce13d`，接手时工作区干净。唯一下一步为 **M06b1：独立的 BSND 实际 KV 长度解析与 CPU 源调度对照**；只新增无 gert/torch 依赖的独立入口，不接生产 metadata 或前向。范围、矩阵、固定源对照及 sanitizer 条件见 plan 的 M06b1。方案已确认，后续在此范围内推进，无需重复确认。

## 本次文档闭环

- 证据目录：`build/mla-m06a-plan/`。`handoff-audit.json` 核对上步验收索引、8 条最终命令及成功退出码、18 个生产/测试文件哈希和扩展文件哈希；与 M04c4 记录一致。上步证据清单哈希与 `checksums-verification.json` 一致，本次未重新哈希全部 1210 个大日志/构建文件，也未重跑 NPU 测试。
- 固定 CANN 源的三个文件及函数锚点已核查：host MLA tiling 的 `DealSameSeqEachBatch`、`CalcMBaseSize`、`CreateSplitInput`，公共 `SplitCore` 的实际长度读取，以及设备 Kernel 的长度解析和零任务分支。工作树中的这些文件与固定 commit 逐字节一致；已有 FA3 源快照用于核对公开签名，未增加接口。
- 本步 25 个本地链接/片段锚点、14 个 M06 子步骤顺序、唯一下一步及两文件范围检查通过，见 `doc-checks.json`；复现脚本为 `check_docs.py`。`git diff --check`、暂存差异检查及 `ci/code_quality.sh --staged` 通过，三个适用文档钩子通过，其余六项因无对应文件跳过；日志和退出码见 `diff-check*`、`staged-diff-check*`、`quality-staged*`。只暂存 plan/status，提交成功后停止。

## 最近一次代码闭环与证据

- M04c4 代码提交为 `85ce13d`，完整验收保存在 `build/mla-m04c4-empty/acceptance.json`，命令、环境和退出码见同目录的 `*-command.json` / `*-exit.json`，构建来源见 `build-provenance.json`。本节为既有结果，不是本次重新运行。
- 2177 项去重 pytest 检查通过，无失败/跳过：CPU/ASan/UBSan 220 项（含 78,345 组固定源调度差分）、空容量功能 180 项、设备拒绝 42 项、入口/AICPU/metadata/签名 1715 项、普通 FA3 20 项。42 个独立进程均触发真实 Trap；最终 FP16/BF16、KV=8193/16384 源 Kernel O/LSE 逐元素一致，双 reference 和三条公开路径重复检查通过。
- 双 dtype 各 13 组源原生非 FD 用例的 metadata 复用、生成加一次/四次前向共 78 个 E2E 中位数比值均 `<=1.0`。一次生成加一次前向的噪声用例经补测闭环，见 `performance-verdict.json`、`noise-recheck.json`。生产 Kernel 比值为 FP16 0.976–1.068、BF16 0.972–1.043，不能声称所有 Kernel 均不劣化；空容量只有调用延迟诊断，没有同义 CANN E2E 基线。
- 原始失败和修正仍保留在上步 `acceptance.json`：NPU/LeakSanitizer 沙箱限制、质量工具路径和 formatter 修正、CPU 背景负载、性能统计窗口及补测准备问题均未改写为通过。最终适用检查已经通过，没有待处理的上步失败。

## 环境与后续复现

- 本次为文档步骤，未构建、导入扩展或探测当前 NPU 健康/占用。上步运行环境为 Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5，实机为 Ascend910B3。后续开发/实机验证前仍需重新核查环境及实际加载路径。
- 当前磁盘扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 的 SHA256 为 `ef4eb188fc2a193a4f4ecb96e07e350756d605f4796d32b7e3324de1f6be0b7a`，与上步实际加载的构建产物一致；这不替代后续进程的加载核查。
- CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f`，工作树干净；只读 CANN 源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，其接手时已有 67 条删除/未跟踪记录保存在 `cann-status-before.txt`，本步未修改。上步 2/3/4 号卡健康记录及性能测量的真实 CPU 背景负载只描述当时条件，不用于推断当前设备空闲。
- 文档检查复现如下。M06b1 开始后复核上步 CPU 最小闭环，并新增实际长度源差分；ASan/UBSan 使用可正常运行 sanitizer 的主机环境。NPU 代码接入时再重跑受影响的 910B 精度、性能和普通 FA3 回归，精确历史命令可从上述命令记录取得。

```bash
python3 build/mla-m06a-plan/check_docs.py
git diff --check
PYTHONPATH="/tmp/mla-m01-precommit:${PYTHONPATH:-}" PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
# 上步 CPU 最小闭环；新的实际长度检查随 M06b1 增补。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_SCHEDULE_SANITIZE=1 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_schedule.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
```
