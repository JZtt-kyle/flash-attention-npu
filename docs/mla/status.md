# MLA 当前状态

## 当前能力与唯一下一步

- 本次完成 M06b1：新增独立、无 gert/torch 依赖的 `BuildBsndKvLengthSchedule`，解析执行处理器可读的 `int32[B]` 实际 KV 长度，完整校验后逐值转换为源 int64 向量，复用固定源 `SplitCore`。支持非均匀、等长短序列和全零实际长度；缺省或整容量长度回到既有 dense 调度。独立入口未列入生产编译，也未接公共前向或 metadata。
- 当前生产能力仍为 M04c4：FP16/BF16、512+64 共享 KV、连续 BSND、Hkv=1、无 mask、显式 `num_splits=1`，非空 KV 容量最多 16384；支持 `B=0`、`Sq=0`、`Sk=0` 及其组合。全局空 KV 返回 `O=0、LSE=+inf`；空 batch/query 保留空 shape。401 字 v1 metadata 和公开路径仍要求所有样本使用完整 KV 容量。
- 分支 `feat/mla-cann-migration`，本步基点 `b2e9355`，接手时工作区干净。唯一下一步为 **M06b2：BSND 实际 KV 长度的 AICPU metadata 生成和设备校验消费者**；公开短长度继续拒绝。已确认范围和完成条件见 [plan](plan.md#m06非分页长度tnd-与-stride)，后续在此范围内推进，无需重复确认。

## 本次 CPU 闭环与证据

- 证据目录：`build/mla-m06b1-lengths/`；最终索引为 `acceptance.json`，复核脚本为 `audit.py`，命令、环境和退出码保存在 `*-command.json` / `*-exit.json`。上步证据哈希、18 个既有生产/测试文件哈希、源仓状态与扩展哈希核查一致，见 `handoff-audit.json`。
- **312 项去重 pytest 检查通过，无失败/跳过**：上步 CPU 最小闭环 220 项全部重跑；最终新增 92 项全部通过，并对共享测试驱动重跑 5 项既有 dense 检查（重复项不累加）。最终证据为 `baseline-cpu.xml` 和 `lengths-verified.xml`。首批 37 项与修复定向 6 项单独留存，不额外计数。
- 新增 **33,736 组实际长度源差分**：25,156 组与原生非 FD 输出完全一致，8,580 组独立检查 FA3 `num_splits=1` 的完整 M 端点映射；后者不称作原生源输出。比较基本块、端点、核数、FD 字段及 workspace，独立枚举所有正长度计算任务并核对每核成本。覆盖 AIC=1..26、C1V1/C1V2、核数两侧 batch、M=128..256 及尾块、KV 实际长度 0/1/511/512/513/8193/16384，包含 18 组计算端点之外的零 KV 尾部。
- 1441 组整容量/缺省长度及全局空容量与原 dense 结果逐字段一致。负值、超容量、长度计数、字段/任务量/地址及 workspace 溢出、局部内存不足、blockDim 约束均检查；拒绝时输出字节保持不变。迁移端和独立固定源 oracle 均启用 ASan/UBSan；源函数逐字节提取并保留清单，见 `lengths-verified/source-manifest.json`、输入、输出摘要与编译记录。
- 原完整矩阵曾失败：15 核混合长度下，源 `CalcSplitPlan` 在成本耗尽后停于尾部零 KV 样本之前，新入口误用 dense 的末端等于 B 校验。已按源行为保留端点，并验证省略的样本全部为零 KV，完整矩阵重跑通过。原失败保存在 `lengths-final.log`、`lengths-final-exit.json`、`failure-cases.json`；修复回归为 `zero-fix.xml`。固定源独立初始化 O/LSE 的依据见 `source-zero-initialization.*`；M06b2/b3 须保留这一依赖，不能仅凭计算分区覆盖空行。
- 生产 `BuildDenseSchedule`、公共入口、401 字 v1 metadata、Kernel 和构建列表未修改；本步未构建生产扩展，未运行 NPU 精度/E2E 或普通 FA3 设备回归，不将这些项目记为本次通过。最近一次设备闭环仍为 M04c4 的 `85ce13d`，其 2177 项检查、同路径 E2E 和原始失败见 `build/mla-m04c4-empty/acceptance.json`。
- 初次 clang-format 修改文件后，代码质量复核和最终暂存检查通过：六项适用钩子通过，其余三项无对应文件跳过，见 `quality-staged-*`。差异与 `git diff --check` / 暂存检查通过；只暂存本步六个文件，提交一次后停止。

## 环境与后续复现

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0。主机探测到 8 张 Ascend910B3，健康均为 OK；这只描述探测时状态，不代表后续设备空闲。初次沙箱内 `npu-smi` 初始化失败已保留，后续主机探测成功，见 `environment-sandbox-failure.json` 和 `environment.json`；sanitizer 均在正常主机环境运行。
- 实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 为 `ef4eb188fc2a193a4f4ecb96e07e350756d605f4796d32b7e3324de1f6be0b7a`，与 M04c4 的构建来源及加载记录一致；本次独立入口不在该扩展中。
- CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f`，工作树干净；只读 CANN 源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，接手时已有的 67 条删除/未跟踪记录保持原样。本步四个代码/测试文件最终哈希见 `final-code-hashes.json`，修复前版本哈希单独保留。
- 完整 CPU 复现命令如下；精确分批运行命令见上述证据目录。M06b2 修改 metadata 协议及设备消费时须另行完成 AICPU/设备和普通 FA3 回归，并按 plan 更新 design 的具体记录契约。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_SCHEDULE_SANITIZE=1 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_schedule.py tests/test_flash_attn_npu_v3_mla_schedule.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
git diff --check
PYTHONPATH="/tmp/mla-m01-precommit:${PYTHONPATH:-}" PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
```
