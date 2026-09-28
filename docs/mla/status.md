# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `783e73f`，接手时工作区干净。M04b1 独立调度实现、CPU 固定源对照和 sanitizer 验收已完成。
- 新增 `mla/mla_schedule.h/.cpp` 和私有 `detail/cann_split_core.h/.cpp`，位于 FA3 的 910 目录。只依赖 C++17 标准库，尚未接入生产构建或 AICPU；保留固定 CANN 来源、版权和 `LICENSE.cann`。
- 生产 MLA 前向与 MLA metadata 仍明确报未实现；普通 FA3 源码、已链接对象和实际加载扩展均与 M03 一致。
- 独立调度范围为连续 BSND、512+64 共享 KV、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask、显式 `num_splits=1`；FP16/BF16 共用调度。它不校验 tensor 地址或实际布局，不能作为生产前向入口。
- 唯一下一步：M04b2，按 design 6 实现同范围 FP16 CATLASS Kernel / Block / Tile 和 FA3 接入，完成 910B 精度、同轮 CANN 性能诊断及普通 FA3 回归。本步提交后停止。

## 本步验证

证据目录：`build/mla-m04b1-schedule/`。

| 项目 | 结果 |
| --- | --- |
| 接手及上步复核 | PASS：重跑 M04a 的 `check_docs.py`，98 个源码锚点及既有文档检查通过；复核上步质量记录；源码仓状态、M03 生产源码/对象/扩展哈希一致 |
| 独立 C++ 构建与 CPU 对照 | PASS：85 项，无 skip；49,251 组输入与固定源基本块、完整端点数组、核数、cost、FD 字段、workspace、启动维度及 BATCH_MODE 完全一致 |
| 边界与复用 | PASS：1..26 个 Cube 核、C1V1/C1V2、batch 在核数及其倍数两侧、M 的 16/128/256 边界、KV 的 16/32/128/256/512 边界、多 query、反序重复调用及相同 cost 的取舍 |
| 字段与资源 | PASS：尺寸窄化、行乘积、源 uint32 LCM/总块数、端点容量、uint64 workspace 加法；L1/UB/L0 刚好足够与少 1 字节；失败不改写原调度，GM 四段边界不超预算 |
| ASan / UBSan / LSan | PASS：沙箱外完整重跑同组 85 项，无 skip，49,251 组结果哈希与普通构建一致；首次沙箱内 LSan 因进程检查受限退出失败，其日志单独保留 |
| 来源与序列化 | PASS：私有 SplitCore 去掉日志、命名空间、include guard 和初始化次序差异后算法 token 与固定源一致；C1V1 FD 端点按源 vector 实际长度复制，见 design 6.2 |
| 文档及提交质量 | PASS：16 处本地链接/锚点、唯一下一步和 metadata E2E 口径一致；差异及 6 项适用质量检查通过；只暂存本步 10 个文件 |
| 生产扩展构建、NPU 精度/性能、普通 FA3 回归 | 未运行：新调度尚未被生产构建引用，公共入口、metadata、kernel 及现有构建未修改；本步没有新的设备精度或性能结论 |

oracle 从固定 commit 的 Git blob 导出原 `split_core.h/.cpp`，直接编译；另外抽取原 MLA host 的 10 个计算函数，仅用最小数据和平台 API shim 去除 gert。oracle 不包含或链接待测调度。比较原 SplitCore vector，不执行源 C1V1 可能越界的 `SetSplitOutput` 序列化循环。平台结果表由调用方提供，首步仅在 CPU 检查；M04b2 仍须核对真实发射与设备行为。

metadata 性能门槛仍为逐例 `FA3 metadata 路径 E2E / 原 CANN E2E <= 1.0`；M04b 无 metadata 数据仅作诊断。

## 环境与最近实机证据

| 项目 | 本轮复核 |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1`；CPU C++ 使用 g++ 11.4.0 |
| CANN / 芯片 | `/usr/local/Ascend/cann-9.0.0-beta.2`；只读查询物理 2 号 Ascend910B3，Warning 且有共享进程；最终探测进程 `npu_initialized=false`，未运行算子 |
| 固定来源 | FIA `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，子模块干净；CANN 源仓已有删除/未跟踪文件保持原样 |
| 平台 API | 910B3 配置为 20 AIC / 40 AIV，L1 524032 B、UB 196352 B、L0A/B/C 为 65536/65536/131072 B，libapi 16 MiB；`CalcTschBlockDim` 已逐核查询；另查 910_9391 配置，不代表 910C 实机验证 |
| 实际加载扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `955ecbe4c2b199b2e1131d9c86c38ce23b15ed8704c55cf2e35de37986b84877`；直接加载并确认 `npu_initialized=false`，仍是 M03 构建产物 |

环境探测最初遇到沙箱驱动限制；一次包入口探测因覆盖 PYTHONPATH 缺失 tbe 失败。保留原 CANN PYTHONPATH、直接加载扩展并在沙箱外只读查询后复核通过。上述卡状态仅为环境快照，不作为性能验收条件。

最近 NPU 闭环仍为 M03（`677c14e`），`build/mla-interface-guard-validation/` 保存 233 项 MLA 契约和 11 项普通 FA3 的通过证据；本轮没有重跑。进入 M04b2 需重新选择设备、强制核查新增编译单元/header 依赖及实际加载产物，再完成 910B FP16 精度、重复执行、同轮固定源码 CANN 性能诊断和普通 FA3 回归。

## 证据与复现

- `previous-step-docs-check.log`、`source-review.log`、`source-repositories.json`：上步复核、源算法 token 核对及只读仓库状态。
- `environment.json`、`environment.log`、`npu-smi.log`、`platform-910b3.log`、`platform-910c.log`、`compiler-version.txt`：实际扩展加载、平台和芯片快照。
- `cpu-final/`、`sanitizers-host/`：原始 Git blob、抽取函数、来源哈希、精确编译命令、独立可执行文件及各组用例/输出摘要哈希；对应 `*-command.json`、`*-exit.json`、`*.log`、`*.xml` 保存运行结果。`sanitizers.log` 记录沙箱内 LSan 失败。
- 最终 `binary-provenance.json`、`acceptance.json`、`quality-final.log`、`checksums.sha256` 保存二进制隔离、生产产物未变化及提交前完整性检查结果。

调度最小闭环（从仓库根目录；第二条需要允许 LSan 进程检查的环境）：

```bash
MLA_CANN_SOURCE_ROOT=/home/zhangqijia/repo/ops-transformer MLA_SCHEDULE_ARTIFACT_DIR=build/mla-m04b1-schedule/cpu-final python3 -m pytest -q tests/test_flash_attn_npu_v3_mla_schedule.py --junitxml=build/mla-m04b1-schedule/cpu-final.xml
MLA_CANN_SOURCE_ROOT=/home/zhangqijia/repo/ops-transformer MLA_SCHEDULE_SANITIZE=1 MLA_SCHEDULE_ARTIFACT_DIR=build/mla-m04b1-schedule/sanitizers-host python3 -m pytest -q tests/test_flash_attn_npu_v3_mla_schedule.py --junitxml=build/mla-m04b1-schedule/sanitizers-host.xml
```

最终暂存检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m04b1-schedule/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
