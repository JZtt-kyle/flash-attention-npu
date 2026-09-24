# MLA 当前交接

## 状态与下一步

- 当前步骤：M03 已完成，MLA 入口识别、参数校验和未支持拦截已闭环。
- 当前分支：`feat/mla-cann-migration`；接手基点 `2bd9762`。用户已确认方案，本次一个提交后停止。
- 本步产物：[910 MLA 校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)、[生产及 fake 入口](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[契约检查](../../tests/test_flash_attn_npu_v3_mla_contract.py)。
- 唯一下一步：M04，实现 512+64、BSND、无 mask、无 split 的 CATLASS 前向；同时验收 FP16/BF16 的 O/LSE、单/多 query、代表性性能及普通 FA3 回归。

**生产 MLA kernel 尚未接通。** M03 只关闭静默忽略和错误分派入口，不能当作 MLA 数值或性能通过；生产路径没有 CANN 回退。

## 当前入口行为

- 按 `qv` 和 Q/K/V 维度识别压缩 64/512 与 prefill 192/128，不按 query 长度分派。无效 rank、Q/K 维度、qv/KV shape、dtype、device、头数、末维 stride 和 PA+TND KV 冲突有明确错误。
- 合法外层非连续 stride 和非零 storage offset 可通过形状校验，然后得到相应 `MLA compressed/prefill forward is not implemented yet`；拒绝过程不读取张量数值、不生成 metadata、不复制完整输入、不分配占位输出。
- dense/varlen 在普通 metadata 生成前拦截，包装器传递实际 `headdim_v`；kvcache 在 metadata 验证/前向调用前拦截。注册 fake 与真实自定义算子执行相同校验；C++ 直调独立拦截，无法绕过 Python。
- 异维 metadata 在公开、注册 fake、自定义算子和 C++ 入口均明确拒绝；metadata 没有 `qv` 参数，不依据维度推定它存在。普通等维 byte metadata 的协议保持兼容。
- 空输入、梯度请求、mask、split、softcap、调度选项、既有 metadata 和追加 KV 都不能绕过未实现拦截，输入不改写。未来接通单一路径时仍须逐项维持尚未实现组合的拒绝。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手核对 | PASS：M02b2 全部证据 checksum、配套 vendor/header/device binary、原 FA3 扩展哈希一致；固定 CANN 源码范围和 CATLASS 未变 |
| 构建溯源 | PASS：仅构建 v3/910；`flash_api.o` 确实重编，depfile 包含新增 MLA header；八个 FA3 kernel 对象和 AICPU 对象保持原哈希，重新链接并复制扩展；构建 `127.83s` |
| 910B MLA 契约 | PASS：`233 passed in 14.80s`，无 skip；Python 真/假自定义算子、C++ 直调、三种公开入口、两种 dtype、dense/TND/PA、外层 stride、非法参数、空输入及未接通选项 |
| 普通 FA3 | PASS：`11 passed in 28.60s`，无 skip；继承 M02b1 的 8 项 dense/varlen/kvcache/FlashDecode 精度和 metadata 用例，加 3 项 metadata/dense/varlen 编译图用例，dense/varlen 含 backward |
| 既有 CPU | PASS：reference/provenance/profile 共 `88 passed in 21.06s`；不执行 NPU kernel |
| 数值与性能边界 | 本步不修改 kernel/AICPU，不运行新 MLA 精度或性能；普通 FA3 仍用原 reference/comparator，阈值未改 |
| 仓库质量 | 代码及测试适用钩子通过；提交前 staged 检查、文档链接/围栏和空白检查见本步 `quality-final.log` / `document-check.log` |

`flash_api.cpp` 附带仓库 clang-format 钩子要求的既有格式规范化；`flash-api-semantic.diff` 以同工具格式化后的旧文件为基准，确认实际逻辑仅新增 header 和前向/metadata 两处入口调用。未修改 CATLASS、kernel、AICPU 或 CANN 源仓。

## 证据与复现

持久证据根目录：

`/home/zhangqijia/repo/flash-attention-npu/build/mla-m03-validation/`

- `takeover-audit.json`、`previous-checksums-retry.log`、`environment-before.json`：接手版本、源码和旧证据核对。
- `build-command.json` / `build.log` / `build-exit.json`、`objects-before.json`、`build-provenance.json`、`loaded-extension.json`：编译依赖、对象及实际加载扩展溯源。
- `source-snapshot.json`、`contract-command.json` / `contract.log` / `contract.xml` / `contract-exit.json`：实际测试源码和 233 项契约结果。
- `regression-command.json` / `regression.log` / `regression.xml` / `regression-exit.json`：11 项普通 FA3 的精确节点、环境及结果。
- `cpu.log`、`quality-code.log`、`quality-final.log`、`document-check.log`、`final-integrity.json`、`checksums.sha256`：CPU、质量及最终证据核对。

实机命令使用 shell `login=false`，沙箱外执行。每次重新检查设备健康/占用；本轮为物理 7 号、进程内 `npu:0`。CANN set_env 的 `PYTHONPATH` 保留；本轮 FA3 契约/回归不加载 custom FIA vendor。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_reference.py \
  tests/test_flash_attn_npu_v3_mla_profile.py tests/test_flash_attn_npu_v3_mla_provenance.py

ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_contract.py

ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY_REPRO'
import json
import subprocess
from pathlib import Path
report = json.loads(Path("build/mla-m03-validation/regression-command.json").read_text())
# 不覆盖原 JUnit 证据；其余参数含本轮的全部精确测试节点。
command = [arg for arg in report["command"] if not arg.startswith("--junitxml=")]
subprocess.run(command, check=True)
PY_REPRO
```

质量工具使用已有 `/tmp/mla-m01-precommit` 和 `/tmp/mla-m01-precommit-cache`；将前者前置到 `PYTHONPATH`，后者设为 `PRE_COMMIT_HOME`，运行 `ci/code_quality.sh --staged`，不绕过钩子。系统 Python 未安装 pre-commit 的两次尝试保留在 `quality-format*.log`。首次 checksum 使用了错误工作目录，改在 M02b2 目录运行后全部匹配；相关失败日志保留。

## 环境与继承基线

| 项目 | 本轮结果（2026-09-24） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python`，3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| FA3 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `eedc31ad27486cb148b01c9b0aababb36aa5a5685dbfa35a57169451201191b4` |
| 芯片 | Ascend910B3 物理 7 号，运行前健康且空闲；不使用告警设备 |

M02b2 的完整证据继续位于 `build/mla-m02b2-validation/`，配套源码包为 `build/mla-m02b-validation/opp-dense/`。36 组 BSND O/LSE 精度/重复执行、144 次分派来源与四次 profiler 采集已闭环；主机同步调用噪声仍未消除，不能用历史 wall 中位数作为跨会话性能门槛。后续 kernel 性能须在同轮匹配环境中与固定源码 CANN 交替测量，逐例比值 `新实现 / CANN <= 1.0` 的约定不变。

本步未修改执行器、reference、comparator、基线包或原始性能数据。并行 ATK 文档/数据更新及既有未跟踪文件保留，未纳入 MLA 实现改动。当前闭环提交可由 `git log -1 -- docs/mla/status.md` 定位。
