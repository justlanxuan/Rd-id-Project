# G6 Protocol Lock

状态：`locked`

人类确认：2026-08-12（Asia/Hong_Kong）

- 统一窗口：`window_len=24`、`stride=16`。
- Custom：循环 inner validation，outer test 的下一个 session 为 validation，
  其余两个 session 为 training。
- 已授权创建实验代码 snapshot commit；protocol record 将在该干净 commit 上生成。

正式 GPU 训练开始前，本文件必须改为 `locked`，生成 protocol hash；锁定后的任何语义修改必须创建新版本，旧结果不得混入新主表。

## 1. 已锁定的两个选择

1. 所有 canonical 训练/测试窗口统一使用 `window_len=24`、`stride=16`。
2. Custom 四折 outer LOSO 使用循环 inner validation：outer test 的下一个 session 为 val，其余两个 session 为 train。

## 2. Custom 固定 folds

| Fold | Train sessions | Validation session | Test session |
|---|---|---|---|
| 1 | 20260211_172257, 20260211_172522 | 20260211_171724 | 20260211_171423 |
| 2 | 20260211_171423, 20260211_172522 | 20260211_172257 | 20260211_171724 |
| 3 | 20260211_171423, 20260211_171724 | 20260211_172522 | 20260211_172257 |
| 4 | 20260211_171724, 20260211_172257 | 20260211_171423 | 20260211_172522 |

- Fine-tune 与 direct 必须使用完全相同的 fold manifest。
- Zero-shot 不读取 Custom train/val，只在对应 outer test session 上评估 source checkpoint。
- test session 不参与归一化、early stopping、checkpoint selection 或超参数选择。

## 3. 数据语义

- IMU：左前臂/左腕 `acc_x,acc_y,acc_z,quat_w,quat_x,quat_y,quat_z`。
- Skeleton：H36M-17；任何 COCO-17 extractor 输出先做显式 mapping。
- TotalCapture：`L_LowArm` Xsens 7D；不再使用 Hybrid 错读 legacy 48D 前七维的旧语义。
- EgoHumans：LeftWrist 7D realistic IMU 与多人 pose2d。
- Custom：使用已验证的 `hybrid_w24_session_out_rawcsv7d_swapsess`
  四折 prepared cache；该版本使用 raw CSV 7D IMU，并修正
  `20260211_171724` 与 `20260211_172257` 的 IMU-person 顺序。缺失历史
  生成 provenance 时标记 `adopted_existing`。

## 4. FrameAcc 口径

- Source 主指标：window candidate assignment FrameAcc，保存 `correct/total`、group size 与 singleton rate。
- TotalCapture 单人 sequence 使用确定性跨 sequence 候选组，`K=4`；主测试 singleton rate 必须为 0。
- EgoHumans 使用同一时窗自然多人候选组；主测试 singleton rate 必须为 0。
- Custom 主指标：segment/frame-level assignment FrameAcc；zero-shot、fine-tune、direct 共用同一 evaluator、window `24/16` 和 session ground truth。评估时从 raw CSV 重建 7D IMU，并对上述两个 session 执行同样的 person-order 修正。
- Custom window-level FrameAcc 作为诊断指标；仅显式排除单候选窗口，并报告排除数量/比例，禁止把 singleton 计为正确。
- Custom 同时报告逐 session、macro-session、micro/weighted、逐 seed 和 `mean ± sample std`。

### 4.1 冻结前 data manifest 预检

manifest 位于 `manifests/data/`，身份 hash 不包含绝对路径与 mtime，由
split CSV 与所有被引用 NPZ 的 SHA256 生成。

| Dataset/fold | Manifest hash | Train/val/test rows | Test group sizes | Singleton rate | Zero IMU/skeleton |
|---|---|---:|---|---:|---:|
| TotalCapture | `6a9c66617e9e73d04be1b59a58a580cd148401b1f78b9fde40c491f6cdef3913` | 8901/1072/879 | 2:11, 3:31, 4:191 | 0 | 0/0 |
| EgoHumans | `d1a16dbe807f69a0bdae857776d054ee12d9b55ba581b13627a2f10bf1037c1f` | 1305/285/1101 | 3:267, 4:75 | 0 | 0/0 |
| Custom fold 1 | `1f8eb1d3e46429ec30d912e273e3b6dff0d0e5ec7a9c279d59b7a06f61a5d060` | 919/492/434 | 1:88, 2:173 | 0.3372 | 0/0 |
| Custom fold 2 | `fdaaf1afbcfef91bf51a068ee9da72d682189751f64cf70f60e6f6d447a8e7f2` | 913/440/492 | 2:246 | 0 | 0/0 |
| Custom fold 3 | `2daa754fef9aaff2326b37f418cd49ddd6deb16c5c3ddbdf821f38bf2f3a407b` | 926/479/440 | 2:220 | 0 | 0/0 |
| Custom fold 4 | `ce6e52c229363839e092237047456d8c8a108a3231c37e2f858bfc77442d8330` | 932/434/479 | 1:5, 2:237 | 0.0207 | 0/0 |

- TotalCapture 按 subject 分割，EgoHumans/Custom 按 session 分割；六个 manifest 的
  split identity 与 `source_sequence` 均无交叉。
- Custom singleton 仅影响 window-level 诊断，按 `exclude` 显式统计；主结果由
  独立 segment/frame evaluator 生成。
- Custom manifest 同时绑定 held-out session 的 segment NPZ、frame timestamp 和
  raw IMU CSV；正式 segment FrameAcc 的外部评估输入变化会改变 manifest hash。
- 旧 `hybrid_w24_session_out` 缓存存在全零 IMU 窗口且 skeleton 数值语义
  与最终协议不同，已排除，不得与上表结果混用。

## 5. 模型、训练与 seed

- Primary model：当前注册的 `hybrid`，输出 IMU/video 128D embedding；架构配置在正式训练前冻结。
- Seeds：`0, 42, 123`，任何失败 seed 原样重跑或显式标记失败。
- Source、fine-tune、direct 训练预算默认 50 epochs；checkpoint 依据 validation loss 选择。
- Fine-tune 的 source checkpoint seed 与 target fine-tune seed 一一对应。

## 6. Extraction 与缓存

- 正式训练允许复用已通过 schema/content/split 检查的历史 skeleton/prepared cache。
- 三数据集 `force=true` 真实 AlphaPose full smoke 已完成，不要求全量重提取；证据见 `extractor_smoke.md`。

## 7. 正式矩阵

- Source train/test：2 source × 3 seed = 6。
- Source→Custom zero-shot：2 source × 4 session × 3 seed = 24 evaluations。
- Source→Custom fine-tune：2 source × 4 fold × 3 seed = 24 train/evaluations。
- Custom direct：4 fold × 3 seed = 12 train/evaluations。
- 最低训练 42 个，评估输出 66 个。

## 8. 锁定动作

本协议收到人类确认后：

1. 将状态改为 `locked` 并记录确认时间；
2. 经人类授权创建可识别的实验代码 snapshot commit；正式调度器拒绝 dirty
   worktree，protocol hash 同时绑定该 commit；
3. 生成本文件、四折 manifest、模型与 metric 配置的联合 SHA256；
4. 生成 seed/fold/source 全部 resolved configs 和 required-cells manifest；
5. 先运行一个 source 与一个 Custom fold 的 seed-0 单 epoch smoke；
6. smoke 通过后才扩展正式矩阵。
