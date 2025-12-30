# OpenArm RL Deployment

本项目实现了基于 SERL (Software for Evolving RL) 框架的 OpenArm 机械臂强化学习部署系统。系统采用分布式架构，分为硬件服务器、训练端 (Learner) 和 执行端 (Actor)。

## 1. 系统架构

系统由三个独立进程组成，通过网络进行通信：

1.  **Hardware Server (`openarm_server.py`)**:
    -   **功能**: 负责与底层硬件（或 Mock 硬件）交互，屏蔽硬件细节。
    -   **接口**: 提供 Flask HTTP API (`/getstate`, `/pose`, `/jointreset`)。
    -   **数据**: 获取关节角度、夹爪状态、摄像头图像；接收笛卡尔位姿指令并执行 IK 解算。
    -   **可视化**: 集成 Viser 可视化工具，实时显示机器人状态。

2.  **RL Environment (`openarm_env.py`)**:
    -   **功能**: 标准 Gym 环境封装，负责将 RL 动作转换为 Server 指令，并将 Server 状态转换为 RL 观测。
    -   **特性**:
        -   支持双臂 (`both`) 或单臂 (`left`/`right`) 控制。
        -   **RelativeFrame**: 使用相对位姿控制 (Delta Position + Delta Rotation)。
        -   **Reward**: 训练时使用 Reward Classifier (分类器) 提供奖励信号。

3.  **RL Agent (`train_rl.py`)**:
    -   **功能**: 核心 RL 训练逻辑，基于 DrQ-v2 算法。
    -   **模式**:
        -   **Learner**: 负责从 Replay Buffer 采样并更新网络参数。
        -   **Actor**: 负责与环境交互，收集数据并发送给 Learner，同时从 Learner 同步最新参数。
    -   **Demo Loading**: 支持加载演示数据 (`rl_success_demos.pkl`) 加速训练，包含奖励重塑逻辑 (Success=0.5, Path=0.0)。

## 2. 快速开始

启动系统需要打开三个终端，分别运行以下命令：

### 第一步：启动硬件服务器
```bash
# 在 moqi_workspace/rl_deploy 目录下
python openarm_server.py
```
*注：如果未连接真实硬件，会自动回退到 Mock 模式。*

### 第二步：启动 Learner (训练端)
```bash
# 在 moqi_workspace/rl_deploy 目录下
./run_learner.sh
```
*脚本内容参考：*
```bash
python train_rl.py --learner --arm=right --exp_name=openarm_rl_test ...
```

### 第三步：启动 Actor (执行端)
```bash
# 在 moqi_workspace/rl_deploy 目录下
./run_actor.sh
```
*脚本内容参考：*
```bash
python train_rl.py --actor --arm=right --ip=<LEARNER_IP> ...
```

## 3. 关键文件说明

| 文件名 | 说明 |
| :--- | :--- |
| `openarm_server.py` | 硬件控制服务器，处理 IK 和相机数据。 |
| `openarm_env.py` | Gym 环境定义，处理观测空间 (Images + State) 和动作空间。 |
| `train_rl.py` | RL 训练主入口，包含 Actor 和 Learner 的逻辑。 |
| `run_learner.sh` | 启动 Learner 的脚本，配置了右臂控制 (`--arm=right`)。 |
| `run_actor.sh` | 启动 Actor 的脚本，连接到 Learner IP。 |
| `rl_success_demos.pkl` | 成功的演示轨迹数据，用于辅助训练。 |

## 4. 逻辑细节

### 数据流
1.  **Camera/Robot** -> `openarm_server.py` (采集图像和状态)
2.  `openarm_server.py` -> (HTTP JSON) -> `openarm_env.py` (获取观测)
3.  `openarm_env.py` -> `train_rl.py` (Actor) (Step 环境)
4.  `train_rl.py` (Actor) -> (Action) -> `openarm_env.py`
5.  `openarm_env.py` -> (HTTP JSON) -> `openarm_server.py` (发送控制指令)
6.  `openarm_server.py` -> **Robot Hardware** (执行运动)

### 奖励机制
-   **Sparse Reward**: 原始环境默认不提供奖励。
-   **Classifier Reward**: 使用预训练的 Reward Classifier (ResNet) 对当前图像进行打分，作为 RL 的奖励信号。
-   **Demo Reward**: 加载演示数据时，最后 10 步被标记为成功 (Reward=0.5)，其余步骤为路径 (Reward=0.0)。

### 坐标系与控制
-   **观测**: TCP Pose (End-Effector) + Gripper Pos + Images.
-   **动作**: Delta XYZ + Delta RPY + Gripper Command.
-   **转换**: `RelativeFrame` Wrapper 将动作转换为相对于当前 TCP 的增量，`openarm_server` 将其转换为绝对位姿并通过 IK 解算为关节角度。