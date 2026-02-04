# ===============================================================
# OmniVLA Inference Script
# ===============================================================
#
# 【功能说明】
# OmniVLA 推理示例代码，用于机器人导航任务
#
# 【重要提示】
# 如果要控制真实机器人，需要：
# 1. 在 run_omnivla() 函数中更新当前状态（GPS位置、图像等）
# 2. 注释掉第119行的 break 语句，使其持续运行
#
# ---------------------------
# 系统路径和基础库导入
# ---------------------------
import sys, os
sys.path.insert(0, '..')  # 将上级目录添加到Python路径，以便导入prismatic模块

# 标准库导入
import time, math, json                    # 时间、数学运算、JSON处理
from typing import Optional, Tuple, Type, Dict  # 类型注解
from dataclasses import dataclass          # 数据类装饰器

# 科学计算和图像处理库
import numpy as np                         # 数值计算
from PIL import Image                      # PIL图像处理
import torch                               # PyTorch深度学习框架
import torch.nn as nn                      # PyTorch神经网络模块
from torch.nn.utils.rnn import pad_sequence  # 序列填充工具
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式训练（本脚本未使用）
import torchvision.transforms as transforms  # 图像变换
import matplotlib.pyplot as plt            # 可视化绘图
import utm                                 # GPS坐标转UTM坐标系工具

# ---------------------------
# OmniVLA 自定义模块导入
# ---------------------------
from prismatic.vla.action_tokenizer import ActionTokenizer  # 动作tokenizer：将连续动作转换为离散token
from prismatic.models.projectors import ProprioProjector    # 本体感知投影器：将GPS位置投影到LLM维度
from prismatic.models.action_heads import L1RegressionActionHead_idcat, L1RegressionDistHead  # 动作头：预测连续动作
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1  # OmniVLA模型主体
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # 模型配置
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # 图像和数据处理器
from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # 提示词构建器
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask  # 动作掩码工具
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE  # 常量定义

# HuggingFace Transformers库
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

# ===============================================================
# 工具函数 (Utility Functions)
# ===============================================================

def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    """
    移除checkpoint中的DDP前缀

    【功能】从分布式训练保存的模型权重中移除 "module." 前缀
    【参数】state_dict: 模型权重字典
    【返回】清理后的权重字典
    """
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    加载模型checkpoint

    【功能】从指定路径加载训练好的模型权重
    【参数】
        module_name: 模块名称（如 "action_head", "pose_projector"）
        path: checkpoint所在目录
        step: 训练步数（如 120000）
        device: 加载到的设备（"cpu" 或 "cuda"）
    【返回】模型权重字典
    """
    # 兼容性处理：pose_projector 可能保存为 proprio_projector
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"

    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def count_parameters(module: nn.Module, name: str) -> None:
    """
    统计模块的可训练参数数量

    【功能】计算并打印模块中需要梯度更新的参数总数
    【参数】
        module: PyTorch模块
        name: 模块名称（用于打印）
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device_id,
    module_args: dict,
    to_bf16: bool = False,
) -> nn.Module:
    """
    初始化模型模块

    【功能】创建模块实例、加载权重、转换数据类型、移动到GPU
    【参数】
        module_class: 模块类（如 ProprioProjector）
        module_name: 模块名称
        cfg: 推理配置对象
        device_id: GPU设备ID
        module_args: 模块初始化参数字典
        to_bf16: 是否转换为bfloat16精度
    【返回】初始化好的模块
    """
    # 1. 创建模块实例
    module = module_class(**module_args)
    count_parameters(module, module_name)

    # 2. 如果需要恢复训练，加载checkpoint
    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    # 3. 转换为bfloat16精度（节省显存，提高速度）
    if to_bf16:
        module = module.to(torch.bfloat16)

    # 4. 移动到GPU
    module = module.to(device_id)
    return module

# ===============================================================
# 推理类 (Inference Class)
# ===============================================================
class Inference:
    """
    OmniVLA 推理引擎

    【功能】封装完整的推理流程，包括：
        - 数据预处理
        - 模型前向传播
        - 动作预测
        - 速度控制
        - 可视化保存
    """

    def __init__(self, save_dir, lan_inst_prompt, goal_utm, goal_compass, goal_image_PIL, action_tokenizer, processor):
        """
        初始化推理引擎

        【参数】
            save_dir: 可视化结果保存目录
            lan_inst_prompt: 语言指令（如 "move toward blue trash bin"）
            goal_utm: 目标GPS位置（UTM坐标系）
            goal_compass: 目标朝向角度（弧度）
            goal_image_PIL: 目标图像（PIL.Image对象）
            action_tokenizer: 动作tokenizer
            processor: 数据处理器（包含图像处理和文本tokenizer）
        """
        self.tick_rate = 3                      # 控制频率：3 Hz
        self.lan_inst_prompt = lan_inst_prompt  # 语言指令
        self.goal_utm = goal_utm                # 目标UTM坐标
        self.goal_compass = goal_compass        # 目标朝向
        self.goal_image_PIL = goal_image_PIL    # 目标图像
        self.action_tokenizer = action_tokenizer  # 动作tokenizer
        self.processor = processor              # 数据处理器
        self.count_id = 0                       # 推理计数器（用于保存文件命名）
        self.linear, self.angular = 0.0, 0.0   # 当前速度命令
        self.datastore_path_image = save_dir   # 可视化保存路径

    # ----------------------------
    # 静态工具方法 (Static Utility Methods)
    # ----------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        """
        计算两点之间的相对位置

        【参数】(x_a, y_a): 点A坐标，(x_b, y_b): 点B坐标
        【返回】(delta_x, delta_y): B相对于A的位置差
        """
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        """
        将世界坐标系下的向量转换到机器人局部坐标系

        【功能】根据机器人朝向，将全局坐标系下的位置差转换为机器人视角的前后左右
        【参数】
            delta_x, delta_y: 世界坐标系下的位置差
            heading_a_rad: 机器人朝向角度（弧度）
        【返回】
            rel_x: 机器人前进方向的距离
            rel_y: 机器人左侧方向的距离
        """
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    # ----------------------------
    # 主循环 (Main Loop)
    # ----------------------------
    def run(self):
        """
        运行推理主循环

        【功能】按照设定的频率（3 Hz）执行推理
        【注意】当前有 break 语句，只执行一次。实际部署时需要注释掉 break
        """
        loop_time = 1 / self.tick_rate  # 计算循环周期：1/3 秒
        start_time = time.time()
        while True:
            if time.time() - start_time > loop_time:
                self.tick()              # 执行一次推理
                start_time = time.time()
                break  # ⚠️ 仅执行一次！实际部署时需要注释掉这行

    def tick(self):
        """
        单次推理tick

        【功能】调用 run_omnivla() 执行一次完整推理，更新速度命令
        """
        self.linear, self.angular = self.run_omnivla()

    # ----------------------------
    # OmniVLA 核心推理函数
    # ----------------------------
    def run_omnivla(self):
        """
        执行一次完整的 OmniVLA 推理

        【流程】
        1. 加载当前GPS位置和朝向
        2. 计算目标位置的归一化表示
        3. 加载当前观测图像
        4. 准备输入数据batch
        5. 执行模型前向传播
        6. 预测waypoints
        7. 选择waypoint并计算速度命令
        8. 保存可视化结果

        【返回】(linear_vel, angular_vel): 线速度和角速度命令
        """
        # ========== 第1步：参数设置 ==========
        thres_dist = 30.0                # 目标距离阈值（米）：超过此距离会被截断
        metric_waypoint_spacing = 0.1    # waypoint间距（米）：用于归一化

        # ========== 第2步：加载当前GPS位置和朝向 ==========
        # ⚠️ 实际部署时，这里应该从传感器读取实时数据
        current_lat = 37.87371258374039   # 当前纬度
        current_lon = -122.26729417226024 # 当前经度
        current_compass = 270.0           # 当前朝向（度）：0=北，90=东，180=南，270=西

        # 将GPS坐标转换为UTM坐标系（单位：米）
        cur_utm = utm.from_latlon(current_lat, current_lon)
        # 将朝向转换为弧度，并取反（因为compass定义与数学角度相反）
        cur_compass = -float(current_compass) / 180.0 * math.pi

        # ========== 第3步：计算目标位置的局部表示 ==========
        # 3.1 计算世界坐标系下的相对位置
        delta_x, delta_y = self.calculate_relative_position(
            cur_utm[0], cur_utm[1],      # 当前UTM坐标
            self.goal_utm[0], self.goal_utm[1]  # 目标UTM坐标
        )

        # 3.2 转换到机器人局部坐标系（前进方向为X轴，左侧为Y轴）
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)

        # 3.3 距离截断：如果目标太远，限制在阈值范围内
        radius = np.sqrt(relative_x**2 + relative_y**2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius

        # 3.4 归一化目标位置表示（4维向量）
        goal_pose_loc_norm = np.array([
            relative_y / metric_waypoint_spacing,   # 归一化的左右位置
            -relative_x / metric_waypoint_spacing,  # 归一化的前后位置（取反）
            np.cos(self.goal_compass - cur_compass),  # 相对朝向的cos值
            np.sin(self.goal_compass - cur_compass)   # 相对朝向的sin值
        ])  # shape: [4]

        # ========== 第4步：加载当前观测图像 ==========
        # ⚠️ 实际部署时，这里应该从摄像头读取实时图像
        current_image_path = "./inference/current_img.jpg"
        current_image_PIL = Image.open(current_image_path).convert("RGB")

        # ========== 第5步：准备语言指令 ==========
        # 如果启用语言提示，使用指令；否则使用占位符
        lan_inst = self.lan_inst_prompt if lan_prompt else "xxxx"

        # ========== 第6步：数据预处理 ==========
        # 将所有输入（图像、语言、GPS）转换为模型所需的batch格式
        batch = self.data_transformer_omnivla(
            current_image_PIL,           # 当前观测图像
            lan_inst,                    # 语言指令
            self.goal_image_PIL,         # 目标图像
            goal_pose_loc_norm,          # 归一化的目标位置
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor
        )

        # ========== 第7步：模型前向传播 ==========
        # 执行VLA模型推理，预测未来的waypoints
        actions, modality_id = self.run_forward_pass(
            vla=vla.eval(),                      # VLA主模型（评估模式）
            action_head=action_head.eval(),      # 动作头（评估模式）
            noisy_action_projector=None,         # 扩散模型投影器（未使用）
            pose_projector=pose_projector.eval(),  # GPS位置投影器
            batch=batch,                         # 输入数据batch
            action_tokenizer=self.action_tokenizer,
            device_id=device_id,
            use_l1_regression=True,              # 使用L1回归预测动作
            use_diffusion=False,                 # 不使用扩散模型
            use_film=False,                      # 不使用FiLM
            num_patches=NUM_PATCHES,             # 视觉patch数量
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
            mode="train",
            idrun=self.count_id,
        )
        self.count_id += 1  # 推理计数器+1

        # ========== 第8步：提取预测的waypoints ==========
        waypoints = actions.float().cpu().numpy()  # shape: [1, 8, 4]
        # waypoints[0, i, :] = [dx, dy, heading_x, heading_y]
        # 8个时间步，每个4维：前后位移、左右位移、朝向x分量、朝向y分量

        # ========== 第9步：选择目标waypoint ==========
        waypoint_select = 4  # 选择第5个waypoint（索引从0开始）
        chosen_waypoint = waypoints[0][waypoint_select].copy()  # shape: [4]
        chosen_waypoint[:2] *= metric_waypoint_spacing  # 反归一化：转换回米
        dx, dy, hx, hy = chosen_waypoint
        # dx: 前进方向的位移（米）
        # dy: 左侧方向的位移（米）
        # hx, hy: 目标朝向的单位向量分量

        # ========== 第10步：PD控制器计算速度命令 ==========
        # 【功能】根据目标waypoint计算线速度和角速度
        EPS = 1e-8  # 极小值，用于避免除零
        DT = 1 / 3  # 时间步长（秒）：对应3 Hz控制频率

        # 定义clip_angle函数（将角度限制在[-π, π]范围内）
        def clip_angle(angle):
            return np.arctan2(np.sin(angle), np.cos(angle))

        # 情况1：目标waypoint几乎在当前位置（dx≈0, dy≈0）
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0  # 不前进
            # 仅调整朝向：根据目标朝向计算角速度
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT

        # 情况2：目标在正左或正右（dx≈0, dy≠0）
        elif np.abs(dx) < EPS:
            linear_vel_value = 0  # 不前进
            # 转向90度：向左或向右
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)

        # 情况3：一般情况（dx≠0）
        else:
            linear_vel_value = dx / DT  # 线速度：前进距离/时间
            angular_vel_value = np.arctan(dy / dx) / DT  # 角速度：转向角度/时间

        # 初步限制速度范围
        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)    # 线速度：[0, 0.5] m/s
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)  # 角速度：[-1.0, 1.0] rad/s

        # ========== 第11步：速度限制（考虑机器人物理约束）==========
        # 【功能】确保速度命令在机器人的物理能力范围内
        maxv, maxw = 0.3, 0.3  # 最大线速度（m/s）和最大角速度（rad/s）

        # 情况1：线速度在限制范围内
        if np.abs(linear_vel_value) <= maxv:
            # 1.1 角速度也在范围内：直接使用
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            # 1.2 角速度超限：按比例缩放
            else:
                rd = linear_vel_value / angular_vel_value  # 转弯半径
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)

        # 情况2：线速度超限
        else:
            # 2.1 几乎不转弯（角速度≈0）
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            # 2.2 需要转弯
            else:
                rd = linear_vel_value / angular_vel_value  # 转弯半径
                # 2.2.1 转弯半径大：限制线速度
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                # 2.2.2 转弯半径小：限制角速度
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)

        # ========== 第12步：保存可视化结果 ==========
        self.save_robot_behavior(
            current_image_PIL,           # 当前观测图像
            self.goal_image_PIL,         # 目标图像
            goal_pose_loc_norm,          # 归一化的目标位置
            waypoints[0],                # 预测的8个waypoints
            linear_vel_value_limit,      # 最终线速度命令
            angular_vel_value_limit,     # 最终角速度命令
            metric_waypoint_spacing,     # waypoint间距
            modality_id.cpu().numpy()    # 模态ID
        )

        # ========== 第13步：输出速度命令 ==========
        print("linear angular", linear_vel_value_limit, angular_vel_value_limit)
        return linear_vel_value_limit, angular_vel_value_limit

    # ----------------------------
    # 保存机器人行为可视化 (Save Robot Behavior Visualization)
    # ----------------------------
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, waypoints,
                            linear_vel, angular_vel, metric_waypoint_spacing, mask_number):
        """
        生成并保存推理结果的可视化图像

        【功能】创建包含3个子图的可视化：
            - 左上：当前观测图像
            - 左下：目标图像
            - 右侧：预测的轨迹图（8个waypoints）

        【参数】
            cur_img: 当前观测图像（PIL.Image）
            goal_img: 目标图像（PIL.Image）
            goal_pose: 归一化的目标位置 [4]
            waypoints: 预测的waypoints [8, 4]
            linear_vel: 线速度命令（m/s）
            angular_vel: 角速度命令（rad/s）
            metric_waypoint_spacing: waypoint间距（米）
            mask_number: 模态ID（用于标注）
        """
        # ========== 创建画布和子图布局 ==========
        fig = plt.figure(figsize=(34, 16), dpi=80)  # 大尺寸画布
        gs = fig.add_gridspec(2, 2)  # 2×2网格布局
        ax_ob = fig.add_subplot(gs[0, 0])      # 左上：当前图像
        ax_goal = fig.add_subplot(gs[1, 0])    # 左下：目标图像
        ax_graph_pos = fig.add_subplot(gs[:, 1])  # 右侧：轨迹图（占2行）

        # ========== 显示图像 ==========
        ax_ob.imshow(np.array(cur_img).astype(np.uint8))    # 当前观测
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))  # 目标图像

        # ========== 绘制预测轨迹 ==========
        # 提取waypoints的位置信息（忽略朝向）
        x_seq = waypoints[:, 0]      # 前进方向的位移序列 [8]
        y_seq_inv = -waypoints[:, 1]  # 左右方向的位移序列（取反用于绘图）[8]

        # 绘制轨迹：从原点(0,0)开始，连接8个waypoints
        ax_graph_pos.plot(
            np.insert(y_seq_inv, 0, 0.0),  # Y坐标：插入原点0
            np.insert(x_seq, 0, 0.0),      # X坐标：插入原点0
            linewidth=4.0,                 # 线宽
            markersize=12,                 # 点大小
            marker='o',                    # 圆点标记
            color='blue'                   # 蓝色
        )

        # ========== 添加模态类型标注 ==========
        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only",        # 0: 仅卫星图
            "pose and satellite",    # 1: GPS + 卫星图
            "satellite and image",   # 2: 卫星图 + 目标图像
            "all",                   # 3: 全部模态
            "pose only",             # 4: 仅GPS
            "pose and image",        # 5: GPS + 目标图像
            "image only",            # 6: 仅目标图像
            "language only",         # 7: 仅语言
            "language and pose"      # 8: 语言 + GPS
        ]
        if mask_type < len(mask_texts):
            # 在轨迹图右下角标注当前使用的模态
            ax_graph_pos.annotate(
                mask_texts[mask_type],
                xy=(1.0, 0.0),           # 锚点位置
                xytext=(-20, 20),        # 文字偏移
                fontsize=18,
                textcoords='offset points'
            )

        # ========== 设置标题和样式 ==========
        ax_ob.set_title("Egocentric current image", fontsize=18)  # 当前图像标题
        ax_goal.set_title("Egocentric goal image", fontsize=18)   # 目标图像标题
        ax_graph_pos.tick_params(axis='x', labelsize=15)  # X轴刻度字体大小
        ax_graph_pos.tick_params(axis='y', labelsize=15)  # Y轴刻度字体大小

        # ========== 绘制目标位置（如果使用GPS模态）==========
        # 模态ID 1, 3, 4, 5, 8 包含GPS信息
        if int(mask_number[0]) in [1, 3, 4, 5, 8]:
            # 在轨迹图上用红色星号标记目标位置
            ax_graph_pos.plot(
                -goal_pose[1],      # Y坐标（左右）
                goal_pose[0],       # X坐标（前后）
                marker='*',         # 星号标记
                color='red',        # 红色
                markersize=15
            )
        else:
            # 如果不使用GPS，设置固定的坐标轴范围
            ax_graph_pos.set_xlim(-3.0, 3.0)
            ax_graph_pos.set_ylim(-0.1, 10.0)

        # 统一设置坐标轴范围
        ax_graph_pos.set_xlim(-3.0, 3.0)   # X轴：左右3米
        ax_graph_pos.set_ylim(-0.1, 10.0)  # Y轴：前方10米

        ax_graph_pos.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)

        # ========== 保存可视化图像 ==========
        save_path = os.path.join(self.datastore_path_image, f"{self.count_id}_ex.jpg")
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")

    # ----------------------------
    # 自定义数据整理器 (Custom Collator)
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        """
        将多个数据实例整理成batch

        【功能】
        - 填充序列到相同长度
        - 合并当前图像和目标图像
        - 创建attention mask
        - 组织成模型输入格式

        【参数】
            instances: 数据实例列表
            model_max_length: 最大序列长度
            pad_token_id: 填充token的ID
            padding_side: 填充方向（"right" 或 "left"）
            pixel_values_dtype: 图像数据类型

        【返回】包含所有输入数据的字典
        """
        IGNORE_INDEX = -100  # 损失计算时忽略的标签值

        # ========== 处理文本序列 ==========
        # 填充input_ids到相同长度
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        # 填充labels到相同长度
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        # 截断到最大长度
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        # 创建attention mask：非填充位置为True
        attention_mask = input_ids.ne(pad_token_id)

        # ========== 处理图像数据 ==========
        pixel_values = [inst["pixel_values_current"] for inst in instances]

        # 提取数据集名称（如果有）
        if "dataset_name" in instances[0]:
            dataset_names = [inst["dataset_name"] for inst in instances]
        else:
            dataset_names = None

        # 合并当前图像和目标图像
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                # 有目标图像：沿通道维度拼接
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
                # 结果shape: [B, 2, C, H, W] - 2张图像（当前+目标）
            else:
                # 仅当前图像
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        # ========== 处理动作和位置数据 ==========
        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        # ========== 组织输出字典 ==========
        output = dict(
            pixel_values=pixel_values.to(),  # 图像数据
            input_ids=input_ids,             # 文本token IDs
            attention_mask=attention_mask,   # 注意力掩码
            labels=labels,                   # 标签（用于训练）
            actions=actions,                 # 动作序列
            goal_pose=goal_pose,             # 目标位置
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

    # ----------------------------
    # 数据格式转换 (Transform Data to Dataset Format)
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                           current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer,
                           base_tokenizer, image_transform, predict_stop_token=True):
        """
        将原始数据转换为模型输入格式

        【功能】
        1. 将动作序列tokenize为字符串
        2. 构建对话格式的提示词
        3. 处理图像（resize + normalize）
        4. 创建训练标签（仅动作部分有梯度）

        【参数】
            inst_obj: 语言指令（或 "xxxx" 表示无指令）
            actions: 动作序列 [8, 4]
            goal_pose_cos_sin: 目标位置 [4]
            current_image_PIL: 当前图像
            goal_image_PIL: 目标图像
            prompt_builder: 提示词构建器类
            action_tokenizer: 动作tokenizer
            base_tokenizer: 文本tokenizer
            image_transform: 图像变换函数
            predict_stop_token: 是否预测停止token

        【返回】包含所有处理后数据的字典
        """
        IGNORE_INDEX = -100  # 损失计算时忽略的标签值

        # ========== 第1步：动作tokenization ==========
        current_action = actions[0]      # 当前动作 [4]
        future_actions = actions[1:]     # 未来7个动作 [7, 4]
        # 将动作转换为token字符串
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)  # token字符串长度

        # ========== 第2步：构建对话格式 ==========
        if inst_obj == "xxxx":
            # 无语言指令的情况
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            # 有语言指令的情况
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        # ========== 第3步：构建提示词 ==========
        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # ========== 第4步：文本tokenization ==========
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        # 仅对动作部分计算损失，其他部分设为IGNORE_INDEX
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # ========== 第5步：图像处理 ==========
        # 应用图像变换：resize到224×224 + normalize
        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        dataset_name = "lelan"  # 数据集名称（用于归一化统计）

        # ========== 第6步：返回处理后的数据 ==========
        return dict(
            pixel_values_current=pixel_values_current,  # 当前图像 [C, H, W]
            pixel_values_goal=pixel_values_goal,        # 目标图像 [C, H, W]
            input_ids=input_ids,                        # 文本token IDs
            labels=labels,                              # 训练标签
            dataset_name=dataset_name,                  # 数据集名称
            actions=torch.as_tensor(actions),           # 动作序列 [8, 4]
            goal_pose=goal_pose_cos_sin,                # 目标位置 [4]
            img_PIL=current_image_PIL,                  # 原始图像（用于可视化）
            inst=inst_obj,                              # 语言指令
        )

    # ----------------------------
    # OmniVLA 数据转换器 (Data Transformer for OmniVLA)
    # ----------------------------
    def data_transformer_omnivla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm,
                                 prompt_builder, action_tokenizer, processor):
        """
        OmniVLA 专用的数据转换函数

        【功能】将原始输入转换为模型batch格式
        【注意】这里的 actions 是虚拟的（随机生成），因为推理时不需要真实动作标签

        【参数】
            current_image_PIL: 当前观测图像
            lan_inst: 语言指令
            goal_image_PIL: 目标图像
            goal_pose_loc_norm: 归一化的目标位置
            prompt_builder: 提示词构建器
            action_tokenizer: 动作tokenizer
            processor: 数据处理器

        【返回】模型输入batch
        """
        # 生成虚拟动作（推理时不需要真实标签，仅用于占位）
        actions = np.random.rand(8, 4)  # [8, 4] - 8个时间步，每个4维
        goal_pose_cos_sin = goal_pose_loc_norm

        # 转换数据格式
        batch_data = self.transform_datatype(
            lan_inst,                    # 语言指令
            actions,                     # 虚拟动作
            goal_pose_cos_sin,           # 目标位置
            current_image_PIL,           # 当前图像
            goal_image_PIL,              # 目标图像
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        # 整理成batch格式
        batch = self.collator_custom(
            instances=[batch_data],      # 单个实例的列表
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right"
        )
        return batch

    # ----------------------------
    # 执行前向传播 (Run Forward Pass)
    # ----------------------------
    def run_forward_pass(self, vla, action_head, noisy_action_projector, pose_projector,
                         batch, action_tokenizer, device_id, use_l1_regression, use_diffusion,
                         use_film, num_patches, compute_diffusion_l1=False,
                         num_diffusion_steps_train=None, mode="vali", idrun=0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        执行模型前向传播并预测动作

        【流程】
        1. 确定输入模态ID（根据全局变量）
        2. VLA模型前向传播（Vision + Language → LLM隐藏状态）
        3. 提取动作对应的隐藏状态
        4. Action Head预测连续动作

        【参数】
            vla: VLA主模型
            action_head: 动作预测头
            noisy_action_projector: 扩散模型投影器（未使用）
            pose_projector: GPS位置投影器
            batch: 输入数据batch
            action_tokenizer: 动作tokenizer（未使用）
            device_id: GPU设备
            use_l1_regression: 是否使用L1回归（未使用）
            use_diffusion: 是否使用扩散模型
            use_film: 是否使用FiLM
            num_patches: 视觉patch数量
            compute_diffusion_l1: 是否计算扩散L1损失（未使用）
            num_diffusion_steps_train: 扩散步数（未使用）
            mode: 模式（"train" 或 "vali"）（未使用）
            idrun: 运行ID（未使用）

        【返回】(predicted_actions, modality_id)
            predicted_actions: 预测的动作 [B, 8, 4]
            modality_id: 模态ID [1]
        """
        metrics = {}
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

        # ========== 第1步：确定输入模态ID ==========
        # 【功能】根据4个全局布尔变量确定使用哪种输入组合
        # satellite, lan_prompt, pose_goal, image_goal 的8种组合
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([0], dtype=torch.float32)  # 仅卫星图
        elif satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([1], dtype=torch.float32)  # 卫星图 + GPS
        elif satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([2], dtype=torch.float32)  # 卫星图 + 目标图像
        elif satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([3], dtype=torch.float32)  # 卫星图 + GPS + 目标图像
        elif not satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([4], dtype=torch.float32)  # 仅GPS
        elif not satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([5], dtype=torch.float32)  # GPS + 目标图像
        elif not satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([6], dtype=torch.float32)  # 仅目标图像（当前配置）
        elif not satellite and lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([7], dtype=torch.float32)  # 仅语言
        elif not satellite and lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([8], dtype=torch.float32)  # 语言 + GPS

        # ========== 第2步：VLA模型前向传播 ==========
        # 【功能】将多模态输入（图像+文本+GPS）编码为LLM隐藏状态
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 导入类型（避免循环导入）
            from transformers.modeling_outputs import CausalLMOutputWithPast

            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),              # 文本token IDs
                attention_mask=batch["attention_mask"].to(device_id),    # 注意力掩码
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),  # 图像数据
                modality_id=modality_id.to(torch.bfloat16).to(device_id),  # 模态ID
                labels=batch["labels"].to(device_id),                    # 标签（推理时不使用）
                output_hidden_states=True,                               # 输出隐藏状态
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),  # GPS位置
                proprio_projector=pose_projector,                        # GPS投影器
                noisy_actions=noisy_actions if use_diffusion else None,  # 扩散模型（未使用）
                noisy_action_projector=noisy_action_projector if use_diffusion else None,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                use_film=use_film,                                       # FiLM（未使用）
            )

        # ========== 第3步：提取动作对应的隐藏状态 ==========
        # 准备数据用于提取动作隐藏状态
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)  # 当前动作的掩码
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)      # 未来动作的掩码

        # 获取最后一层的隐藏状态
        last_hidden_states = output.hidden_states[-1]  # shape: (B, seq_len, D)

        # 提取文本部分的隐藏状态（跳过视觉patches）
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # num_patches: 视觉patch数量（当前图像 + 目标图像 + GPS位置）
        # -1: 去掉最后一个token

        # 提取动作部分的隐藏状态
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]  # 选择动作对应的位置
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)     # 重塑为 [B, 32, D]
            .to(torch.bfloat16)
        )  # NUM_ACTIONS_CHUNK=8, ACTION_DIM=4 → 32个token

        # ========== 第4步：Action Head预测连续动作 ==========
        with torch.no_grad():
            # 从隐藏状态预测连续动作值
            predicted_actions = action_head.predict_action(
                actions_hidden_states,                                   # 动作隐藏状态 [B, 32, D]
                modality_id.to(torch.bfloat16).to(device_id)            # 模态ID
            )  # 输出: [B, 8, 4] - 8个waypoints，每个4维

        # ========== 第5步：返回预测结果 ==========
        return predicted_actions, modality_id


# ===============================================================
# 推理配置 (Inference Configuration)
# ===============================================================
class InferenceConfig:
    """
    推理配置类

    【参数说明】
        resume: 是否从checkpoint恢复
        vla_path: 模型路径
        resume_step: 恢复的训练步数
        use_l1_regression: 使用L1回归预测动作（True）
        use_diffusion: 使用扩散模型预测动作（False，未启用）
        use_film: 使用FiLM调制（False，未启用）
        num_images_in_input: 输入图像数量（2 = 当前图像 + 目标图像）
        use_lora: 使用LoRA微调（True）
        lora_rank: LoRA秩（32）
        lora_dropout: LoRA dropout率（0.0）
    """
    resume: bool = True
    vla_path: str = "./omnivla-original"           # 原始模型路径
    resume_step: Optional[int] = 120000            # checkpoint步数
    #vla_path: str = "./omnivla-finetuned-cast"    # 微调模型路径（备选）
    #resume_step: Optional[int] = 210000           # 微调checkpoint步数（备选）
    use_l1_regression: bool = True                 # 使用L1回归
    use_diffusion: bool = False                    # 不使用扩散模型
    use_film: bool = False                         # 不使用FiLM
    num_images_in_input: int = 2                   # 2张图像输入
    use_lora: bool = True                          # 使用LoRA
    lora_rank: int = 32                            # LoRA秩=32
    lora_dropout: float = 0.0                      # 无dropout

def define_model(cfg: InferenceConfig) -> None:
    """
    定义并加载模型

    【功能】
    1. 设置GPU设备
    2. 注册OpenVLA模型到HuggingFace
    3. 加载VLA主模型
    4. 加载GPS位置投影器
    5. 加载Action Head
    6. 创建Action Tokenizer

    【参数】cfg: 推理配置对象
    【返回】(vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor)
    """
    # ========== 第1步：路径和设备设置 ==========
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")

    # GPU设置
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_id.type == "cuda":
        torch.cuda.set_device(device_id.index or 0)  # 设置当前GPU
        torch.cuda.empty_cache()                     # 清空缓存

    # 打印常量信息
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"      # 8个时间步
        f"\tACTION_DIM: {ACTION_DIM}\n"                    # 每个动作4维
        f"\tPOSE_DIM: {POSE_DIM}\n"                        # GPS位置4维
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # ========== 第2步：注册OpenVLA模型到HuggingFace ==========
    # 【注意】如果模型在HF Hub上，则不需要手动注册
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    # ========== 第3步：加载Processor和VLA主模型 ==========
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,    # 使用bfloat16精度
        low_cpu_mem_usage=True,        # 低CPU内存占用
    ).to(device_id)

    # 设置输入图像数量（2张：当前+目标）
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)

    # ========== 第4步：加载GPS位置投影器 ==========
    pose_projector = init_module(
        ProprioProjector,              # GPS位置投影器类
        "pose_projector",              # 模块名称
        cfg,                           # 配置
        device_id,                     # 设备
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},  # 参数：LLM维度和GPS维度
    )

    # ========== 第5步：加载Action Head ==========
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead_idcat,  # L1回归Action Head类
            "action_head",                 # 模块名称
            cfg,                           # 配置
            device_id,                     # 设备
            {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,                  # 转换为bfloat16
        )

    # ========== 第6步：计算视觉patch数量 ==========
    # 每张图像的patch数 × 图像数量
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    NUM_PATCHES += 1  # +1 用于GPS位置

    # ========== 第7步：创建Action Tokenizer ==========
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ========== 第8步：返回所有组件 ==========
    return vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor

# ===============================================================
# 主入口 (Main Entry)
# ===============================================================
if __name__ == "__main__":
    # ========== 第1步：选择输入模态 ==========
    # 【功能】设置4个全局布尔变量，控制使用哪些输入
    pose_goal = False      # ❌ 不使用GPS目标
    satellite = False      # ❌ 不使用卫星图
    image_goal = True      # ✅ 使用目标图像
    lan_prompt = False     # ❌ 不使用语言指令
    # 结果：modality_id = 6（仅目标图像模式）

    # ========== 第2步：定义目标信息 ==========
    # 语言指令（当前未使用）
    lan_inst_prompt = "move toward blue trash bin"

    # GPS目标（当前未使用）
    goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
    goal_utm = utm.from_latlon(goal_lat, goal_lon)  # GPS → UTM坐标系
    goal_compass = -float(goal_compass) / 180.0 * math.pi  # 度 → 弧度

    # 目标图像（当前使用）
    goal_image_PIL = Image.open("./inference/goal_img.jpg").convert("RGB")

    # ========== 第3步：加载模型 ==========
    cfg = InferenceConfig()
    vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor = define_model(cfg)

    # ========== 第4步：创建推理对象并运行 ==========
    inference = Inference(
        save_dir="./inference",              # 可视化保存目录
        lan_inst_prompt=lan_inst_prompt,     # 语言指令
        goal_utm=goal_utm,                   # GPS目标（UTM坐标）
        goal_compass=goal_compass,           # 目标朝向
        goal_image_PIL=goal_image_PIL,       # 目标图像
        action_tokenizer=action_tokenizer,   # 动作tokenizer
        processor=processor,                 # 数据处理器
    )

    # 运行推理（执行一次tick）
    # 【注意】第119行的break使得只运行一次，实际部署需要注释掉
    inference.run()
