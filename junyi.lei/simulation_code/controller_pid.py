wind_flag = True
# Implement a controller

import math
import os
import atexit

def controller(state, target_pos, dt, wind_enabled=False):
    """
    完整 PID 控制器 (带数据记录与防崩溃保护)
    Complete PID Controller (with Data Logging & Crash Protection)
    """
    # ==============================================================
    # 0. 极限安全保护 (Extreme Safety Checks)
    # 防止 dt=0 导致除以 0，或者传入了非法状态 (NaN)
    # Prevent division by zero or invalid state inputs
    # ==============================================================
    if dt <= 1e-4:
        return (0.0, 0.0, 0.0, 0.0)
    
    for val in state:
        if math.isnan(val) or math.isinf(val):
            return (0.0, 0.0, 0.0, 0.0)

    # 1. 状态解析 (State & Target Parsing)
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

        # 在状态解析之后，添加目标切换检测与重置逻辑
    # 假设 state 和 target_pos 已经解析完成
    if not hasattr(controller, 'last_target'):
        controller.last_target = None

    # 检测目标是否发生变化
    if controller.last_target != target_pos:
        # 重置积分项（XY 和 Z）
        if hasattr(controller, 'integral_error'):
            controller.integral_error = [0.0, 0.0, 0.0]
        else:
            controller.integral_error = [0.0, 0.0, 0.0]  # 若尚未初始化则直接赋值
        # 重置微分历史（包括偏航）
        if hasattr(controller, 'prev_error'):
            controller.prev_error = [0.0, 0.0, 0.0, 0.0]
        else:
            controller.prev_error = [0.0, 0.0, 0.0, 0.0]
        # 更新记录的目标值
        controller.last_target = target_pos

    # ==============================================================
    # 🌟 数据记录模块 (Data Logging Module)
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 50     # 每50帧写入一次磁盘
        controller.file_path = "data_pid.csv"
        controller.sim_time = 0.0
        
        # 优化点：不再检查文件是否存在，而是直接用 'w' 模式打开。
        # 这会自动清空旧的 data.csv 并写入全新的表头。
        with open(controller.file_path, 'w') as f:
            f.write("time,target_x,target_y,target_z,target_yaw,x,y,z,yaw\n")
        
        # 退出时保存缓冲区剩余数据
        def flush_buffer():
            if hasattr(controller, "buffer") and controller.buffer:
                with open(controller.file_path, 'a') as f:
                    f.writelines(controller.buffer)
        
        import atexit
        atexit.register(flush_buffer)
        controller.is_initialized_csv = True

    # 累加时间并记录
    controller.sim_time += dt
    # 使用字符串格式化确保数据整齐
    record = f"{controller.sim_time:.4f},{tx:.4f},{ty:.4f},{tz:.4f},{tyaw:.4f},{x:.4f},{y:.4f},{z:.4f},{yaw:.4f}\n"
    controller.buffer.append(record)

    # 达到阈值时追加写入磁盘 (使用 'a' 模式)
    if len(controller.buffer) >= controller.buffer_limit:
        with open(controller.file_path, 'a') as f:
            f.writelines(controller.buffer)
        controller.buffer.clear()

    # ==============================================================
    # 🚀 核心 PID 参数 (Core PID Parameters) 
    # [可调整区域 / Tuning Area]
    # ==============================================================
    # Kp: 比例，控制靠近目标的速度 / Proportional, speed towards target
    # Ki: 积分，用来抵抗风等持续干扰 / Integral, to fight steady wind
    # Kd: 微分，相当于刹车，防止超调 / Derivative, acts as a brake, prevents overshoot
    Kp_xy = 1.1
    Ki_xy = 0.035
    Kd_xy = 0.1

    Kp_z = 2.0
    Ki_z = 0.0
    Kd_z = 0.1

    Kp_yaw = 2.0
    Kd_yaw = 0.1
    # ==============================================================
    # 初始化历史变量 (Initialize Persistent Variables)
    # ==============================================================
    if not hasattr(controller, 'integral_error'):
        controller.integral_error = [0.0, 0.0, 0.0] # [x, y, z]
    if not hasattr(controller, 'prev_error'):
        controller.prev_error = [0.0, 0.0, 0.0, 0.0] # [x, y, z, yaw]

    # ==============================================================
    # 误差计算 (Error Calculation in World Frame)
    # ==============================================================
    ex = tx - x
    ey = ty - y
    ez = tz - z
    
    # 偏航角限制在 [-pi, pi] 之间 / Wrap yaw error to [-pi, pi]
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi

    # ==============================================================
    # P, I, D 项计算 (Calculate P, I, D terms)
    # ==============================================================
    # 微分 (Derivative)
    dx = (ex - controller.prev_error[0]) / dt
    dy = (ey - controller.prev_error[1]) / dt
    dz = (ez - controller.prev_error[2]) / dt
    dyaw = (eyaw - controller.prev_error[3]) / dt

    # 积分与抗饱和 (Integral & Anti-Windup)
    controller.integral_error[0] += ex * dt
    controller.integral_error[1] += ey * dt
    controller.integral_error[2] += ez * dt

    max_integral = 0.2 # 防止积分爆炸 / Prevent integral windup
    for i in range(3):
        controller.integral_error[i] = max(-max_integral, min(controller.integral_error[i], max_integral))

    ix, iy, iz = controller.integral_error

    # 计算全局速度指令 / Compute World Frame Velocity
    vx_world = (Kp_xy * ex) + (Ki_xy * ix) + (Kd_xy * dx)
    vy_world = (Kp_xy * ey) + (Ki_xy * iy) + (Kd_xy * dy)
    vz_world = (Kp_z * ez)  + (Ki_z * iz)  + (Kd_z * dz)
    yaw_rate_cmd = (Kp_yaw * eyaw) + (Kd_yaw * dyaw)

    # ==============================================================
    # 坐标系转换：世界 -> 机身 (World Frame to Body Frame)
    # ==============================================================
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    vx_body = vx_world * cos_yaw + vy_world * sin_yaw
    vy_body = -vx_world * sin_yaw + vy_world * cos_yaw
    vz_body = vz_world 

    # 保存误差供下一帧使用 / Save error for next frame
    controller.prev_error = [ex, ey, ez, eyaw]

    # ==============================================================
    # 最终输出限幅与清洗 (Final Output Clamping & Cleaning)
    # 彻底杜绝 NaN 传给模拟器 / Absolutely prevent NaN sent to simulator
    # ==============================================================
    def clean_and_clamp(val, min_val, max_val):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return max(min_val, min(val, max_val))

    vx_body = clean_and_clamp(vx_body, -1.0, 1.0)
    vy_body = clean_and_clamp(vy_body, -1.0, 1.0)
    vz_body = clean_and_clamp(vz_body, -1.0, 1.0)
    yaw_rate_cmd = clean_and_clamp(yaw_rate_cmd, -1.74, 1.74)

    def is_bad(val):
        return math.isnan(val) or math.isinf(val)
        
    if is_bad(vx_body) or is_bad(vy_body) or is_bad(vz_body) or is_bad(yaw_rate_cmd):
        # 如果数据坏了，宁愿让飞机掉下来，也不要让模拟器崩溃！
        return (0.0, 0.0, 0.0, 0.0)

    return (vx_body, vy_body, vz_body, yaw_rate_cmd)