import numpy as np
import math
import os
import atexit
import matplotlib as mpl

def controller(state, target_pos, dt, wind_enabled=False):
    # ==============================================================
    # 0. 终极防崩溃补丁与底线安全保护 (Anti-crash & Safety)
    # ==============================================================
    if not hasattr(controller, 'hotkey_fixed'):
        if 'k' in mpl.rcParams['keymap.xscale']: mpl.rcParams['keymap.xscale'].remove('k')
        if 'l' in mpl.rcParams['keymap.yscale']: mpl.rcParams['keymap.yscale'].remove('l')
        controller.hotkey_fixed = True

    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    # 1. 状态解析 (Parse state)
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==============================================================
    # 🌟 附加数据记录模块 (Data Logging Module)
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 50     # 每50帧写入一次磁盘
        controller.file_path = "data_dobc.csv"
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
    # 2. DOBC 持久化变量初始化 (Initialize DOBC Persistent States)
    # ==============================================================
    if not hasattr(controller, 'prev_pos'):
        controller.prev_pos = np.array([x, y, z])
        controller.prev_v_cmd = np.array([0.0, 0.0, 0.0]) # 上一帧的期望速度 (Previous commanded velocity)
        controller.d_hat = np.array([0.0, 0.0, 0.0])      # 干扰观测值 (Disturbance estimate)
        controller.prev_pos_err = np.array([0.0, 0.0, 0.0])

    current_pos = np.array([x, y, z])
    target_p = np.array([tx, ty, tz])

    # ==============================================================
    # 🚀 核心模块 A：干扰观测器 (Disturbance Observer - DOBC)
    # ==============================================================
    # 1. 测算实际速度 (Calculate actual velocity)
    v_actual = (current_pos - controller.prev_pos) / dt
    
    # 2. 观测原始干扰：实际速度减去我们上一帧给定的期望速度
    # (Raw disturbance: actual velocity minus what we commanded)
    d_raw = v_actual - controller.prev_v_cmd
    
    # 3. Q-Filter 低通滤波得到干净的干扰估算 (Q-Filter for DOBC)
    # 滤波系数 alpha_dobc: 0.15 左右能很好地过滤掉定位噪声，保留真实的低频风力
    alpha_dobc = 0.15  
    if wind_enabled:
        controller.d_hat = (alpha_dobc * d_raw) + ((1.0 - alpha_dobc) * controller.d_hat)
    else:
        controller.d_hat = np.array([0.0, 0.0, 0.0])

    # ==============================================================
    # 🚀 核心模块 B：基础位置环 PD 控制器 (Baseline PD Controller)
    # 注意：有了 DOBC，我们彻底抛弃了积分项 (I=0)!
    # ==============================================================
    Kp = np.array([1.2, 1.2, 2.0]) 
    Kd = np.array([0.15, 0.15, 0.1])

    pos_err = target_p - current_pos
    d_pos_err = (pos_err - controller.prev_pos_err) / dt
    
    # 计算基础的防错补偿速度 (Calculate baseline PID velocity)
    v_pid = (Kp * pos_err) + (Kd * d_pos_err)

    # ==============================================================
    # 🚀 核心模块 C：控制律前馈补偿 (DOBC Compensation Law)
    # ==============================================================
    # 最终期望速度 = PD计算出的速度 - 估算出的风力干扰
    # (Final Command = Baseline Control - Estimated Disturbance)
    v_cmd_world = v_pid - controller.d_hat

    # 限制世界坐标系下的最大速度，防止系统崩溃
    v_cmd_world = np.clip(v_cmd_world, -1.5, 1.5)

    # ==============================================================
    # 3. 偏航角控制 (Yaw Control)
    # ==============================================================
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 2.0 * eyaw 

    # ==============================================================
    # 4. 坐标系转换与收尾 (Frame Transformation & History)
    # ==============================================================
    vx_w, vy_w, vz_w = v_world = v_cmd_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    # World Frame to Body Frame (世界坐标系转换到机身坐标系)
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    # 更新历史状态 (Update history states)
    controller.prev_pos = current_pos
    controller.prev_pos_err = pos_err
    # 极其重要：记录当前下发的速度，作为下一帧观测器的对比基准
    controller.prev_v_cmd = v_cmd_world 

    # ==============================================================
    # 5. 最终输出限幅与清洗 (Final Output Clamping & Cleaning)
    # ==============================================================
    def clean_and_clamp(val, min_val, max_val):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(np.clip(val, min_val, max_val))

    vx_body = clean_and_clamp(vx_body, -1.0, 1.0)
    vy_body = clean_and_clamp(vy_body, -1.0, 1.0)
    vz_body = clean_and_clamp(vz_body, -1.0, 1.0)
    yaw_rate_cmd = clean_and_clamp(yaw_rate_cmd, -1.74, 1.74)

    if math.isnan(vx_body) or math.isnan(vy_body) or math.isnan(vz_body):
        return (0.0, 0.0, 0.0, 0.0)

    return (vx_body, vy_body, vz_body, yaw_rate_cmd)