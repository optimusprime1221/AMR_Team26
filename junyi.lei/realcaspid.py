import math
import os
import atexit

def controller(state, target_pos, timestamp):
    # ========================================================================
    # 🌟🌟🌟 核心调参区 (CORE TUNING AREA) 🌟🌟🌟
    # 请在实验室测试时，只修改这里的参数！
    # Please only modify parameters here during your lab session!
    # ========================================================================
    
    # --- 1. 外环：位置控制 (Outer Loop: Position -> Target Velocity) ---
    # Kp: 决定靠近目标的速度 / Determines speed towards target
    # Ki: 消除悬停时的静态误差 / Eliminates steady-state error
    # Kd: 提供位置阻尼，防止冲过头 / Provides positional damping, prevents overshoot
    Kp_pos = [1.0, 1.0, 1.2]  # [x, y, z]
    Ki_pos = [0.1, 0.1, 0.1]  # 室内无风建议设得很小或 0 / Keep small or 0 indoors
    Kd_pos = [0.2, 0.2, 0.1]

    # --- 2. 内环：速度补偿 (Inner Loop: Velocity -> Output Command) ---
    # 决定 Tello 跟踪期望速度的紧密程度 / Determines how tightly Tello tracks desired velocity
    Kp_vel = [0.5, 0.5, 0.5]  # [x, y, z] 

    # --- 3. 偏航角控制 (Yaw Control - Single Loop) ---
    Kp_yaw = 1.5
    Kd_yaw = 0.1

    # --- 4. 信号处理与安全限制 (Signal Processing & Safety Limits) ---
    # 速度低通滤波系数 (0.01~1.0)：越小越平滑但延迟高，越大越灵敏但噪声大
    # Velocity Low-pass filter alpha: smaller = smoother but delayed, larger = responsive but noisy
    VEL_FILTER_ALPHA = 0.2 
    
    # 绝对安全限幅 (实飞保命参数) / Absolute safety limits for real flight
    MAX_SPEED = 0.6     # 最高允许速度 (m/s) / Max allowed speed
    MAX_YAW_RATE = 1.0  # 最高偏航角速度 (rad/s) / Max yaw rate

    # ========================================================================
    # -------------------------- 以下为底层算法实现 --------------------------
    # ========================================================================

    # --------------------------------------------------------------
    # 0. 时间计算与安全拦截 (Time Calculation & Safety)
    # --------------------------------------------------------------
    if not hasattr(controller, "prev_time"):
        controller.prev_time = timestamp
        controller.start_time = timestamp
        return (0.0, 0.0, 0.0, 0.0)

    dt = (timestamp - controller.prev_time) / 1000.0 
    if dt <= 1e-4: # 防止 Vicon 丢帧导致 dt=0 / Prevent dt=0 if Vicon drops frame
        return (0.0, 0.0, 0.0, 0.0)

    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # --------------------------------------------------------------
    # 1. 实验室数据记录模块 (Lab Data Logger)
    # --------------------------------------------------------------
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 10  # 极小缓冲池防止死机丢失 / Small buffer prevents data loss on crash
        controller.file_path = "cascade_flight_data.csv"
        
        if not os.path.exists(controller.file_path):
            with open(controller.file_path, 'w') as f:
                f.write("time_sec,tx,ty,tz,tyaw,x,y,z,yaw,vx_cmd,vy_cmd\n")
        
        def flush_buffer():
            if hasattr(controller, "buffer") and controller.buffer:
                with open(controller.file_path, 'a') as f:
                    f.writelines(controller.buffer)
        atexit.register(flush_buffer)
        controller.is_initialized_csv = True

    # --------------------------------------------------------------
    # 2. 状态持久化变量初始化 (Initialize Persistent States)
    # --------------------------------------------------------------
    if not hasattr(controller, 'prev_pos'):
        controller.prev_pos = [x, y, z]
        controller.filtered_vel = [0.0, 0.0, 0.0]
        controller.pos_integral = [0.0, 0.0, 0.0]
        controller.prev_pos_err = [0.0, 0.0, 0.0]
        controller.prev_yaw_err = 0.0

    # --------------------------------------------------------------
    # 3. 速度低通滤波 (Velocity Estimation & Low-Pass Filter)
    # 极其重要：平滑 Vicon 定位的毫米级跳动噪声
    # Critical: Smooths out millimeter-level jitter from Vicon
    # --------------------------------------------------------------
    current_pos = [x, y, z]
    raw_vel = [(current_pos[i] - controller.prev_pos[i]) / dt for i in range(3)]
    
    for i in range(3):
        controller.filtered_vel[i] = (VEL_FILTER_ALPHA * raw_vel[i]) + ((1.0 - VEL_FILTER_ALPHA) * controller.filtered_vel[i])

    # --------------------------------------------------------------
    # 4. 外环控制：计算期望速度 (Outer Loop: Calculate Target Velocity)
    # --------------------------------------------------------------
    pos_err = [tx - x, ty - y, tz - z]
    d_pos_err = [(pos_err[i] - controller.prev_pos_err[i]) / dt for i in range(3)]

    # 积分与抗饱和 (限制积分最大影响为 0.5 m/s)
    # Integral with anti-windup (limited to 0.5 m/s impact)
    target_vel = [0.0, 0.0, 0.0]
    for i in range(3):
        controller.pos_integral[i] += pos_err[i] * dt
        controller.pos_integral[i] = max(-0.5, min(controller.pos_integral[i], 0.5))
        
        target_vel[i] = (Kp_pos[i] * pos_err[i]) + (Ki_pos[i] * controller.pos_integral[i]) + (Kd_pos[i] * d_pos_err[i])
        # 限制期望速度不会超过物理极限
        target_vel[i] = max(-MAX_SPEED, min(target_vel[i], MAX_SPEED))

    # --------------------------------------------------------------
    # 5. 内环控制：前馈 + 比例补偿 (Inner Loop: Feedforward + P Comp)
    # --------------------------------------------------------------
    v_world = [0.0, 0.0, 0.0]
    for i in range(3):
        vel_err = target_vel[i] - controller.filtered_vel[i]
        # 最终输出 = 期望速度(前馈) + 速度误差反馈补偿
        v_world[i] = target_vel[i] + (Kp_vel[i] * vel_err)

    # --------------------------------------------------------------
    # 6. 偏航角控制 (Yaw Control)
    # --------------------------------------------------------------
    eyaw = (tyaw - yaw + math.pi) % (2 * math.pi) - math.pi
    dyaw = (eyaw - controller.prev_yaw_err) / dt
    yaw_cmd = (Kp_yaw * eyaw) + (Kd_yaw * dyaw)

    # --------------------------------------------------------------
    # 7. 坐标转换与历史更新 (Frame Transformation & History Update)
    # --------------------------------------------------------------
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    vx_b = v_world[0] * cos_y + v_world[1] * sin_y
    vy_b = -v_world[0] * sin_y + v_world[1] * cos_y
    vz_b = v_world[2]

    controller.prev_pos = current_pos
    controller.prev_pos_err = pos_err
    controller.prev_yaw_err = eyaw
    controller.prev_time = timestamp

    # --------------------------------------------------------------
    # 8. 安全清洗与限幅 (Safety Clamping)
    # --------------------------------------------------------------
    def clean_and_clamp(val, limit):
        if math.isnan(val) or math.isinf(val): return 0.0
        return float(max(-limit, min(val, limit)))

    vx_b = clean_and_clamp(vx_b, MAX_SPEED)
    vy_b = clean_and_clamp(vy_b, MAX_SPEED)
    vz_b = clean_and_clamp(vz_b, MAX_SPEED)
    yaw_cmd = clean_and_clamp(yaw_cmd, MAX_YAW_RATE)

    # 记录日志 (加上了最终输出速度，方便事后画图分析内环表现)
    # Log data including final velocity commands for analysis
    elapsed_time = (timestamp - controller.start_time) / 1000.0
    record = f"{elapsed_time:.4f},{tx:.4f},{ty:.4f},{tz:.4f},{tyaw:.4f},{x:.4f},{y:.4f},{z:.4f},{yaw:.4f},{vx_b:.4f},{vy_b:.4f}\n"
    controller.buffer.append(record)
    if len(controller.buffer) >= controller.buffer_limit:
        with open(controller.file_path, 'a') as f: f.writelines(controller.buffer)
        controller.buffer.clear()

    return (vx_b, vy_b, vz_b, yaw_cmd)