import math
import os
import atexit

# 注意：实飞接口只有三个参数，没有 dt 和 wind_enabled
# Note: Real flight interface has only 3 arguments, no dt or wind_enabled
def controller(state, target_pos, timestamp):
    
    # ==============================================================
    # 0. 时间差 dt 的真实计算 (绝对毫秒时间戳 -> 秒)
    # Calculate true dt in seconds from absolute millisecond timestamp
    # ==============================================================
    if not hasattr(controller, "prev_time"):
        controller.prev_time = timestamp
        controller.start_time = timestamp # 记录起始时间用于画图 / Record start time for plotting
        return (0.0, 0.0, 0.0, 0.0)       # 第一帧安全返回 0 / Safe return for the first frame

    # 毫秒转换为秒 / Convert milliseconds to seconds
    dt = (timestamp - controller.prev_time) / 1000.0 
    
    # 保护机制：如果 Vicon 丢帧导致时间没变，跳过计算
    # Safety: Skip if Vicon drops a frame and time hasn't changed
    if dt <= 1e-4:
        return (0.0, 0.0, 0.0, 0.0)

    # 1. 状态解析 / Parse state
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==============================================================
    # 🌟 实飞版数据记录模块 (低延迟，高频率写入)
    # Real Flight Data Logging (Low latency, frequent writes)
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 10     # 阈值设为 10，防撞击强停导致数据丢失 / Small buffer to prevent data loss on crash
        controller.file_path = "real_flight_data.csv"
        
        if not os.path.exists(controller.file_path):
            with open(controller.file_path, 'w') as f:
                f.write("time_sec,target_x,target_y,target_z,target_yaw,x,y,z,yaw\n")
        
        def flush_buffer():
            if hasattr(controller, "buffer") and controller.buffer:
                with open(controller.file_path, 'a') as f:
                    f.writelines(controller.buffer)
        atexit.register(flush_buffer)
        controller.is_initialized_csv = True

    # 记录从起飞开始经过的秒数 / Record elapsed seconds since takeoff
    elapsed_time = (timestamp - controller.start_time) / 1000.0
    record = f"{elapsed_time:.4f},{tx:.4f},{ty:.4f},{tz:.4f},{tyaw:.4f},{x:.4f},{y:.4f},{z:.4f},{yaw:.4f}\n"
    controller.buffer.append(record)

    if len(controller.buffer) >= controller.buffer_limit:
        with open(controller.file_path, 'a') as f:
            f.writelines(controller.buffer)
        controller.buffer.clear()

    # ==============================================================
    # 2. 实飞调参区 (Tuning for Real Tello)
    # 建议：室内无风，积分(I)项极易导致震荡，初期请设为 0
    # Tip: Indoors with no wind, I-term causes oscillations. Keep at 0 initially.
    # ==============================================================
    Kp_xy, Ki_xy, Kd_xy = 0.8, 0.0, 0.3  # P稍微调小，D稍微调大增加阻尼感
    Kp_z,  Ki_z,  Kd_z  = 1.0, 0.0, 0.2  # 高度控制通常可以稍微积极一点
    Kp_yaw, Kd_yaw      = 1.5, 0.1

    if not hasattr(controller, 'integral'):
        controller.integral = [0.0, 0.0, 0.0]
        controller.prev_err = [0.0, 0.0, 0.0, 0.0]

    # ==============================================================
    # 3. 基础 PID 计算 (Basic PID Calculation)
    # ==============================================================
    ex = tx - x
    ey = ty - y
    ez = tz - z
    
    # 偏航角误差处理 [-pi, pi] (Controller practical lab.pdf 提示 Vicon 返回值为 -pi 到 pi)
    eyaw = (tyaw - yaw + math.pi) % (2 * math.pi) - math.pi

    # 微分项 (Derivative)
    dx = (ex - controller.prev_err[0]) / dt
    dy = (ey - controller.prev_err[1]) / dt
    dz = (ez - controller.prev_err[2]) / dt
    dyaw = (eyaw - controller.prev_err[3]) / dt

    # 积分与抗饱和 (Integral with Anti-windup)
    max_int = 1.0
    controller.integral[0] = max(-max_int, min(controller.integral[0] + ex * dt, max_int))
    controller.integral[1] = max(-max_int, min(controller.integral[1] + ey * dt, max_int))
    controller.integral[2] = max(-max_int, min(controller.integral[2] + ez * dt, max_int))

    # 计算全局坐标系输出 / World frame output
    vx_w = Kp_xy * ex + Ki_xy * controller.integral[0] + Kd_xy * dx
    vy_w = Kp_xy * ey + Ki_xy * controller.integral[1] + Kd_xy * dy
    vz_w = Kp_z  * ez + Ki_z  * controller.integral[2] + Kd_z  * dz
    yaw_cmd = Kp_yaw * eyaw + Kd_yaw * dyaw

    # ==============================================================
    # 4. 坐标系转换 (World -> Body)
    # 实验室坐标系约定：x 向前, y 向左, z 向上 / x forward, y left, z up
    # ==============================================================
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    vx_b = vx_w * cos_y + vy_w * sin_y
    vy_b = -vx_w * sin_y + vy_w * cos_y
    vz_b = vz_w

    # 更新历史记录用于下一帧计算 / Update history for next frame
    controller.prev_err = [ex, ey, ez, eyaw]
    controller.prev_time = timestamp

    # ==============================================================
    # 5. 保守的安全限幅 (Conservative Safety Clamping)
    # 实飞非常重要！防止满油门冲向墙壁 / Critical for real flight to avoid crashing
    # ==============================================================
    def clean_and_clamp(val, limit):
        if math.isnan(val) or math.isinf(val): return 0.0
        return float(max(-limit, min(val, limit)))

    # 室内测试初期，强烈建议速度限制在 0.6 m/s 左右
    # Highly recommend limiting speed to ~0.6 m/s for early indoor tests
    SAFE_SPEED = 0.6 
    SAFE_YAW_RATE = 1.0

    vx_b = clean_and_clamp(vx_b, SAFE_SPEED)
    vy_b = clean_and_clamp(vy_b, SAFE_SPEED)
    vz_b = clean_and_clamp(vz_b, SAFE_SPEED)
    yaw_cmd = clean_and_clamp(yaw_cmd, SAFE_YAW_RATE)

    return (vx_b, vy_b, vz_b, yaw_cmd)