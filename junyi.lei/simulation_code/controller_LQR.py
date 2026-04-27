import numpy as np
import math

def controller(state, target_pos, dt, wind_enabled=False):
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

     # ==============================================================
    # 🌟 数据记录模块 (Data Logging Module)
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 50     # 每50帧写入一次磁盘
        controller.file_path = "data_lqr.csv"
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


    if not hasattr(controller, 'prev_pos'):
        controller.prev_pos = np.array([x, y, z])
        controller.integral_err = np.array([0.0, 0.0, 0.0])

    # ==============================================================
    # LQR 状态构建 / Build LQR State Vector X = [pos_err, vel_err, int_err]^T
    # ==============================================================
    current_pos = np.array([x, y, z])
    pos_error = np.array([tx - x, ty - y, tz - z])
    
    # 估算当前速度误差 (目标速度设为0) / Estimate velocity error (target vel = 0)
    current_vel = (current_pos - controller.prev_pos) / dt
    vel_error = np.array([0.0, 0.0, 0.0]) - current_vel

    # 更新位置积分误差以应对风力 / Update integral error for wind rejection (LQI formulation)
    if wind_enabled:
        controller.integral_err += pos_error * dt
        controller.integral_err = np.clip(controller.integral_err, -2.0, 2.0)
    else:
        controller.integral_err = np.array([0.0, 0.0, 0.0])

    # ==============================================================
    # 离线计算的 LQR 最优增益矩阵 K / Offline Computed LQR Optimal Gain Matrix K
    # 假设使用双积分器模型推导，Q 惩罚位置误差，R 惩罚控制输入
    # K = [K_pos, K_vel, K_int]
    # ==============================================================
    # Note: These values should be tuned offline using lqr() function in control systems library.
    K_pos = np.array([1.5, 1.5, 1.2]) 
    K_vel = np.array([0.5, 0.5, 0.4])
    K_int = np.array([0.8, 0.8, 0.6])

    # 核心 LQR 控制律： u = K * X
    # Core LQR Control Law: u = K * X
    v_world = (K_pos * pos_error) + (K_vel * vel_error) + (K_int * controller.integral_err)

    # 偏航角计算保持简单的 P 控制 / Keep Yaw simple
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 1.5 * eyaw 

    # ==============================================================
    # 坐标系转换 / Frame Transformation
    # ==============================================================
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    controller.prev_pos = current_pos

    return (float(np.clip(vx_body, -1.0, 1.0)), 
            float(np.clip(vy_body, -1.0, 1.0)), 
            float(np.clip(vz_body, -1.0, 1.0)), 
            float(np.clip(yaw_rate_cmd, -1.74, 1.74)))