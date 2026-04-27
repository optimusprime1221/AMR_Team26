import numpy as np
import math

def controller(state, target_pos, dt, wind_enabled=False):
   
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    # 1. 状态解析
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==============================================================
    # 🌟 数据记录模块 (Data Logging Module)
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 50     # 每50帧写入一次磁盘
        controller.file_path = "data_caspid.csv"
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
    # 2. 核心调参区 (Tuning Area) - 结合了你提供的参数
    # ==============================================================
    # [外环：位置 -> 期望速度] (控制到达目标的快慢，抵抗稳态风)
    Kp_outer = np.array([1.1, 1.1, 2.0]) 
    Ki_outer = np.array([0.035, 0.035, 0.0]) 
    Kd_outer = np.array([0.1, 0.1, 0.1]) # 外环阻尼，防止冲过头

    # [内环：期望速度 -> 速度指令补偿] (控制底层响应)
    Kp_inner = np.array([0.8, 0.8, 0.8])
    Ki_inner = np.array([0.0, 0.0, 0.0]) # 内环积分设为0，避免内外环“打架”
    Kd_inner = np.array([0.05, 0.05, 0.05]) # 保留了你的 D 项，但配合了下方的滤波器使用

    # ==============================================================
    # 3. 初始化持久化变量
    # ==============================================================
    if not hasattr(controller, 'prev_pos'):
        controller.prev_pos = np.array([x, y, z])
        controller.filtered_vel = np.array([0.0, 0.0, 0.0])
        controller.pos_integral = np.array([0.0, 0.0, 0.0])
        controller.vel_integral = np.array([0.0, 0.0, 0.0])
        controller.prev_pos_err = np.array([0.0, 0.0, 0.0])
        controller.prev_vel_err = np.array([0.0, 0.0, 0.0])

    # ---------- 新增：目标切换时重置 ----------
    if not hasattr(controller, 'last_target'):
        controller.last_target = None
    
    if controller.last_target != target_pos:
        controller.prev_pos = np.array([x, y, z])
        controller.filtered_vel = np.array([0.0, 0.0, 0.0])
        controller.pos_integral = np.array([0.0, 0.0, 0.0])
        controller.vel_integral = np.array([0.0, 0.0, 0.0])
        controller.prev_pos_err = np.array([0.0, 0.0, 0.0])
        controller.prev_vel_err = np.array([0.0, 0.0, 0.0])
        controller.last_target = target_pos

    # ==============================================================
    # 4. 速度低通滤波 (压制震荡的神器)
    # 因为你保留了内环的 Kd (0.05)，不加滤波无人机会剧烈抽搐
    # ==============================================================
    alpha = 0.3 # 滤波系数
    current_pos = np.array([x, y, z])
    raw_vel = (current_pos - controller.prev_pos) / dt
    controller.filtered_vel = (alpha * raw_vel) + ((1.0 - alpha) * controller.filtered_vel)

    # ==============================================================
    # 5. [外环：位置环] Outer Loop
    # ==============================================================
    pos_error = np.array([tx - x, ty - y, tz - z])
    d_pos_err = (pos_error - controller.prev_pos_err) / dt

    # 外环积分与抗饱和
    controller.pos_integral += pos_error * dt
    controller.pos_integral = np.clip(controller.pos_integral, -0.2, 0.2)

    # 计算期望速度
    target_vel = (Kp_outer * pos_error) + (Ki_outer * controller.pos_integral) + (Kd_outer * d_pos_err)
    target_vel = np.clip(target_vel, -1.5, 1.5) # 限制最高期望速度

    # ==============================================================
    # 6. [内环：速度环] Inner Loop
    # ==============================================================
    vel_error = target_vel - controller.filtered_vel
    d_vel_err = (vel_error - controller.prev_vel_err) / dt

    controller.vel_integral += vel_error * dt
    controller.vel_integral = np.clip(controller.vel_integral, -0.5, 0.5)

    # 前馈 (直接给 target_vel) + 补偿 (PID)
    v_world = target_vel + (Kp_inner * vel_error) + (Ki_inner * controller.vel_integral) + (Kd_inner * d_vel_err)

    # ==============================================================
    # 7. [偏航角控制] Yaw Control
    # ==============================================================
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 2.0 * eyaw 

    # ==============================================================
    # 8. [坐标系转换与收尾]
    # ==============================================================
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    # World to Body
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    # 更新历史状态
    controller.prev_pos = current_pos
    controller.prev_pos_err = pos_error
    controller.prev_vel_err = vel_error

    # ==============================================================
    # 9. 最终输出限幅与清洗 (防崩溃机制)
    # ==============================================================
    def clean_and_clamp(val, min_val, max_val):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(np.clip(val, min_val, max_val))

    vx_body = clean_and_clamp(vx_body, -1.0, 1.0)
    vy_body = clean_and_clamp(vy_body, -1.0, 1.0)
    vz_body = clean_and_clamp(vz_body, -1.0, 1.0)
    yaw_rate_cmd = clean_and_clamp(yaw_rate_cmd, -1.74, 1.74)

    # 彻底杜绝脏数据传给物理引擎
    if math.isnan(vx_body) or math.isnan(vy_body) or math.isnan(vz_body):
        return (0.0, 0.0, 0.0, 0.0)

    return (vx_body, vy_body, vz_body, yaw_rate_cmd)