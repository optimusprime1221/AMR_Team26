import numpy as np
import math


def controller(state, target_pos, dt, wind_enabled=False):

    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    # 1. Parse state
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==============================================================
    # 🌟 Additional Data Logging Module
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 50     # Write to disk every 50 frames
        controller.file_path = "data_dobc.csv"
        controller.sim_time = 0.0
        
        # Optimization: no longer check if file exists, just open in 'w' mode.
        # This will automatically clear the old data.csv and write a brand new header.
        with open(controller.file_path, 'w') as f:
            f.write("time,target_x,target_y,target_z,target_yaw,x,y,z,yaw\n")
        
        # Flush remaining buffer data on exit
        def flush_buffer():
            if hasattr(controller, "buffer") and controller.buffer:
                with open(controller.file_path, 'a') as f:
                    f.writelines(controller.buffer)
        
        import atexit
        atexit.register(flush_buffer)
        controller.is_initialized_csv = True

    # Accumulate time and record
    controller.sim_time += dt
    # Use string formatting to keep data tidy
    record = f"{controller.sim_time:.4f},{tx:.4f},{ty:.4f},{tz:.4f},{tyaw:.4f},{x:.4f},{y:.4f},{z:.4f},{yaw:.4f}\n"
    controller.buffer.append(record)

    # When threshold is reached, append-write to disk (use 'a' mode)
    if len(controller.buffer) >= controller.buffer_limit:
        with open(controller.file_path, 'a') as f:
            f.writelines(controller.buffer)
        controller.buffer.clear()
    # ==============================================================
    # 2. Initialize DOBC Persistent States
    # ==============================================================
    if not hasattr(controller, 'prev_pos'):
        controller.prev_pos = np.array([x, y, z])
        controller.prev_v_cmd = np.array([0.0, 0.0, 0.0]) # Previous commanded velocity
        controller.d_hat = np.array([0.0, 0.0, 0.0])      # Disturbance estimate
        controller.prev_pos_err = np.array([0.0, 0.0, 0.0])

    current_pos = np.array([x, y, z])
    target_p = np.array([tx, ty, tz])

    # ==============================================================
    # 🚀 Core Module A: Disturbance Observer (DOBC)
    # ==============================================================
    # 1. Calculate actual velocity
    v_actual = (current_pos - controller.prev_pos) / dt
    
    # 2. Raw disturbance: actual velocity minus what we commanded
    d_raw = v_actual - controller.prev_v_cmd
    
    # 3. Q-Filter for DOBC to get clean disturbance estimate
    # Filter coefficient alpha_dobc: around 0.15 effectively filters out positioning noise while retaining real low-frequency wind
    alpha_dobc = 0.15  
    if wind_enabled:
        controller.d_hat = (alpha_dobc * d_raw) + ((1.0 - alpha_dobc) * controller.d_hat)
    else:
        controller.d_hat = np.array([0.0, 0.0, 0.0])

    # ==============================================================
    # 🚀 Core Module B: Baseline Position Loop PD Controller
    # Note: With DOBC, we completely discard the integral term (I=0)!
    # ==============================================================
    Kp = np.array([1.2, 1.2, 2.0]) 
    Kd = np.array([0.15, 0.15, 0.1])

    pos_err = target_p - current_pos
    d_pos_err = (pos_err - controller.prev_pos_err) / dt
    
    # Compute baseline PID velocity
    v_pid = (Kp * pos_err) + (Kd * d_pos_err)

    # ==============================================================
    # 🚀 Core Module C: DOBC Compensation Law
    # ==============================================================
    # Final commanded velocity = Baseline Control - Estimated Disturbance
    v_cmd_world = v_pid - controller.d_hat

    # Limit max velocity in world frame to prevent system crash
    v_cmd_world = np.clip(v_cmd_world, -1.5, 1.5)

    # ==============================================================
    # 3. Yaw Control
    # ==============================================================
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 2.0 * eyaw 

    # ==============================================================
    # 4. Frame Transformation & History
    # ==============================================================
    vx_w, vy_w, vz_w = v_world = v_cmd_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    # World Frame to Body Frame
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    # Update history states
    controller.prev_pos = current_pos
    controller.prev_pos_err = pos_err
    # Extremely important: record the current commanded velocity as the reference for the next frame's observer
    controller.prev_v_cmd = v_cmd_world 

    # ==============================================================
    # 5. Final Output Clamping & Cleaning
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