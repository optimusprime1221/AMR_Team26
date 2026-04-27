import numpy as np
import math

def controller(state, target_pos, dt, wind_enabled=False):
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

     # ==============================================================
    # 🌟 Data Logging Module
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 50     # Write to disk every 50 frames
        controller.file_path = "data_lqr.csv"
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


    if not hasattr(controller, 'prev_pos'):
        controller.prev_pos = np.array([x, y, z])
        controller.integral_err = np.array([0.0, 0.0, 0.0])

    # ==============================================================
    # Build LQR State Vector X = [pos_err, vel_err, int_err]^T
    # ==============================================================
    current_pos = np.array([x, y, z])
    pos_error = np.array([tx - x, ty - y, tz - z])
    
    # Estimate current velocity error (target vel = 0)
    current_vel = (current_pos - controller.prev_pos) / dt
    vel_error = np.array([0.0, 0.0, 0.0]) - current_vel

    # Update position integral error for wind rejection (LQI formulation)
    if wind_enabled:
        controller.integral_err += pos_error * dt
        controller.integral_err = np.clip(controller.integral_err, -2.0, 2.0)
    else:
        controller.integral_err = np.array([0.0, 0.0, 0.0])

    # ==============================================================
    # Offline Computed LQR Optimal Gain Matrix K
    # Assumes a double integrator model, Q penalizes position error, R penalizes control input
    # K = [K_pos, K_vel, K_int]
    # ==============================================================
    # Note: These values should be tuned offline using lqr() function in control systems library.
    K_pos = np.array([1.5, 1.5, 1.2]) 
    K_vel = np.array([0.5, 0.5, 0.4])
    K_int = np.array([0.8, 0.8, 0.6])

    # Core LQR Control Law: u = K * X
    v_world = (K_pos * pos_error) + (K_vel * vel_error) + (K_int * controller.integral_err)

    # Keep Yaw simple
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 1.5 * eyaw 

    # ==============================================================
    # Frame Transformation
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