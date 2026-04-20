import numpy as np
import math
import os
import atexit

def controller(state, target_pos, timestamp,_):
    # ==============================================================
    # 0. Time Calculation & Baseline Safety Protection
    # Real flight uses absolute timestamp (milliseconds) instead of dt
    # ==============================================================
    if not hasattr(controller, "prev_time"):
        controller.prev_time = timestamp
        controller.start_time = timestamp
        return (0.0, 0.0, 0.0, 0.0)

    # Convert milliseconds to seconds
    dt = (timestamp - controller.prev_time) / 1000.0 
    
    # Safety check: Prevent division by zero if Vicon system drops a frame
    if dt <= 1e-4:
        return (0.0, 0.0, 0.0, 0.0)

    # 1. Parse state
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==============================================================
    # 2. Core Tuning Area & Safety Limits (Real Flight Config)
    # ==============================================================
    # Absolute safety limits to prevent crashing into walls
    MAX_SPEED = 0.6    # Maximum drone speed (m/s)
    MAX_YAW_RATE = 1.0 # Maximum yaw rotation (rad/s)

    # [Outer Loop: Position -> Target Velocity] 
    Kp_outer = np.array([1.2, 1.2, 1.0]) 
    Ki_outer = np.array([0.1, 0.1, 0.1]) # Keep I-term small for indoor environments
    Kd_outer = np.array([0.1, 0.1, 0.1]) 

    # [Inner Loop: Target Velocity -> Velocity Command Compensation] 
    Kp_inner = np.array([0.8, 0.8, 0.8])
    Ki_inner = np.array([0.0, 0.0, 0.0]) # Keep at 0 to avoid fighting outer loop
    Kd_inner = np.array([0.05, 0.05, 0.05]) # Inner D term is kept, but heavily relies on filter

    # ==============================================================
    # 3. Initialize persistent variables
    # ==============================================================
    if not hasattr(controller, 'prev_pos'):
        controller.prev_pos = np.array([x, y, z])
        controller.filtered_vel = np.array([0.0, 0.0, 0.0])
        controller.pos_integral = np.array([0.0, 0.0, 0.0])
        controller.vel_integral = np.array([0.0, 0.0, 0.0])
        controller.prev_pos_err = np.array([0.0, 0.0, 0.0])
        controller.prev_vel_err = np.array([0.0, 0.0, 0.0])

    # ==============================================================
    # 4. Data Logging Module (Real Flight Config)
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 10     # Smaller buffer (10) for real flight safety
        controller.file_path = "real_cascade_numpy_data.csv"
        
        if not os.path.exists(controller.file_path):
            with open(controller.file_path, 'w') as f:
                f.write("time_sec,target_x,target_y,target_z,target_yaw,x,y,z,yaw,vx_cmd,vy_cmd,vz_cmd\n")
        
        def flush_buffer():
            if hasattr(controller, "buffer") and controller.buffer:
                with open(controller.file_path, 'a') as f:
                    f.writelines(controller.buffer)
        atexit.register(flush_buffer)
        controller.is_initialized_csv = True

    # ==============================================================
    # 5. Velocity Low-Pass Filter (Essential for Vicon Noise)
    # ==============================================================
    alpha = 0.3 # Filtering coefficient to suppress Vicon jitter
    current_pos = np.array([x, y, z])
    raw_vel = (current_pos - controller.prev_pos) / dt
    controller.filtered_vel = (alpha * raw_vel) + ((1.0 - alpha) * controller.filtered_vel)

    # ==============================================================
    # 6. [Outer Loop: Position Loop] 
    # ==============================================================
    pos_error = np.array([tx - x, ty - y, tz - z])
    d_pos_err = (pos_error - controller.prev_pos_err) / dt

    # Outer loop integral and anti-windup (Restricted to +/- 0.5 m/s impact)
    controller.pos_integral += pos_error * dt
    controller.pos_integral = np.clip(controller.pos_integral, -0.5, 0.5)

    # Calculate target velocity
    target_vel = (Kp_outer * pos_error) + (Ki_outer * controller.pos_integral) + (Kd_outer * d_pos_err)
    target_vel = np.clip(target_vel, -MAX_SPEED, MAX_SPEED) 

    # ==============================================================
    # 7. [Inner Loop: Velocity Loop]
    # ==============================================================
    vel_error = target_vel - controller.filtered_vel
    d_vel_err = (vel_error - controller.prev_vel_err) / dt

    controller.vel_integral += vel_error * dt
    controller.vel_integral = np.clip(controller.vel_integral, -0.3, 0.3)

    # Feedforward + PID Compensation
    v_world = target_vel + (Kp_inner * vel_error) + (Ki_inner * controller.vel_integral) + (Kd_inner * d_vel_err)

    # ==============================================================
    # 8. [Yaw Control] 
    # ==============================================================
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 1.5 * eyaw 

    # ==============================================================
    # 9. [Coordinate Transformation & History Update]
    # ==============================================================
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    # World Frame to Body Frame
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    # Update history states for the next frame
    controller.prev_pos = current_pos
    controller.prev_pos_err = pos_error
    controller.prev_vel_err = vel_error
    controller.prev_time = timestamp

    # ==============================================================
    # 10. Final Output Clamping & Logging
    # ==============================================================
    def clean_and_clamp(val, min_val, max_val):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(np.clip(val, min_val, max_val))

    vx_body = clean_and_clamp(vx_body, -MAX_SPEED, MAX_SPEED)
    vy_body = clean_and_clamp(vy_body, -MAX_SPEED, MAX_SPEED)
    vz_body = clean_and_clamp(vz_body, -MAX_SPEED, MAX_SPEED)
    yaw_rate_cmd = clean_and_clamp(yaw_rate_cmd, -MAX_YAW_RATE, MAX_YAW_RATE)

    if math.isnan(vx_body) or math.isnan(vy_body) or math.isnan(vz_body):
        return (0.0, 0.0, 0.0, 0.0)

    # Accumulate time and record
    elapsed_time = (timestamp - controller.start_time) / 1000.0
    record = f"{elapsed_time:.4f},{tx:.4f},{ty:.4f},{tz:.4f},{tyaw:.4f},{x:.4f},{y:.4f},{z:.4f},{yaw:.4f},{vx_body:.4f},{vy_body:.4f},{vz_body:.4f}\n"
    controller.buffer.append(record)

    if len(controller.buffer) >= controller.buffer_limit:
        with open(controller.file_path, 'a') as f:
            f.writelines(controller.buffer)
        controller.buffer.clear()

    print(vx_body,vy_body,vz_body,yaw_rate_cmd)
    return (vx_body, vy_body, vz_body, yaw_rate_cmd)