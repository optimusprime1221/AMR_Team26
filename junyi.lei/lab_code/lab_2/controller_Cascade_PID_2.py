import numpy as np
import math

def controller(state, target_pos, dt, wind_enabled=False):
   
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    # 1. State parsing
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==============================================================
    # 🌟 Data Logging Module
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 50     # Write to disk every 50 frames
        controller.file_path = "data_caspid2.csv"
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
    # 2. Core Tuning Area - incorporating your provided parameters
    # ==============================================================
    # [Outer loop: position -> desired velocity] (controls speed to target, resists steady wind)
    Kp_outer = np.array([1.1, 1.1, 2.0]) 
    Ki_outer = np.array([0.035, 0.035, 0.0]) 
    Kd_outer = np.array([0.1, 0.1, 0.1]) 

    # [Inner loop: desired velocity -> velocity command compensation] (controls low-level response)
    Kp_inner = np.array([0.8, 0.8, 0.8])
    Ki_inner = np.array([0.0, 0.0, 0.0]) # Inner loop integral set to 0 to avoid "fighting" between loops
    Kd_inner = np.array([0.05, 0.05, 0.05]) # Keeps your D term, but used with the filter below

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

    # ---------- Added: reset on target change ----------
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
    # 4. Low-pass velocity filter (magic tool to suppress oscillations)
    # Because you keep the inner loop Kd (0.05), without filtering the drone will twitch violently
    # ==============================================================
    alpha = 0.3 # Filter coefficient
    current_pos = np.array([x, y, z])
    raw_vel = (current_pos - controller.prev_pos) / dt
    controller.filtered_vel = (alpha * raw_vel) + ((1.0 - alpha) * controller.filtered_vel)

    # ==============================================================
    # 5. [Outer loop: Position loop]
    # ==============================================================
    pos_error = np.array([tx - x, ty - y, tz - z])
    d_pos_err = (pos_error - controller.prev_pos_err) / dt

    # Outer loop integral with anti-windup
    controller.pos_integral += pos_error * dt
    controller.pos_integral = np.clip(controller.pos_integral, -0.2, 0.2)

    # Compute desired velocity
    target_vel = (Kp_outer * pos_error) + (Ki_outer * controller.pos_integral) + (Kd_outer * d_pos_err)
    target_vel = np.clip(target_vel, -1.5, 1.5) # Limit max desired velocity

    # ==============================================================
    # 6. [Inner loop: Velocity loop]
    # ==============================================================
    vel_error = target_vel - controller.filtered_vel
    d_vel_err = (vel_error - controller.prev_vel_err) / dt

    controller.vel_integral += vel_error * dt
    controller.vel_integral = np.clip(controller.vel_integral, -0.5, 0.5)

    # Feedforward (target_vel directly) + compensation (PID)
    v_world = target_vel + (Kp_inner * vel_error) + (Ki_inner * controller.vel_integral) + (Kd_inner * d_vel_err)

    # ==============================================================
    # 7. Yaw Control
    # ==============================================================
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 2.0 * eyaw 

    # ==============================================================
    # 8. Coordinate transformation and finalization
    # ==============================================================
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    # World to Body
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    # Update history states
    controller.prev_pos = current_pos
    controller.prev_pos_err = pos_error
    controller.prev_vel_err = vel_error

    # ==============================================================
    # 9. Final output clamping and cleaning (crash protection)
    # ==============================================================
    def clean_and_clamp(val, min_val, max_val):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(np.clip(val, min_val, max_val))

    vx_body = clean_and_clamp(vx_body, -1.0, 1.0)
    vy_body = clean_and_clamp(vy_body, -1.0, 1.0)
    vz_body = clean_and_clamp(vz_body, -1.0, 1.0)
    yaw_rate_cmd = clean_and_clamp(yaw_rate_cmd, -1.74, 1.74)

    # Absolutely prevent bad data from being sent to the physics engine
    if math.isnan(vx_body) or math.isnan(vy_body) or math.isnan(vz_body):
        return (0.0, 0.0, 0.0, 0.0)

    return (vx_body, vy_body, vz_body, yaw_rate_cmd)