wind_flag = True
# Implement a controller

import math
import os
import atexit

def controller(state, target_pos, dt, wind_enabled=False):
    """
    Complete PID Controller (with Data Logging & Crash Protection)
    """
    # ==============================================================
    # 0. Extreme Safety Checks
    # Prevent division by zero or invalid state inputs
    # ==============================================================
    if dt <= 1e-4:
        return (0.0, 0.0, 0.0, 0.0)
    
    for val in state:
        if math.isnan(val) or math.isinf(val):
            return (0.0, 0.0, 0.0, 0.0)

    # 1. State & Target Parsing
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # After state parsing, add target switching detection and reset logic
    # Assume state and target_pos have already been parsed
    if not hasattr(controller, 'last_target'):
        controller.last_target = None

    # Detect if target has changed
    if controller.last_target != target_pos:
        # Reset integral terms (XY and Z)
        if hasattr(controller, 'integral_error'):
            controller.integral_error = [0.0, 0.0, 0.0]
        else:
            controller.integral_error = [0.0, 0.0, 0.0]  # Assign directly if not yet initialized
        # Reset derivative history (including yaw)
        if hasattr(controller, 'prev_error'):
            controller.prev_error = [0.0, 0.0, 0.0, 0.0]
        else:
            controller.prev_error = [0.0, 0.0, 0.0, 0.0]
        # Update recorded target
        controller.last_target = target_pos

    # ==============================================================
    # 🌟 Data Logging Module
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 50     # Write to disk every 50 frames
        controller.file_path = "data_pid.csv"
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
    # 🚀 Core PID Parameters 
    # [Tuning Area]
    # ==============================================================
    # Kp: Proportional, speed towards target
    # Ki: Integral, to fight steady wind
    # Kd: Derivative, acts as a brake, prevents overshoot
    Kp_xy = 1.1
    Ki_xy = 0.035
    Kd_xy = 0.1

    Kp_z = 2.0
    Ki_z = 0.0
    Kd_z = 0.1

    Kp_yaw = 2.0
    Kd_yaw = 0.1
    # ==============================================================
    # Initialize Persistent Variables
    # ==============================================================
    if not hasattr(controller, 'integral_error'):
        controller.integral_error = [0.0, 0.0, 0.0] # [x, y, z]
    if not hasattr(controller, 'prev_error'):
        controller.prev_error = [0.0, 0.0, 0.0, 0.0] # [x, y, z, yaw]

    # ==============================================================
    # Error Calculation in World Frame
    # ==============================================================
    ex = tx - x
    ey = ty - y
    ez = tz - z
    
    # Wrap yaw error to [-pi, pi]
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi

    # ==============================================================
    # Calculate P, I, D terms
    # ==============================================================
    # Derivative
    dx = (ex - controller.prev_error[0]) / dt
    dy = (ey - controller.prev_error[1]) / dt
    dz = (ez - controller.prev_error[2]) / dt
    dyaw = (eyaw - controller.prev_error[3]) / dt

    # Integral & Anti-Windup
    controller.integral_error[0] += ex * dt
    controller.integral_error[1] += ey * dt
    controller.integral_error[2] += ez * dt

    max_integral = 0.2 # Prevent integral windup
    for i in range(3):
        controller.integral_error[i] = max(-max_integral, min(controller.integral_error[i], max_integral))

    ix, iy, iz = controller.integral_error

    # Compute World Frame Velocity
    vx_world = (Kp_xy * ex) + (Ki_xy * ix) + (Kd_xy * dx)
    vy_world = (Kp_xy * ey) + (Ki_xy * iy) + (Kd_xy * dy)
    vz_world = (Kp_z * ez)  + (Ki_z * iz)  + (Kd_z * dz)
    yaw_rate_cmd = (Kp_yaw * eyaw) + (Kd_yaw * dyaw)

    # ==============================================================
    # Coordinate Transformation: World Frame -> Body Frame
    # ==============================================================
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    vx_body = vx_world * cos_yaw + vy_world * sin_yaw
    vy_body = -vx_world * sin_yaw + vy_world * cos_yaw
    vz_body = vz_world 

    # Save error for next frame
    controller.prev_error = [ex, ey, ez, eyaw]

    # ==============================================================
    # Final Output Clamping & Cleaning
    # Absolutely prevent NaN sent to simulator
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
        # If data is corrupted, rather let the drone fall than crash the simulator!
        return (0.0, 0.0, 0.0, 0.0)

    return (vx_body, vy_body, vz_body, yaw_rate_cmd)