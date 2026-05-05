import numpy as np
import math

def controller(state, target_pos, dt, wind_enabled=False):
    """
    Position and yaw controller for a quadrotor.

    Parameters
    ----------
    state : tuple (x, y, z, roll, pitch, yaw)
        Current state of the drone.
    target_pos : tuple (tx, ty, tz, tyaw)
        Desired position and yaw angle.
    dt : float
        Time step in seconds.
    wind_enabled : bool, optional
        Flag indicating whether wind disturbances are active (not used internally).
    """
    # Guard against extremely small time steps to avoid division by zero
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    # 1. State parsing
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==============================================================
    # Data logging module
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        # Buffer for batched writing to CSV
        controller.buffer = []
        controller.buffer_limit = 50     # Flush buffer after collecting this many lines
        controller.file_path = "data_caspid.csv"
        controller.sim_time = 0.0        # Accumulated simulation time
        # Write header line
        with open(controller.file_path, 'w') as f:
            f.write("time,target_x,target_y,target_z,target_yaw,x,y,z,yaw\n")
        
        def flush_buffer():
            """Write buffered records to CSV file if any exist."""
            if hasattr(controller, "buffer") and controller.buffer:
                with open(controller.file_path, 'a') as f:
                    f.writelines(controller.buffer)
        
        # Register flush at program exit to avoid data loss
        import atexit
        atexit.register(flush_buffer)
        controller.is_initialized_csv = True

    # Accumulate time and record current state & target
    controller.sim_time += dt
    record = f"{controller.sim_time:.4f},{tx:.4f},{ty:.4f},{tz:.4f},{tyaw:.4f},{x:.4f},{y:.4f},{z:.4f},{yaw:.4f}\n"
    controller.buffer.append(record)

    # Flush buffer when it reaches the limit
    if len(controller.buffer) >= controller.buffer_limit:
        with open(controller.file_path, 'a') as f:
            f.writelines(controller.buffer)
        controller.buffer.clear()

    # ==============================================================
    # 2. Core tuning parameters (stiff P-loop + mild I-loop)
    # ==============================================================
    # Outer-loop PID gains (position to velocity)
    # High P gains make the drone aggressively correct position errors
    Kp_outer = np.array([1.8, 1.8, 1.2])   
    # Low I gains only trim steady-state error gently
    Ki_outer = np.array([0.6, 0.6, 0.5])   
    # Strong D gains dampen oscillations caused by high P
    Kd_outer = np.array([0.5, 0.5, 0.3]) 

    # Inner-loop PID gains (velocity to acceleration)
    Kp_inner = np.array([0.6, 0.6, 0.6])   
    Ki_inner = np.array([0.0, 0.0, 0.0])   # Integral is disabled in the inner loop
    Kd_inner = np.array([0.25, 0.25, 0.05]) 

    # EMA smoothing factors for velocity estimation and its derivative
    alpha_vel = np.array([0.75, 0.75, 0.25]) 
    alpha_d_vel = np.array([0.6, 0.6, 0.3]) 

    # ==============================================================
    # 3. Initialize persistent variables
    # ==============================================================
    if not hasattr(controller, 'prev_pos'):
        # Previous position for velocity computation
        controller.prev_pos = np.array([x, y, z])
        # Previous position error (used for integral windup logic)
        controller.prev_pos_err = np.array([0.0, 0.0, 0.0]) 
        # Filtered velocity (world frame)
        controller.filtered_vel = np.array([0.0, 0.0, 0.0])
        # Integral of position error (world frame)
        controller.pos_integral = np.array([0.0, 0.0, 0.0]) 
        # Previous velocity error (inner loop)
        controller.prev_vel_err = np.array([0.0, 0.0, 0.0])
        # Filtered derivative of velocity error
        controller.filtered_d_vel = np.array([0.0, 0.0, 0.0])

    # Reset all persistent states when the target changes
    if not hasattr(controller, 'last_target'):
        controller.last_target = None
    
    if controller.last_target != target_pos:
        controller.prev_pos = np.array([x, y, z])
        controller.prev_pos_err = np.array([tx - x, ty - y, tz - z])
        controller.filtered_vel = np.array([0.0, 0.0, 0.0])
        controller.pos_integral = np.array([0.0, 0.0, 0.0])
        controller.prev_vel_err = np.array([0.0, 0.0, 0.0])
        controller.filtered_d_vel = np.array([0.0, 0.0, 0.0])
        controller.last_target = target_pos

    # ==============================================================
    # 4. Velocity computation with per-axis EMA filter
    # ==============================================================
    current_pos = np.array([x, y, z])
    raw_vel = (current_pos - controller.prev_pos) / dt
    controller.filtered_vel = alpha_vel * raw_vel + (1.0 - alpha_vel) * controller.filtered_vel

    # ==============================================================
    # 5. Outer-loop PID (position loop) with anti-windup and freeze logic
    # ==============================================================
    pos_error = np.array([tx - x, ty - y, tz - z])
    # Derivative of position error is approximated by negative of filtered velocity
    d_pos_err = -controller.filtered_vel

    for i in range(3):
        # When the error crosses zero (sign change), reduce integral accumulation
        # to avoid overshoot while retaining some wind disturbance memory.
        if pos_error[i] * controller.prev_pos_err[i] < 0:
            controller.pos_integral[i] *= 0.5 

        # Directional integral freeze: if the error is small and the drone
        # is already moving fast enough toward the target, freeze the integral
        # to prevent overshoot. Otherwise allow normal integration.
        if abs(pos_error[i]) < 0.3:
            is_approaching = (pos_error[i] * controller.filtered_vel[i]) > 0
            is_moving_fast = abs(controller.filtered_vel[i]) > 0.05

            if is_approaching and is_moving_fast:
                # Integral is frozen (no accumulation)
                pass
            else:
                # Drone is being pushed away or stuck; integral allowed to grow
                controller.pos_integral[i] += pos_error[i] * Ki_outer[i] * dt
            
        # Clamp integral to a physically meaningful wind compensation limit (0.8 m/s)
        controller.pos_integral[i] = np.clip(controller.pos_integral[i], -0.8, 0.8)

    target_vel = (Kp_outer * pos_error) + controller.pos_integral + (Kd_outer * d_pos_err)
    target_vel = np.clip(target_vel, -1.5, 1.5)

    # ==============================================================
    # 6. Inner-loop PID (velocity loop)
    # ==============================================================
    vel_error = target_vel - controller.filtered_vel
    
    raw_d_vel_err = (vel_error - controller.prev_vel_err) / dt
    controller.filtered_d_vel = alpha_d_vel * raw_d_vel_err + (1.0 - alpha_d_vel) * controller.filtered_d_vel

    # World-frame acceleration command (velocity feedforward + PID)
    v_world = target_vel + (Kp_inner * vel_error) + (Kd_inner * controller.filtered_d_vel)

    # ==============================================================
    # 7. Yaw control (proportional only)
    # ==============================================================
    eyaw = tyaw - yaw
    # Wrap angle error to [-pi, pi]
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 2.0 * eyaw 

    # ==============================================================
    # 8. Coordinate transformation & state update
    # ==============================================================
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    # Rotate world-frame velocity command into body frame
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    # Update persistent variables for next iteration
    controller.prev_pos = current_pos
    controller.prev_pos_err = pos_error 
    controller.prev_vel_err = vel_error

    # ==============================================================
    # 9. Output clamping and NaN/Inf cleaning
    # ==============================================================
    def clean_and_clamp(val, min_val, max_val):
        """Return clamped float, replacing NaN/Inf with 0.0."""
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(np.clip(val, min_val, max_val))

    vx_body = clean_and_clamp(vx_body, -1.5, 1.5)
    vy_body = clean_and_clamp(vy_body, -1.5, 1.5)
    vz_body = clean_and_clamp(vz_body, -1.5, 1.5)
    yaw_rate_cmd = clean_and_clamp(yaw_rate_cmd, -1.74, 1.74)

    # Final safety check: if any command is still invalid, return zeros
    if math.isnan(vx_body) or math.isnan(vy_body) or math.isnan(vz_body):
        return (0.0, 0.0, 0.0, 0.0)

    return (vx_body, vy_body, vz_body, yaw_rate_cmd)