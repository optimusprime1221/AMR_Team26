import numpy as np
import math

def controller(state, target_pos, dt, wind_enabled=False):
    """
    Cascaded PID Controller for Quadrotor Position and Yaw.

    This function implements a dual-loop control strategy:
    1. Outer Loop (Position): Computes target velocities based on position error, 
       incorporating integral separation and zero-crossing reset to handle wind 
       disturbances without excessive overshoot.
    2. Inner Loop (Velocity): Refines velocity commands using the difference between 
       target and estimated actual velocity.
    The controller includes EMA filtering for noise reduction and coordinates 
    transformation from the World frame to the Body frame.
    """

    # Prevent division by zero or numerical instability if dt is too small
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    # ==============================================================
    # 1. State Extraction
    # ==============================================================
    # Current state: World position (x, y, z) and Euler angles (roll, pitch, yaw)
    x, y, z, roll, pitch, yaw = state
    # Target state: World target position and target yaw
    tx, ty, tz, tyaw = target_pos


    # ==============================================================
    # 2. PID Gain Configuration
    # ==============================================================
    # Outer Loop PID Gains (Position)
    Kp_outer = np.array([1.8, 1.8, 1.2])   # Proportional gain for position error
    Ki_outer = np.array([2.5, 2.5, 2.0])   # High integral gain for robust wind rejection
    Kd_outer = np.array([0.6, 0.6, 0.3])   # Derivative gain for position error

    # Inner Loop PD Gains (Velocity)
    Kp_inner = np.array([0.6, 0.6, 0.6])   
    Kd_inner = np.array([0.4, 0.4, 0.05])  

    # Exponential Moving Average (EMA) Filter Coefficients
    alpha_vel   = np.array([0.75, 0.75, 0.25])   # Velocity estimation filter
    alpha_d_vel = np.array([0.6, 0.6, 0.3])      # Velocity derivative filter

    # ==============================================================
    # 3. Persistence Variable Initialization
    # Handles history tracking across control loops.
    # ==============================================================
    if not hasattr(controller, 'prev_pos'):
        controller.prev_pos = np.array([x, y, z])
        controller.prev_pos_err = np.array([0.0, 0.0, 0.0]) 
        controller.filtered_vel = np.array([0.0, 0.0, 0.0])
        controller.pos_integral = np.array([0.0, 0.0, 0.0]) 
        controller.prev_vel_err = np.array([0.0, 0.0, 0.0])
        controller.filtered_d_vel = np.array([0.0, 0.0, 0.0])

    if not hasattr(controller, 'last_target'):
        controller.last_target = None
    
    # Reset integral and history if target position changes
    if controller.last_target != target_pos:
        controller.prev_pos = np.array([x, y, z])
        controller.prev_pos_err = np.array([tx - x, ty - y, tz - z])
        controller.filtered_vel = np.array([0.0, 0.0, 0.0])
        controller.pos_integral = np.array([0.0, 0.0, 0.0])
        controller.prev_vel_err = np.array([0.0, 0.0, 0.0])
        controller.filtered_d_vel = np.array([0.0, 0.0, 0.0])
        controller.last_target = target_pos

    # ==============================================================
    # 4. Velocity Estimation
    # Estimates actual velocity via finite difference and EMA filtering.
    # ==============================================================
    current_pos = np.array([x, y, z])
    raw_vel = (current_pos - controller.prev_pos) / dt
    controller.filtered_vel = alpha_vel * raw_vel + (1.0 - alpha_vel) * controller.filtered_vel

    # ==============================================================
    # 5. Outer Loop PID: Position Control
    # Includes Zero-Crossing reset and Integral Separation logic.
    # ==============================================================
    pos_error = np.array([tx - x, ty - y, tz - z])
    d_pos_err = -controller.filtered_vel

    for i in range(3):
        # ---------- Zero-Crossing Reset ----------
        # Reset integral if error changes sign (crossing target) to prevent overshoot.
        if pos_error[i] * controller.prev_pos_err[i] < 0:
            controller.pos_integral[i] = 0.0 

        # ---------- Integral Separation ----------
        # Only enable integration when error is within 1.0m to prevent windup during travel.
        if abs(pos_error[i]) < 1.0:
            # Check if drone is approaching target at sufficient speed
            is_approaching = (pos_error[i] * controller.filtered_vel[i]) > 0
            is_moving_fast = abs(controller.filtered_vel[i]) > 0.05

            if is_approaching and is_moving_fast:
                # Freeze integral if momentum is already carrying drone to target
                pass
            else:
                # Accumulate integral to fight static errors or wind disturbances
                controller.pos_integral[i] += pos_error[i] * Ki_outer[i] * dt
            
        # ---------- Integral Clamping ----------
        # Limit integral influence to 1.2 m/s for stability
        controller.pos_integral[i] = np.clip(controller.pos_integral[i], -1.2, 1.2)

    # Calculate target velocity: P + I + D, clamped to 1.5 m/s
    target_vel = (Kp_outer * pos_error) + controller.pos_integral + (Kd_outer * d_pos_err)
    target_vel = np.clip(target_vel, -1.5, 1.5)

    # ==============================================================
    # 6. Inner Loop PID: Velocity Control
    # ==============================================================
    vel_error = target_vel - controller.filtered_vel
    raw_d_vel_err = (vel_error - controller.prev_vel_err) / dt
    controller.filtered_d_vel = alpha_d_vel * raw_d_vel_err + (1.0 - alpha_d_vel) * controller.filtered_d_vel

    # World-frame velocity command
    v_world = target_vel + (Kp_inner * vel_error) + (Kd_inner * controller.filtered_d_vel)

    # ==============================================================
    # 7. Yaw Control
    # Simple proportional control with angle wrapping.
    # ==============================================================
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 2.0 * eyaw  

    # ==============================================================
    # 8. Coordinate Transformation
    # Transforms World-frame velocity to Body-frame velocity.
    # ==============================================================
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    # Update history for next iteration
    controller.prev_pos = current_pos
    controller.prev_pos_err = pos_error 
    controller.prev_vel_err = vel_error

    # ==============================================================
    # 9. Output Clamping and Safety Sanity Check
    # ==============================================================
    def clean_and_clamp(val, min_val, max_val):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(np.clip(val, min_val, max_val))

    vx_body = clean_and_clamp(vx_body, -1.5, 1.5)
    vy_body = clean_and_clamp(vy_body, -1.5, 1.5)
    vz_body = clean_and_clamp(vz_body, -1.5, 1.5)
    yaw_rate_cmd = clean_and_clamp(yaw_rate_cmd, -1.74, 1.74)

    if math.isnan(vx_body) or math.isnan(vy_body) or math.isnan(vz_body):
        return (0.0, 0.0, 0.0, 0.0)

    return (vx_body, vy_body, vz_body, yaw_rate_cmd)