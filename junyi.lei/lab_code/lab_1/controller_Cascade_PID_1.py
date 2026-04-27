import numpy as np
import math
import matplotlib as mpl # Used to fix the 'k' key crash bug

def controller(state, target_pos, dt, wind_enabled=False):
    # ==============================================================
    # 0. Ultimate anti-crash patch & baseline safety protection
    # ==============================================================
    # Fix the Matplotlib Bug where the 'k' key triggers a log-scale crash
    if not hasattr(controller, 'hotkey_fixed'):
        if 'k' in mpl.rcParams['keymap.xscale']:
            mpl.rcParams['keymap.xscale'].remove('k')
        if 'l' in mpl.rcParams['keymap.yscale']:
            mpl.rcParams['keymap.yscale'].remove('l')
        controller.hotkey_fixed = True

    # Safety check for initial zero dt
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    # 1. Parse state
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==============================================================
    # 2. Core Tuning Area - Incorporating your parameters
    # ==============================================================
    # [Outer Loop: Position -> Target Velocity] 
    # (Controls speed to target, resists steady wind)
    Kp_outer = np.array([1.2, 1.2, 1.0]) 
    
    # [Optimization] Moved the anti-wind I term to the outer loop to 
    # truly eliminate steady-state error ("blown away and can't return")
    Ki_outer = np.array([0.5, 0.5, 0.5]) if wind_enabled else np.array([0.0, 0.0, 0.0])
    Kd_outer = np.array([0.1, 0.1, 0.1]) # Outer loop damping to prevent overshoot

    # [Inner Loop: Target Velocity -> Velocity Command Compensation] 
    # (Controls low-level response)
    Kp_inner = np.array([0.8, 0.8, 0.8])
    Ki_inner = np.array([0.0, 0.0, 0.0]) # Inner loop integral set to 0 to prevent loops from fighting
    Kd_inner = np.array([0.05, 0.05, 0.05]) # Kept your D term, but paired it with the filter below

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
    # 4. Velocity Low-Pass Filter (The magic tool to suppress oscillations)
    # Since you kept the inner loop Kd (0.05), the drone would violently 
    # twitch without filtering.
    # ==============================================================
    alpha = 0.3 # Filtering coefficient
    current_pos = np.array([x, y, z])
    raw_vel = (current_pos - controller.prev_pos) / dt
    controller.filtered_vel = (alpha * raw_vel) + ((1.0 - alpha) * controller.filtered_vel)

    # ==============================================================
    # 5. [Outer Loop: Position Loop] 
    # ==============================================================
    pos_error = np.array([tx - x, ty - y, tz - z])
    d_pos_err = (pos_error - controller.prev_pos_err) / dt

    # Outer loop integral and anti-windup
    controller.pos_integral += pos_error * dt
    controller.pos_integral = np.clip(controller.pos_integral, -1.0, 1.0)

    # Calculate target velocity
    target_vel = (Kp_outer * pos_error) + (Ki_outer * controller.pos_integral) + (Kd_outer * d_pos_err)
    target_vel = np.clip(target_vel, -1.5, 1.5) # Limit maximum target velocity

    # ==============================================================
    # 6. [Inner Loop: Velocity Loop]
    # ==============================================================
    vel_error = target_vel - controller.filtered_vel
    d_vel_err = (vel_error - controller.prev_vel_err) / dt

    controller.vel_integral += vel_error * dt
    controller.vel_integral = np.clip(controller.vel_integral, -0.5, 0.5)

    # Feedforward (directly passing target_vel) + Compensation (PID)
    v_world = target_vel + (Kp_inner * vel_error) + (Ki_inner * controller.vel_integral) + (Kd_inner * d_vel_err)

    # ==============================================================
    # 7. [Yaw Control] 
    # ==============================================================
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 2.0 * eyaw 

    # ==============================================================
    # 8. [Coordinate Transformation & Cleanup]
    # ==============================================================
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    # World Frame to Body Frame
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    # Update history state
    controller.prev_pos = current_pos
    controller.prev_pos_err = pos_error
    controller.prev_vel_err = vel_error

    # ==============================================================
    # 9. Final output clamping and cleaning (Anti-crash mechanism)
    # ==============================================================
    def clean_and_clamp(val, min_val, max_val):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(np.clip(val, min_val, max_val))

    vx_body = clean_and_clamp(vx_body, -1.0, 1.0)
    vy_body = clean_and_clamp(vy_body, -1.0, 1.0)
    vz_body = clean_and_clamp(vz_body, -1.0, 1.0)
    yaw_rate_cmd = clean_and_clamp(yaw_rate_cmd, -1.74, 1.74)

    # Completely prevent dirty data from being passed to the physics engine
    if math.isnan(vx_body) or math.isnan(vy_body) or math.isnan(vz_body):
        return (0.0, 0.0, 0.0, 0.0)

    return (vx_body, vy_body, vz_body, yaw_rate_cmd)