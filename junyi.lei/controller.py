# wind_flag = False
# Implement a controller

import numpy as np
import math

def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))

    # Unpack state and target
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==========================================
    # --- PID TUNING PARAMETERS ---
    # Change these values to adjust flight behavior
    # ==========================================
    
    # Proportional (P): How fast it moves toward the target
    Kp_xy = 1.2
    Kp_z = 1.0
    Kp_yaw = 2.0
    
    # Integral (I): How hard it fights steady-state errors (like wind)
    Ki_xy = 0.5 if wind_enabled else 0.0
    Ki_z = 0.5 if wind_enabled else 0.0
    
    # Derivative (D): How much it slows down when approaching the target (prevents overshoot)
    Kd_xy = 0.15
    Kd_z = 0.1
    Kd_yaw = 0.1
    
    # ==========================================

    # 1. Initialize persistent variables for I and D terms
    if not hasattr(controller, 'integral_error'):
        controller.integral_error = np.array([0.0, 0.0, 0.0])
    if not hasattr(controller, 'prev_error'):
        # Stores previous errors for: [x, y, z, yaw]
        controller.prev_error = np.array([0.0, 0.0, 0.0, 0.0]) 

    # 2. Calculate current errors (World Frame)
    ex = tx - x
    ey = ty - y
    ez = tz - z

    # Calculate yaw error and wrap it to the range [-pi, pi]
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    
    current_error = np.array([ex, ey, ez, eyaw])

    # 3. Calculate Derivative (Rate of change of error)
    # Formula: D = (Current Error - Previous Error) / dt
    derivative = (current_error - controller.prev_error) / dt
    dx, dy, dz, dyaw = derivative

    # 4. Calculate Integral (Accumulated error)
    if wind_enabled:
        controller.integral_error[0] += ex * dt
        controller.integral_error[1] += ey * dt
        controller.integral_error[2] += ez * dt
        
        # Anti-windup constraint
        max_integral = 2.0
        controller.integral_error = np.clip(controller.integral_error, -max_integral, max_integral)
    else:
        controller.integral_error = np.array([0.0, 0.0, 0.0])

    ix, iy, iz = controller.integral_error

    # 5. Compute the final PID outputs (World Frame)
    vx_world = (Kp_xy * ex) + (Ki_xy * ix) + (Kd_xy * dx)
    vy_world = (Kp_xy * ey) + (Ki_xy * iy) + (Kd_xy * dy)
    vz_world = (Kp_z * ez)  + (Ki_z * iz)  + (Kd_z * dz)
    yaw_rate_cmd = (Kp_yaw * eyaw) + (Kd_yaw * dyaw)

    # 6. Transform World Frame velocities to Body Frame
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    vx_body = vx_world * cos_yaw + vy_world * sin_yaw
    vy_body = -vx_world * sin_yaw + vy_world * cos_yaw
    vz_body = vz_world 

    # 7. Save current error for the next loop
    controller.prev_error = current_error

    # 8. Clip final outputs to safety limits
    vx_body = float(np.clip(vx_body, -1.0, 1.0))
    vy_body = float(np.clip(vy_body, -1.0, 1.0))
    vz_body = float(np.clip(vz_body, -1.0, 1.0))
    yaw_rate_cmd = float(np.clip(yaw_rate_cmd, -1.74533, 1.74533))

    return (vx_body, vy_body, vz_body, yaw_rate_cmd)