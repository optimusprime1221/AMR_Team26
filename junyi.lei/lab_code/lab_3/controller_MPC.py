import numpy as np
import math

def controller(state, target_pos, dt, wind_enabled=False):
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # ==============================================================
    # Static variables and pre-computation (Executes only once)
    # ==============================================================
    if not hasattr(controller, 'is_initialized'):
        # 1. State tracking for Integral and Derivative terms
        controller.integral_err = np.array([0.0, 0.0, 0.0])
        controller.prev_error = np.array([0.0, 0.0, 0.0])
        
        # 2. Pre-compute MPC gain to eliminate computational delay
        N = 5
        Q = 5.0
        R = 2.0
        H = np.array([[dt * (i+1)] for i in range(N)])
        Q_mat = np.eye(N) * Q
        Ht_Q = H.T @ Q_mat
        Ht_Q_H_plus_R = Ht_Q @ H + np.array([[R]])
        K_mpc = np.linalg.inv(Ht_Q_H_plus_R) @ H.T @ Q_mat
        controller.mpc_gain = np.sum(K_mpc)
        
        # 3. Data logging initialization
        controller.buffer = []
        controller.buffer_limit = 50
        controller.file_path = "data_mpc.csv"
        controller.sim_time = 0.0
        
        with open(controller.file_path, 'w') as f:
            f.write("time,target_x,target_y,target_z,target_yaw,x,y,z,yaw\n")
        
        def flush_buffer():
            if hasattr(controller, "buffer") and controller.buffer:
                with open(controller.file_path, 'a') as f:
                    f.writelines(controller.buffer)
        
        import atexit
        atexit.register(flush_buffer)
        controller.is_initialized = True

    # ==============================================================
    # Error calculation and PID compensation
    # ==============================================================
    pos_error = np.array([tx - x, ty - y, tz - z])
    
    # Accumulate integral error with Anti-windup limit
    controller.integral_err += pos_error * dt
    controller.integral_err = np.clip(controller.integral_err, -2.0, 2.0)
    
    # Calculate derivative error (Damping term)
    derivative_err = (pos_error - controller.prev_error) / dt
    controller.prev_error = pos_error

    # ==============================================================
    # Data logging module
    # ==============================================================
    controller.sim_time += dt
    record = f"{controller.sim_time:.4f},{tx:.4f},{ty:.4f},{tz:.4f},{tyaw:.4f},{x:.4f},{y:.4f},{z:.4f},{yaw:.4f}\n"
    controller.buffer.append(record)

    if len(controller.buffer) >= controller.buffer_limit:
        with open(controller.file_path, 'a') as f:
            f.writelines(controller.buffer)
        controller.buffer.clear()

    # ==============================================================
    # Control Law
    # ==============================================================
    # Reduced Kd to make the response less sluggish (moving back a little)
    Kd = 0.25 
    Ki = 0.2 if wind_enabled else 0.0
    
    # Final velocity command = MPC Proportional + Integral + Derivative damping
    v_world = (controller.mpc_gain * pos_error) + (Ki * controller.integral_err) + (Kd * derivative_err)

    # Yaw control logic
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 1.5 * eyaw 

    # Coordinate frame rotation (World to Body)
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    return (float(np.clip(vx_body, -1.0, 1.0)), 
            float(np.clip(vy_body, -1.0, 1.0)), 
            float(np.clip(vz_body, -1.0, 1.0)), 
            float(np.clip(yaw_rate_cmd, -1.74, 1.74)))