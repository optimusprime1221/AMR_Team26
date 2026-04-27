import numpy as np
import math

def controller(state, target_pos, dt, wind_enabled=False):
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0, 0.0)

    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    if not hasattr(controller, 'integral_err'):
        controller.integral_err = np.array([0.0, 0.0, 0.0])

    pos_error = np.array([tx - x, ty - y, tz - z])

    # ==============================================================
    # 🌟 Data Logging Module
    # ==============================================================
    if not hasattr(controller, "is_initialized_csv"):
        controller.buffer = []
        controller.buffer_limit = 50     # Write to disk every 50 frames
        controller.file_path = "data_mpc.csv"
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
    # Simplified Analytical MPC (Horizon N=3)
    # System Model: X(k+1) = A*X(k) + B*U(k) 
    # Here we assume control input u is velocity, so position p(k+1) = p(k) + u*dt
    # ==============================================================
    # Weighting matrices
    Q = 5.0  # State cost (penalize position error)
    R = 2.0   # Input cost (penalize aggressive motion)
    
    # Since the system is linear and decoupled per axis, we compute the optimal predictive gain per axis (Unconstrained Solution)
    # For model p(k+1) = p(k) + u*dt, the prediction matrices H and P can be extracted.
    # After simplifying the math, the optimal predictive gain is equivalent to a dynamically computed proportional coefficient:
    # optimal_u = [ (H^T Q H + R)^-1 H^T Q ] * Error
    
    # Prediction Matrices construction (N=3)
    N = 5
    H = np.array([[dt * (i+1)] for i in range(N)])   # [dt, 2dt, ..., N*dt]
    Q_mat = np.eye(N) * Q
    
    # Core matrix operations: K_mpc = (H^T * Q * H + R)^-1 * H^T * Q
    # Matrix operations using raw numpy
    Ht_Q = H.T @ Q_mat
    Ht_Q_H_plus_R = Ht_Q @ H + np.array([[R]])
    K_mpc = np.linalg.inv(Ht_Q_H_plus_R) @ H.T @ Q_mat
    
    # K_mpc is a 1x3 matrix. We only care about the first step command (Receding Horizon)
    # Sum the gain for current error projection
    mpc_gain = np.sum(K_mpc)
    
    # Apply analytical MPC gain and add integral term for wind rejection
    v_world = (mpc_gain * pos_error) + (0.5 * controller.integral_err)

    # Yaw logic
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 1.5 * eyaw 

    # Coordinate rotation
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    return (float(np.clip(vx_body, -1.0, 1.0)), 
            float(np.clip(vy_body, -1.0, 1.0)), 
            float(np.clip(vz_body, -1.0, 1.0)), 
            float(np.clip(yaw_rate_cmd, -1.74, 1.74)))