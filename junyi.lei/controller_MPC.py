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

    # 应对风力的稳态积分 / Integral for steady-state wind
    if wind_enabled:
        controller.integral_err += pos_error * dt
        controller.integral_err = np.clip(controller.integral_err, -2.0, 2.0)
    else:
        controller.integral_err = np.array([0.0, 0.0, 0.0])

    # ==============================================================
    # 极简离散解析 MPC (预测期 N=3) / Simplified Analytical MPC (Horizon N=3)
    # 系统模型 / System Model: X(k+1) = A*X(k) + B*U(k) 
    # 此处假设控制输入 u 就是速度，所以位置 p(k+1) = p(k) + u*dt
    # ==============================================================
    # 权重矩阵 / Weighting matrices
    Q = 10.0  # State cost (惩罚位置误差)
    R = 1.0   # Input cost (惩罚剧烈运动)
    
    # 由于系统是线性独立的解耦轴，我们对单个轴计算最优预测增益 (Unconstrained Solution)
    # 对于模型 p(k+1) = p(k) + u*dt，预测矩阵 H 和 P 可以被提取。
    # 简化数学推导后，最优预测增益相当于一个动态计算的比例系数：
    # optimal_u = [ (H^T Q H + R)^-1 H^T Q ] * Error
    
    # 预测矩阵构建 (N=3) / Prediction Matrices construction
    H = np.array([[dt], [2*dt], [3*dt]])
    Q_mat = np.eye(3) * Q
    
    # 核心矩阵运算: K_mpc = (H^T * Q * H + R)^-1 * H^T * Q
    # Matrix operations using raw numpy
    Ht_Q = H.T @ Q_mat
    Ht_Q_H_plus_R = Ht_Q @ H + np.array([[R]])
    K_mpc = np.linalg.inv(Ht_Q_H_plus_R) @ H.T @ Q_mat
    
    # K_mpc is a 1x3 matrix. We only care about the first step command (Receding Horizon)
    # 取矩阵的第一行和第一列的增益和 / Sum the gain for current error projection
    mpc_gain = np.sum(K_mpc)
    
    # 应用解析 MPC 增益并叠加积分项抗风 / Apply MPC gain + Integral term for wind
    v_world = (mpc_gain * pos_error) + (0.5 * controller.integral_err)

    # 偏航角计算 / Yaw logic
    eyaw = tyaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_cmd = 1.5 * eyaw 

    # 坐标系转换 / Coordinate rotation
    vx_w, vy_w, vz_w = v_world
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    
    vx_body = vx_w * cos_yaw + vy_w * sin_yaw
    vy_body = -vx_w * sin_yaw + vy_w * cos_yaw
    vz_body = vz_w

    return (float(np.clip(vx_body, -1.0, 1.0)), 
            float(np.clip(vy_body, -1.0, 1.0)), 
            float(np.clip(vz_body, -1.0, 1.0)), 
            float(np.clip(yaw_rate_cmd, -1.74, 1.74)))