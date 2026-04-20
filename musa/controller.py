wind_flag = True
# Set to True so the marker tests wind compensation (worth 1 mark)

# METHOD: Cascade PID Controller (3 layers)
# This is an advanced controller method as it uses more than 2 PID layers.
#
# A standard PID takes position error and outputs a velocity command directly.
# A cascade PID splits this into multiple loops:
#
#   Outer loop  (position)  — takes position error, outputs a desired velocity setpoint
#   Middle loop (velocity)  — takes velocity error, outputs a refined velocity command
#   Inner loop  (smoothing) — low-pass filters the output to reduce jitter and overshoot
#
# This structure means the drone is always trying to hit a smooth velocity target
# rather than reacting to raw position error directly, which reduces overshoot
# and gives a much cleaner approach to the target point.

import math
import numpy as np

def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [x (m), y (m), z (m), roll (rad), pitch (rad), yaw (rad)]
    # target_pos format: (x (m), y (m), z (m), yaw (rad))
    # dt: time step (s)
    # wind_enabled: whether wind disturbance compensation should be active
    # returns: (vx (m/s), vy (m/s), vz (m/s), yaw_rate (rad/s)) in body frame

    # -----------------------------------------------------------------------
    # TUNING PARAMETERS
    # -----------------------------------------------------------------------

    # --- Outer loop (position -> desired velocity) ---
    # Higher Kp_pos = reaches target faster but more overshoot
    # Higher Kd_pos = smoother approach, less overshoot
    Kp_pos = 1.2
    Ki_pos = 0.3 if wind_enabled else 0.0
    Kd_pos = 0.25

    # --- Middle loop (velocity correction) ---
    # This loop refines the velocity setpoint from the outer loop.
    # Keep these gains smaller than outer loop or it becomes unstable.
    Kp_vel = 0.4
    Kd_vel = 0.05

    # --- Inner loop (low-pass filter coefficient) ---
    # Blends previous filtered output with new command.
    # 0.0 = no filtering (instant response), 0.3 = smooth but slightly slower
    alpha = 0.2

    # --- Yaw PID ---
    Kp_yaw = 2.0
    Kd_yaw = 0.1

    # Safety limits
    max_vel      = 1.0
    max_yaw_rate = 1.74
    max_integral = 2.0

    # -----------------------------------------------------------------------
    # Unpack state and target
    # -----------------------------------------------------------------------
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, t_yaw = target_pos

    # -----------------------------------------------------------------------
    # Initialise persistent state on the function object
    # -----------------------------------------------------------------------
    if not hasattr(controller, 'prev_pos_err'):
        # Previous errors for derivative terms
        controller.prev_pos_err  = np.array([0.0, 0.0, 0.0])
        controller.prev_vel_err  = np.array([0.0, 0.0, 0.0])
        controller.prev_eyaw     = 0.0
        # Integrators for wind compensation
        controller.integral_pos  = np.array([0.0, 0.0, 0.0])
        # Filtered output from inner loop (starts at zero)
        controller.filtered_vel  = np.array([0.0, 0.0, 0.0])

    # -----------------------------------------------------------------------
    # OUTER LOOP — Position PID
    # Computes a desired velocity from position error.
    # This tells the drone "you need to be moving at X m/s to reach the target".
    # -----------------------------------------------------------------------
    pos_err = np.array([tx - x, ty - y, tz - z])

    # Integral — only used for wind rejection
    if wind_enabled:
        controller.integral_pos += pos_err * dt
        controller.integral_pos  = np.clip(controller.integral_pos, -max_integral, max_integral)
    else:
        controller.integral_pos = np.array([0.0, 0.0, 0.0])

    # Derivative of position error
    d_pos_err = (pos_err - controller.prev_pos_err) / dt
    controller.prev_pos_err = pos_err.copy()

    # Desired velocity setpoint from outer loop
    vel_desired = (Kp_pos * pos_err
                 + Ki_pos * controller.integral_pos
                 + Kd_pos * d_pos_err)

    # -----------------------------------------------------------------------
    # MIDDLE LOOP — Velocity PID
    # Compares the desired velocity from the outer loop against the
    # previous actual command sent, and corrects for any discrepancy.
    # This adds extra damping and helps reject disturbances mid-flight.
    # -----------------------------------------------------------------------
    vel_err = vel_desired - controller.filtered_vel  # error between desired and actual cmd
    d_vel_err = (vel_err - controller.prev_vel_err) / dt
    controller.prev_vel_err = vel_err.copy()

    vel_corrected = vel_desired + Kp_vel * vel_err + Kd_vel * d_vel_err

    # -----------------------------------------------------------------------
    # INNER LOOP — Low-pass filter (smoothing)
    # Blends the new corrected velocity with the previous filtered output.
    # This removes high-frequency jitter from the derivative terms and
    # gives the drone a smoother, more natural-looking flight path.
    # Formula: filtered = alpha * previous + (1 - alpha) * new
    # -----------------------------------------------------------------------
    controller.filtered_vel = (alpha * controller.filtered_vel
                              + (1.0 - alpha) * vel_corrected)

    vx_world, vy_world, vz_world = controller.filtered_vel

    # -----------------------------------------------------------------------
    # Yaw PID (single loop — cascade not needed for yaw)
    # -----------------------------------------------------------------------
    eyaw = t_yaw - yaw
    eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]

    dyaw = (eyaw - controller.prev_eyaw) / dt
    controller.prev_eyaw = eyaw

    yaw_rate = Kp_yaw * eyaw + Kd_yaw * dyaw

    # -----------------------------------------------------------------------
    # Coordinate frame transformation — world frame -> body frame
    # PID computes velocities in the global world frame (north/east/up).
    # The drone expects commands in its body frame (forward/left/up),
    # which is rotated by the current yaw angle.
    # Rotating by -yaw converts from world to body frame.
    # -----------------------------------------------------------------------
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    vx_body =  vx_world * cos_yaw + vy_world * sin_yaw
    vy_body = -vx_world * sin_yaw + vy_world * cos_yaw
    vz_body =  vz_world  # z is unaffected by yaw rotation

    # -----------------------------------------------------------------------
    # Clip outputs to safe operating limits
    # -----------------------------------------------------------------------
    vx_body  = float(np.clip(vx_body,  -max_vel,      max_vel))
    vy_body  = float(np.clip(vy_body,  -max_vel,      max_vel))
    vz_body  = float(np.clip(vz_body,  -max_vel,      max_vel))
    yaw_rate = float(np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate))

    return (vx_body, vy_body, vz_body, yaw_rate)