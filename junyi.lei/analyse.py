import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def calculate_metrics(time, actual, target):
    """
    Calculate performance metrics: Rise Time (90%), Overshoot, Steady-State Error
    """
    final_target = target.iloc[-1]
    initial_val = actual.iloc[0]
    step_size = abs(final_target - initial_val)
    
    # 1. Steady state error (mean of last 50 points)
    steady_state_actual = actual.iloc[-50:].mean()
    ss_error = abs(final_target - steady_state_actual)
    
    # Return NaN if there is no target step change
    if step_size < 1e-5:
        return np.nan, 0.0, ss_error

    # 2. Rise time (Time to reach 90% of step change)
    threshold_90 = initial_val + 0.9 * (final_target - initial_val)
    if final_target > initial_val:
        reached = actual >= threshold_90
    else:
        reached = actual <= threshold_90
        
    if reached.any():
        idx = reached.idxmax()
        rise_time = time.loc[idx] - time.iloc[0]
    else:
        rise_time = np.nan
        
    # 3. Overshoot (%)
    if final_target > initial_val:
        peak = actual.max()
        overshoot = (peak - final_target) / step_size * 100
    else:
        peak = actual.min()
        overshoot = (final_target - peak) / step_size * 100
        
    overshoot = max(0, overshoot) # Set to 0 if no overshoot
    
    return rise_time, overshoot, ss_error

def process_csv(file_path):
    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path)
    
    if 'time' not in df.columns:
        print(f"Error: 'time' column not found in {file_path}. Skipping.")
        return
        
    time = df['time']
    
    # Convert yaw from radians to degrees and normalize to 0-360
    df['yaw_deg'] = np.degrees(df['yaw']) % 360
    df['target_yaw_deg'] = np.degrees(df['target_yaw']) % 360
    
    # Calculate metrics for X, Y, Z
    metrics = []
    for axis in ['x', 'y', 'z']:
        rt, os_val, sse = calculate_metrics(time, df[axis], df[f'target_{axis}'])
        metrics.append((rt, os_val, sse))
        
    metrics = np.array(metrics)
    avg_rt, avg_os, avg_sse = np.nanmean(metrics, axis=0)
    
    # Calculate 3D total error
    error_x = df['target_x'] - df['x']
    error_y = df['target_y'] - df['y']
    error_z = df['target_z'] - df['z']
    total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
    
    # Set up layout using GridSpec to make the 5th plot taller
    fig = plt.figure(figsize=(10, 16))
    gs = fig.add_gridspec(6, 1) # Divide figure into 6 rows
    
    plot_configs = [
        (0, 'x', 'X Position (m)', df['target_x'], df['x']),
        (1, 'y', 'Y Position (m)', df['target_y'], df['y']),
        (2, 'z', 'Z Position (m)', df['target_z'], df['z']),
        (3, 'yaw', 'Yaw Angle (deg)', df['target_yaw_deg'], df['yaw_deg'])
    ]
    
    # Plot the first 4 subplots (each takes 1 row)
    for pos, axis, ylabel, target_data, actual_data in plot_configs:
        ax = fig.add_subplot(gs[pos, 0])
        ax.plot(time, target_data, label=f'Target {axis.upper()}', linestyle='--', color='black', linewidth=1.5)
        ax.plot(time, actual_data, label=f'Actual {axis.upper()}', color='#1f77b4', linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)

    # Plot the 5th subplot (Total Error) taking 2 rows (rows 4 and 5)
    ax_err = fig.add_subplot(gs[4:6, 0])
    ax_err.plot(time, total_error, label='XYZ Total Error', color='red', linewidth=1.5)
    
    # Add Zero Error Reference Line
    ax_err.axhline(0, color='green', linestyle='--', linewidth=2, label='Zero Error Ref')
    
    ax_err.set_ylabel('Error Distance (m)', fontsize=12)
    ax_err.set_xlabel('Time (s)', fontsize=12)
    ax_err.legend(loc='upper right')
    ax_err.grid(True, linestyle=':', alpha=0.6)
    
    # Add text box with average metrics
    info_text = (
        f"Average Metrics (X, Y, Z combined):\n"
        f"Avg Rise Time (90%): {avg_rt:.3f} s\n"
        f"Avg Overshoot: {avg_os:.2f} %\n"
        f"Avg Steady-State Error: {avg_sse:.4f} m"
    )
    
    ax_err.text(0.5, 0.5, info_text, transform=ax_err.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.suptitle(f'Data Analysis Report: {os.path.basename(file_path)}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    save_name = f"{os.path.splitext(file_path)[0]}_analysis.png"
    plt.savefig(save_name, dpi=150)
    plt.close()
    print(f"✅ Plot saved as: {save_name}\n")

if __name__ == "__main__":
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found in the current directory.")
    else:
        for file in csv_files:
            process_csv(file)