import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def calculate_metrics(time, actual, target):
    final_target = target.iloc[-1]
    initial_val = actual.iloc[0]
    step_size = abs(final_target - initial_val)
    
    if step_size < 1e-5:
        return np.nan, 0.0, None

    # 上升阶段判定标准：95%
    threshold_95 = initial_val + 0.95 * (final_target - initial_val)
    if final_target > initial_val:
        reached = actual >= threshold_95
    else:
        reached = actual <= threshold_95
        
    rise_time = np.nan
    idx_95 = None
    if reached.any():
        idx_95 = reached.idxmax()
        rise_time = time.loc[idx_95] - time.iloc[0]
        
    # 超调量 (%)
    if final_target > initial_val:
        peak = actual.max()
        overshoot = (peak - final_target) / step_size * 100
    else:
        peak = actual.min()
        overshoot = (final_target - peak) / step_size * 100
    overshoot = max(0, overshoot)
    
    return rise_time, overshoot, idx_95

def process_csv(file_path):
    print(f"\n{'='*50}\n📊 Processing file: {file_path}\n{'='*50}")
    df = pd.read_csv(file_path)
    
    if 'time' not in df.columns:
        print(f"Error: 'time' column not found in {file_path}. Skipping.")
        return
        
    time = df['time']
    df['yaw_deg'] = np.degrees(df['yaw']) % 360
    df['target_yaw_deg'] = np.degrees(df['target_yaw']) % 360
    
    df['target_id'] = (df[['target_x', 'target_y', 'target_z', 'target_yaw']].diff().abs().sum(axis=1) > 1e-4).cumsum()
    
    df['pos_err'] = np.sqrt((df['target_x'] - df['x'])**2 + (df['target_y'] - df['y'])**2 + (df['target_z'] - df['z'])**2)
    df['yaw_err'] = np.abs((df['target_yaw'] - df['yaw'] + np.pi) % (2 * np.pi) - np.pi)

    all_eval_pos_errors = []
    all_eval_yaw_errors = []
    rise_times = []
    overshoots = []

    for target_id, group in df.groupby('target_id'):
        group_time = group['time']
        segment_rts = []
        segment_os = []
        
        for axis in ['x', 'y', 'z']:
            rt, os_val, _ = calculate_metrics(group_time, group[axis], group[f'target_{axis}'])
            segment_rts.append(rt)
            segment_os.append(os_val)
            
        rise_times.append(np.nanmean(segment_rts))
        overshoots.append(np.nanmean(segment_os))
        
        # =========================================================================
        # 🌟 修改点：严格遵循 10s 飞行 + 后 10s 评估的官方规则
        # =========================================================================
        t_start = group['time'].iloc[0]
        eval_start = t_start + 10.0
        eval_end = t_start + 20.0
        
        print(f"📍 [Target {target_id}] Evaluating steady-state from {eval_start:.1f}s to {eval_end:.1f}s:")
        
        # 截取从 t_start+10 到 t_start+20 的完整数据段
        window = group[(group['time'] >= eval_start) & (group['time'] <= eval_end)]
        
        if len(window) > 0: 
            w_pos_mean = window['pos_err'].mean()
            w_pos_std = window['pos_err'].std()
            print(f"    ✅ Evaluated 10s window (N={len(window)}): Mean Err = {w_pos_mean:.5f}m, Std = {w_pos_std:.5f}\n")
            all_eval_pos_errors.extend(window['pos_err'].tolist())
            all_eval_yaw_errors.extend(window['yaw_err'].tolist())
        else:
            print(f"    ⚠️ Warning: No data found in the 10s-20s window. Did the simulation run long enough?\n")
            # 如果模拟时间没跑够20s，作为fallback取最后数据，避免报错
            if len(group) > 50:
                all_eval_pos_errors.extend(group['pos_err'].iloc[-50:].tolist())
                all_eval_yaw_errors.extend(group['yaw_err'].iloc[-50:].tolist())

    # 计算最终评分统计量
    pos_mean = np.mean(all_eval_pos_errors) if all_eval_pos_errors else np.nan
    pos_std = np.std(all_eval_pos_errors) if all_eval_pos_errors else np.nan
    yaw_mean = np.mean(all_eval_yaw_errors) if all_eval_yaw_errors else np.nan
    yaw_std = np.std(all_eval_yaw_errors) if all_eval_yaw_errors else np.nan
    
    avg_rt = np.nanmean(rise_times) if rise_times else np.nan
    avg_os = np.nanmean(overshoots) if overshoots else np.nan

    status = {
        "Pos Mean": "Pass" if pos_mean < 0.01 else "Fail",
        "Pos Std": "Pass" if pos_std < 0.01 else "Fail",
        "Yaw Mean": "Pass" if yaw_mean < 0.01 else "Fail",
        "Yaw Std": "Pass" if yaw_std < 0.001 else "Fail"
    }

    # =========================================================================
    # 绘图部分
    # =========================================================================
    fig = plt.figure(figsize=(10, 20))
    gs = fig.add_gridspec(9, 1)
    
    axes_config = [('x', 'X Position (m)'), ('y', 'Y Position (m)'), ('z', 'Z Position (m)')]
    for i, (axis, ylabel) in enumerate(axes_config):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(time, df[f'target_{axis}'], '--k', label=f'Target {axis.upper()}')
        ax.plot(time, df[axis], color='#1f77b4', label=f'Actual {axis.upper()}')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

    ax_yaw = fig.add_subplot(gs[3, 0])
    ax_yaw.plot(time, df['target_yaw_deg'], '--k', label='Target Yaw')
    ax_yaw.plot(time, df['yaw_deg'], color='#1f77b4', label='Actual Yaw')
    ax_yaw.set_ylabel('Yaw (deg)'); ax_yaw.legend(); ax_yaw.grid(True, alpha=0.3)

    total_error = df['pos_err']
    ax_err = fig.add_subplot(gs[4:6, 0])
    ax_err.plot(time, total_error, color='red', label='XYZ Total Error')
    ax_err.axhline(0.01, color='green', linestyle=':', linewidth=2, label='Grading Threshold (0.01m)')
    ax_err.set_ylabel('Error (m)'); ax_err.set_xlabel('Time (s)'); ax_err.legend(); ax_err.grid(True)
    
    # 修改了报告文字，使其更符合当前的测量方法
    info_text = (
        f"Dynamic Analysis Metrics:\n"
        f"Avg Rise Time (95% Path): {avg_rt:.3f} s\n"
        f"Avg Overshoot: {avg_os:.2f} %\n"
        f"10s-20s Steady-State Error: {pos_mean:.5f} m"
    )
    ax_err.text(0.5, 0.5, info_text, transform=ax_err.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='gray', alpha=0.9))

    ax_table = fig.add_subplot(gs[7:9, 0])
    ax_table.axis('off')
    table_data = [
        ["Grading Metric", "Measured Value (10s-20s Window)", "Requirement (<)", "Result"],
        ["Positional Error Mean", f"{pos_mean:.5f} m", "0.01 m", status["Pos Mean"]],
        ["Positional Error Std", f"{pos_std:.5f}", "0.01", status["Pos Std"]],
        ["Yaw Error Mean", f"{yaw_mean:.5f} rad", "0.01 rad", status["Yaw Mean"]],
        ["Yaw Error Std", f"{yaw_std:.5f} rad", "0.001 rad", status["Yaw Std"]]
    ]
    
    ccolors = [['white']*4 for _ in range(5)]
    for r in range(1, 5):
        res = table_data[r][3]
        ccolors[r][3] = '#d4edda' if res == 'Pass' else '#f8d7da'

    tab = ax_table.table(cellText=table_data, cellColours=ccolors, loc='center', cellLoc='center', colWidths=[0.35, 0.3, 0.2, 0.15])
    tab.auto_set_font_size(False); tab.set_fontsize(11); tab.scale(1, 2.2)
    for (r, c), cell in tab.get_celld().items():
        if r == 0: cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#40466e')

    # 更新了标题名称
    plt.suptitle(f'Autograder Report (Strict 10s-20s Eval Window)\nFile: {os.path.basename(file_path)}', fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    save_name = f"{os.path.splitext(file_path)[0]}_grading_report.png"
    plt.savefig(save_name, dpi=150); plt.close()
    print(f"✅ Grading report saved as: {save_name}\n")

if __name__ == "__main__":
    csv_files = glob.glob("*.csv")
    for file in csv_files:
        process_csv(file)