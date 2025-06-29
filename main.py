import numpy as np
import matplotlib.pyplot as plt

# --- Vehicle and Environmental Parameters ---
# Sourced from the analytical report tables and Section 2.4

# 2025 Hyundai Kona 1.6T 2WD
params_1_6T = {
    "name": "2025 Kona 1.6T 2WD (170 KM)",
    "mass_kg": 1450,  # Lighter than AWD version
    "drag_coeff": 0.27,
    "frontal_area_m2": 2.49,
    "tire_radius_m": 0.344,
    "drivetrain_efficiency": 0.92, # Higher efficiency for 2WD
    "gear_ratios": [4.717, 2.906, 1.864, 1.423, 1.224, 1.000, 0.790, 0.635],
    "final_drive": 3.510,
    "shift_time_s": 0.4,
    "shift_rpm": 6200,
    "color": "red"
}

# 2024 Hyundai Kona 1.0T FWD
params_1_0T = {
    "name": "2024 Kona 1.0T FWD (120 KM)",
    "mass_kg": 1410,
    "drag_coeff": 0.27,
    "frontal_area_m2": 2.45,
    "tire_radius_m": 0.343,
    "drivetrain_efficiency": 0.90, # Assumed 90% for FWD + DCT
    "gear_ratios": [3.643, 2.174, 1.826, 1.024, 0.809, 0.854, 0.717],
    # Dual final drive system: one for gears 1,2,4,5 and another for 3,6,7
    "final_drive": {1: 4.643, 2: 4.643, 3: 3.611, 4: 4.643, 5: 4.643, 6: 3.611, 7: 3.611},
    "shift_time_s": 0.25,
    "shift_rpm": 6200,
    "color": "grey"
}

# Environmental Constants
rho_air_kg_m3 = 1.225  # Air density at sea level
g_ms2 = 9.81          # Acceleration due to gravity
Crr = 0.012           # Coefficient of rolling resistance for typical tires

# --- Engine Performance Curve Synthesis ---
# Based on Section 2.3 of the report, creating a plausible curve shape
# between the manufacturer's stated peak torque/power points.

def get_engine_torque_1_6T(rpm):
    """
    CALIBRATED torque curve for the 1.6T engine.
    Tuned to achieve ~8.5s 0-100km/h time and 125.0 kW (170 HP) at 6000 RPM.
    """
    if rpm < 1900:
        # Even slower ramp-up to peak torque
        return np.interp(rpm, [1000, 1900], [105, 218])
    elif rpm <= 4000:
        # Further reduced peak torque plateau
        return 218
    elif rpm <= 6500:
        # Gradual drop-off, calibrated for 125 kW at 6000 RPM
        return np.interp(rpm, [4000, 6000, 6500], [218, 199, 168])
    else:
        return 0

def get_engine_torque_1_0T(rpm):
    """
    CALIBRATED torque curve for the 1.0T engine based on datasheet.
    From the chart: ~20.4 kgm peak torque (200 Nm) from 2000-3500 RPM
    """
    if rpm < 2000:
        # Linear ramp-up to peak torque at 2000 RPM
        return np.interp(rpm, [1000, 2000], [100, 200])
    elif rpm <= 3500:
        # Flat torque plateau at peak
        return 200
    elif rpm <= 6000:
        # Gradual drop-off after peak torque
        return np.interp(rpm, [3500, 6000], [200, 120])
    else:
        return 80

def calculate_power_kw(torque_nm, rpm):
    """Calculates power in kW from torque in Nm and RPM. This is a fundamental physics formula."""
    if rpm == 0:
        return 0
    # Power (Watts) = Torque (Nm) * Angular Velocity (rad/s)
    # Angular Velocity = RPM * 2 * pi / 60
    return (torque_nm * rpm * 2 * np.pi / 60) / 1000

# --- Plot 1: Engine Performance Curves (Figure 1 from report) ---
def plot_engine_curves():
    rpm_range = np.linspace(1000, 6500, 500)
    
    torque_1_6T = [get_engine_torque_1_6T(rpm) for rpm in rpm_range]
    power_1_6T = [calculate_power_kw(torque, rpm) for torque, rpm in zip(torque_1_6T, rpm_range)]
    
    torque_1_0T = [get_engine_torque_1_0T(rpm) for rpm in rpm_range]
    power_1_0T = [calculate_power_kw(torque, rpm) for torque, rpm in zip(torque_1_0T, rpm_range)]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel('Engine Speed (RPM)')
    ax1.set_ylabel('Torque (Nm)', color='tab:blue')
    ax1.plot(rpm_range, torque_1_6T, color=params_1_6T["color"], linestyle='-', label='1.6T Torque')
    ax1.plot(rpm_range, torque_1_0T, color=params_1_0T["color"], linestyle='-', label='1.0T Torque')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (kW)', color='tab:red')
    ax2.plot(rpm_range, power_1_6T, color=params_1_6T["color"], linestyle='--', label='1.6T Power')
    ax2.plot(rpm_range, power_1_0T, color=params_1_0T["color"], linestyle='--', label='1.0T Power')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle('Slika 1: Sintetizirane krivulje zmogljivosti motorja', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9,0.85))
    plt.show()

# --- Vehicle Dynamics Simulation ---
def simulate_acceleration(params, engine_torque_func, target_speed_kmh=160):
    """Simulates 0-target km/h acceleration based on vehicle parameters."""
    m, Cd, A, r_wheel, eta_dt, gear_ratios, shift_rpm, shift_time = \
        params["mass_kg"], params["drag_coeff"], params["frontal_area_m2"], \
        params["tire_radius_m"], params["drivetrain_efficiency"], params["gear_ratios"], \
        params["shift_rpm"], params["shift_time_s"]
    
    final_drive_map = params["final_drive"] if isinstance(params["final_drive"], dict) else {i+1: params["final_drive"] for i in range(len(gear_ratios))}
    
    dt, v_ms, t, current_gear = 0.01, 0.01, 0.0, 1
    time_data, speed_data_kmh, rpm_data, gear_data = [], [], [], []

    while v_ms * 3.6 < target_speed_kmh:
        overall_ratio = gear_ratios[current_gear - 1] * final_drive_map[current_gear]
        rpm = (v_ms * 60) / (2 * np.pi * r_wheel) * overall_ratio
        
        if rpm >= shift_rpm and current_gear < len(gear_ratios):
            t += shift_time
            current_gear += 1
            overall_ratio = gear_ratios[current_gear - 1] * final_drive_map[current_gear]
            rpm = (v_ms * 60) / (2 * np.pi * r_wheel) * overall_ratio

        T_eng = engine_torque_func(rpm)
        F_tractive = (T_eng * overall_ratio * eta_dt) / r_wheel
        F_aero = 0.5 * rho_air_kg_m3 * Cd * A * v_ms**2
        F_rolling = Crr * m * g_ms2
        F_net = F_tractive - (F_aero + F_rolling)
        
        if F_net < 0: F_net = 0

        acceleration = F_net / m
        v_ms += acceleration * dt
        t += dt
        
        time_data.append(t)
        speed_data_kmh.append(v_ms * 3.6)
        rpm_data.append(rpm)
        gear_data.append(current_gear)

    return time_data, speed_data_kmh, rpm_data, gear_data

# Run simulations for both vehicles
sim_results_1_6T = simulate_acceleration(params_1_6T, get_engine_torque_1_6T)
sim_results_1_0T = simulate_acceleration(params_1_0T, get_engine_torque_1_0T)

# --- Plot 2: Speed vs. Time (Figure 2 from report) ---
def plot_acceleration_curves():
    plt.figure(figsize=(12, 7))
    plt.plot(sim_results_1_6T[0], sim_results_1_6T[1], label=params_1_6T["name"], color=params_1_6T["color"], linewidth=2)
    plt.plot(sim_results_1_0T[0], sim_results_1_0T[1], label=params_1_0T["name"], color=params_1_0T["color"], linewidth=2)
    
    # Calculate intersection times for all speed markers
    time_100_1_6T = np.interp(100, sim_results_1_6T[1], sim_results_1_6T[0])
    time_100_1_0T = np.interp(100, sim_results_1_0T[1], sim_results_1_0T[0])
    time_120_1_6T = np.interp(120, sim_results_1_6T[1], sim_results_1_6T[0])
    time_120_1_0T = np.interp(120, sim_results_1_0T[1], sim_results_1_0T[0])
    time_130_1_6T = np.interp(130, sim_results_1_6T[1], sim_results_1_6T[0])
    time_130_1_0T = np.interp(130, sim_results_1_0T[1], sim_results_1_0T[0])
    time_140_1_6T = np.interp(140, sim_results_1_6T[1], sim_results_1_6T[0])
    time_140_1_0T = np.interp(140, sim_results_1_0T[1], sim_results_1_0T[0])
    
    # Add horizontal lines and annotations
    plt.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    plt.text(time_100_1_6T + 0.2, 98, f'{time_100_1_6T:.1f}s', va='top', ha='left', color=params_1_6T["color"], weight='bold')
    plt.text(time_100_1_0T + 0.2, 98, f'{time_100_1_0T:.1f}s', va='top', ha='left', color=params_1_0T["color"], weight='bold')
    
    plt.axhline(120, color='gray', linestyle='--', linewidth=0.8)
    plt.text(time_120_1_6T + 0.2, 118, f'{time_120_1_6T:.1f}s', va='top', ha='left', color=params_1_6T["color"], weight='bold')
    plt.text(time_120_1_0T + 0.2, 118, f'{time_120_1_0T:.1f}s', va='top', ha='left', color=params_1_0T["color"], weight='bold')
    
    plt.axhline(130, color='gray', linestyle='--', linewidth=0.8)
    plt.text(time_130_1_6T + 0.2, 128, f'{time_130_1_6T:.1f}s', va='top', ha='left', color=params_1_6T["color"], weight='bold')
    plt.text(time_130_1_0T + 0.2, 128, f'{time_130_1_0T:.1f}s', va='top', ha='left', color=params_1_0T["color"], weight='bold')
    
    plt.axhline(140, color='gray', linestyle='--', linewidth=0.8)
    plt.text(time_140_1_6T + 0.2, 138, f'{time_140_1_6T:.1f}s', va='top', ha='left', color=params_1_6T["color"], weight='bold')
    plt.text(time_140_1_0T + 0.2, 138, f'{time_140_1_0T:.1f}s', va='top', ha='left', color=params_1_0T["color"], weight='bold')
    
    plt.title('Slika 2: Simulacija pospeška (0-160 km/h)', fontsize=16)
    plt.xlabel('Čas (sekunde)')
    plt.ylabel('Hitrost (km/h)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xlim(0, max(sim_results_1_0T[0]))
    plt.ylim(0, 165)
    plt.show()

# --- Plot 3: Tractive Effort vs. Speed (Figure 3 from report) ---
def plot_tractive_effort():
    speed_range_kmh = np.linspace(1, 150, 200)
    speed_range_ms = speed_range_kmh / 3.6
    avg_mass = (params_1_6T["mass_kg"] + params_1_0T["mass_kg"]) / 2
    F_resist = (0.5 * rho_air_kg_m3 * params_1_6T["drag_coeff"] * params_1_6T["frontal_area_m2"] * speed_range_ms**2) + (Crr * avg_mass * g_ms2)

    plt.figure(figsize=(14, 8))
    plt.plot(speed_range_kmh, F_resist, label='Resistive Forces (Aero + Rolling)', color='red', linewidth=2.5, zorder=10)

    for gear in range(1, 6): # Plot first 5 gears
        # 1.6T
        overall_ratio_16 = params_1_6T["gear_ratios"][gear-1] * params_1_6T["final_drive"]
        rpm_in_gear_16 = (speed_range_ms * 60) / (2 * np.pi * params_1_6T["tire_radius_m"]) * overall_ratio_16
        torque_in_gear_16 = np.array([get_engine_torque_1_6T(rpm) for rpm in rpm_in_gear_16])
        tractive_force_16 = (torque_in_gear_16 * overall_ratio_16 * params_1_6T["drivetrain_efficiency"]) / params_1_6T["tire_radius_m"]
        valid_range_16 = (rpm_in_gear_16 > 1000) & (rpm_in_gear_16 <= params_1_6T["shift_rpm"])
        plt.plot(speed_range_kmh[valid_range_16], tractive_force_16[valid_range_16], color=params_1_6T["color"], linestyle='-', label=f'1.6T' if gear==1 else "")

        # 1.0T
        final_drive_10 = params_1_0T["final_drive"][gear]
        overall_ratio_10 = params_1_0T["gear_ratios"][gear-1] * final_drive_10
        rpm_in_gear_10 = (speed_range_ms * 60) / (2 * np.pi * params_1_0T["tire_radius_m"]) * overall_ratio_10
        torque_in_gear_10 = np.array([get_engine_torque_1_0T(rpm) for rpm in rpm_in_gear_10])
        tractive_force_10 = (torque_in_gear_10 * overall_ratio_10 * params_1_0T["drivetrain_efficiency"]) / params_1_0T["tire_radius_m"]
        valid_range_10 = (rpm_in_gear_10 > 1000) & (rpm_in_gear_10 <= params_1_0T["shift_rpm"])
        plt.plot(speed_range_kmh[valid_range_10], tractive_force_10[valid_range_10], color=params_1_0T["color"], linestyle='--', label=f'1.0T' if gear==1 else "")

    plt.title('Slika 3: Vlečna sila v odvisnosti od hitrosti vozila', fontsize=16)
    plt.xlabel('Hitrost (km/h)')
    plt.ylabel('Sila (Newton)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.ylim(0, 18000)
    plt.xlim(0, 150)
    plt.show()

# --- Plot 4: Engine RPM at Cruising Speed (Table 4 from report) ---
def plot_cruising_rpm():
    speed_kmh = np.linspace(60, 130, 100)
    speed_ms = speed_kmh / 3.6

    # 1.6T in 8th gear
    overall_ratio_1_6T = params_1_6T["gear_ratios"][-1] * params_1_6T["final_drive"]
    rpm_1_6T = (speed_ms * 60) / (2 * np.pi * params_1_6T["tire_radius_m"]) * overall_ratio_1_6T

    # 1.0T in 7th gear
    overall_ratio_1_0T = params_1_0T["gear_ratios"][-1] * params_1_0T["final_drive"][7]
    rpm_1_0T = (speed_ms * 60) / (2 * np.pi * params_1_0T["tire_radius_m"]) * overall_ratio_1_0T

    plt.figure(figsize=(12, 7))
    plt.plot(speed_kmh, rpm_1_6T, label=f'{params_1_6T["name"]} (8th Gear)', color=params_1_6T["color"])
    plt.plot(speed_kmh, rpm_1_0T, label=f'{params_1_0T["name"]} (7th Gear)', color=params_1_0T["color"])
    
    rpm_at_120_1_6T = np.interp(120, speed_kmh, rpm_1_6T)
    rpm_at_120_1_0T = np.interp(120, speed_kmh, rpm_1_0T)
    plt.axvline(120, color='red', linestyle=':', linewidth=1)
    plt.scatter([120, 120], [rpm_at_120_1_6T, rpm_at_120_1_0T], color='red', zorder=5)
    plt.text(120.5, rpm_at_120_1_6T, f' {rpm_at_120_1_6T:.0f} RPM', ha='left', va='center', color=params_1_6T["color"])
    plt.text(120.5, rpm_at_120_1_0T, f' {rpm_at_120_1_0T:.0f} RPM', ha='left', va='center', color=params_1_0T["color"])

    plt.title('Slika 4: Vrtljaji motorja pri vožnji po avtocesti (najvišja prestava)', fontsize=16)
    plt.xlabel('Hitrost (km/h)')
    plt.ylabel('Vrtljaji motorja (RPM)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    print("Ustvarjanje grafov zmogljivosti za Hyundai Kona 1.6T 2WD vs 1.0T FWD...")
    # To run this script, you would typically call these functions.
    # For example:
    plot_engine_curves()
    plot_acceleration_curves()
    plot_tractive_effort()
    plot_cruising_rpm()
    print("Ustvarjanje grafov končano.")
