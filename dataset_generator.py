import numpy as np
import pandas as pd
import os

def generate_ev_battery_dataset(output_path='data/ev_battery_degradation.csv', n_samples=10000, seed=42):
    np.random.seed(seed)
    
    vehicle_ids = [f"EV_{i+1:05d}" for i in range(n_samples)]
    
    car_models = [
        'Tesla Model 3', 'BYD Atto 3', 'Nissan Leaf', 'Hyundai Kona', 
        'Chevrolet Bolt EV', 'BMW i3', 'Volkswagen ID.4'
    ]
    model_choice = np.random.choice(car_models, size=n_samples, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07])
    
    # Battery chemistry: LFP vs NMC
    # Certain models often use LFP (e.g. BYD, standard Tesla Model 3), others NMC
    battery_types = []
    for model in model_choice:
        if model in ['BYD Atto 3']:
            battery_types.append(np.random.choice(['LFP', 'NMC'], p=[0.85, 0.15]))
        elif model in ['Tesla Model 3']:
            battery_types.append(np.random.choice(['LFP', 'NMC'], p=[0.50, 0.50]))
        else:
            battery_types.append(np.random.choice(['LFP', 'NMC'], p=[0.20, 0.80]))
    battery_types = np.array(battery_types)
    
    # Battery capacity: 40 kWh to 100 kWh
    capacity_map = {
        'Tesla Model 3': (55, 82),
        'BYD Atto 3': (50, 60),
        'Nissan Leaf': (40, 62),
        'Hyundai Kona': (39, 64),
        'Chevrolet Bolt EV': (60, 66),
        'BMW i3': (42, 45),
        'Volkswagen ID.4': (58, 82)
    }
    battery_capacity = np.array([
        np.round(np.random.uniform(*capacity_map[m]), 1) for m in model_choice
    ])
    
    # Vehicle Age in Months (6 to 96 months, approx 0.5 to 8 years)
    vehicle_age_months = np.random.randint(6, 97, size=n_samples)
    
    # Total Charging Cycles: roughly 10 - 25 cycles per month with noise
    cycles_per_month = np.random.uniform(8, 22, size=n_samples)
    total_cycles = np.clip(
        (vehicle_age_months * cycles_per_month + np.random.normal(0, 50, size=n_samples)).astype(int),
        50, 2200
    )
    
    # Average Temperature (°C): 12°C to 42°C, normal dist around 26°C
    avg_temperature_c = np.round(np.clip(np.random.normal(26.5, 6.5, size=n_samples), 12.0, 44.0), 1)
    
    # Fast Charge Ratio: 0.05 to 0.85
    fast_charge_ratio = np.round(np.clip(np.random.beta(2, 5, size=n_samples), 0.05, 0.88), 2)
    
    # Driving style
    driving_styles = np.random.choice(['Conservative', 'Moderate', 'Aggressive'], size=n_samples, p=[0.30, 0.50, 0.20])
    
    # Average Discharge Rate (C-rate): Conservative ~0.8-1.5, Moderate ~1.2-2.2, Aggressive ~1.8-3.2
    discharge_rates = []
    for style in driving_styles:
        if style == 'Conservative':
            discharge_rates.append(np.random.uniform(0.7, 1.4))
        elif style == 'Moderate':
            discharge_rates.append(np.random.uniform(1.2, 2.0))
        else:
            discharge_rates.append(np.random.uniform(1.8, 3.2))
    avg_discharge_rate_c = np.round(np.array(discharge_rates), 2)
    
    # Semi-empirical degradation calculation (Physics-based simulation)
    # Chemistry durability factor: LFP has longer cycle life (degrades ~35% slower per cycle)
    chem_factor = np.where(battery_types == 'LFP', 0.68, 1.0)
    
    # Cycle aging impact: typical battery loses ~20% SoH over 1500-2000 cycles
    cycle_deg = (total_cycles / 2000.0) * 16.0 * chem_factor
    
    # Arrhenius thermal stress impact: high temp accelerates SEI growth
    # Reference temp = 22°C; above 28°C degradation increases significantly
    temp_deg = np.maximum(0, (avg_temperature_c - 24.0) * 0.45)
    
    # Fast charge stress (DC fast charge causes lithium plating and thermal gradients)
    fast_charge_deg = fast_charge_ratio * 5.5 * chem_factor
    
    # Discharge stress (aggressive high C-rate driving)
    discharge_deg = (avg_discharge_rate_c - 1.0) * 2.2
    
    # Calendar aging: small calendar loss per month
    calendar_deg = (vehicle_age_months / 12.0) * 1.2
    
    total_degradation = cycle_deg + temp_deg + fast_charge_deg + discharge_deg + calendar_deg
    # Add small measurement noise
    noise = np.random.normal(0, 0.75, size=n_samples)
    
    # State of Health percentage: base 100% minus degradation
    soh_percent = np.clip(np.round(100.0 - total_degradation + noise, 2), 52.0, 99.8)
    
    # Internal Resistance (Ohm): base resistance ~ 0.022 Ohm (22 mΩ), rises as SoH degrades
    base_ir = np.where(battery_types == 'LFP', 0.024, 0.020)
    ir_increase = ((100.0 - soh_percent) / 45.0) * 0.045
    ir_noise = np.random.normal(0, 0.002, size=n_samples)
    internal_resistance_ohm = np.round(np.clip(base_ir + ir_increase + ir_noise, 0.016, 0.092), 4)
    
    # Battery Status label (Healthy vs Replace Required)
    battery_status = np.where(soh_percent >= 80.0, 'Healthy', 'Replace Required')
    
    df = pd.DataFrame({
        'Vehicle_ID': vehicle_ids,
        'Car_Model': model_choice,
        'Battery_Type': battery_types,
        'Battery_Capacity_kWh': battery_capacity,
        'Vehicle_Age_Months': vehicle_age_months,
        'Total_Charging_Cycles': total_cycles,
        'Avg_Temperature_C': avg_temperature_c,
        'Fast_Charge_Ratio': fast_charge_ratio,
        'Avg_Discharge_Rate_C': avg_discharge_rate_c,
        'Driving_Style': driving_styles,
        'Internal_Resistance_Ohm': internal_resistance_ohm,
        'SoH_Percent': soh_percent,
        'Battery_Status': battery_status
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} samples with 13 columns to {output_path}")
    print(df.head())
    print("\nSummary Statistics of SoH_Percent:")
    print(df['SoH_Percent'].describe())
    return df

if __name__ == '__main__':
    generate_ev_battery_dataset()
