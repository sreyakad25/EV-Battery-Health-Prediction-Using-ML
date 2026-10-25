import os
import json
import joblib
import pandas as pd
import numpy as np

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')

class MLPredictionService:
    _instance = None
    _models = {}
    _metrics = None
    _feature_importance = None
    _scatter_sample = None
    _distribution_sample = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_artifacts()
        return cls._instance

    def load_artifacts(self):
        model_files = {
            'Random Forest': 'model_random_forest.joblib',
            'Linear Regression': 'model_linear_regression.joblib',
            'Decision Tree': 'model_decision_tree.joblib'
        }
        for name, filename in model_files.items():
            path = os.path.join(ARTIFACTS_DIR, filename)
            if os.path.exists(path):
                try:
                    self._models[name] = joblib.load(path)
                except Exception as e:
                    print(f"Error loading {name}: {e}")

        metrics_path = os.path.join(ARTIFACTS_DIR, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self._metrics = json.load(f)

        fi_path = os.path.join(ARTIFACTS_DIR, 'feature_importance.json')
        if os.path.exists(fi_path):
            with open(fi_path, 'r') as f:
                self._feature_importance = json.load(f)

        sc_path = os.path.join(ARTIFACTS_DIR, 'scatter_sample.json')
        if os.path.exists(sc_path):
            with open(sc_path, 'r') as f:
                self._scatter_sample = json.load(f)

        dist_path = os.path.join(ARTIFACTS_DIR, 'distribution_sample.json')
        if os.path.exists(dist_path):
            with open(dist_path, 'r') as f:
                self._distribution_sample = json.load(f)

    def get_available_models(self):
        return list(self._models.keys())

    def get_metrics(self):
        return self._metrics or {}

    def get_feature_importance(self):
        return self._feature_importance or {}

    def get_scatter_sample(self):
        return self._scatter_sample or []

    def get_distribution_sample(self):
        return self._distribution_sample or {'bins': [], 'counts': []}

    def predict(self, input_data, model_name='Random Forest'):
        if not self._models:
            self.load_artifacts()

        if model_name not in self._models:
            model_name = 'Random Forest' if 'Random Forest' in self._models else list(self._models.keys())[0]

        pipeline = self._models[model_name]

        # Vehicle & Battery Chemistry Mapping
        car_model = input_data.get('car_model') or input_data.get('vehicle_model') or 'Tesla Model 3'
        battery_type = input_data.get('battery_type') or input_data.get('battery_chemistry')
        if not battery_type:
            if 'BYD' in car_model:
                battery_type = 'LFP'
            else:
                battery_type = 'NMC'

        # Battery Capacity defaults if omitted in Simple Mode
        capacity_defaults = {
            'Tesla Model 3': 60.0,
            'BYD Atto 3': 60.5,
            'Nissan Leaf': 40.0,
            'Hyundai Kona': 64.0,
            'Chevrolet Bolt EV': 65.0,
            'BMW i3': 42.2,
            'Volkswagen ID.4': 77.0
        }
        try:
            battery_capacity = float(input_data.get('battery_capacity') or input_data.get('battery_capacity_kwh') or capacity_defaults.get(car_model, 60.0))
        except (ValueError, TypeError):
            battery_capacity = capacity_defaults.get(car_model, 60.0)

        # Vehicle Age in Months
        try:
            raw_age = float(input_data.get('vehicle_age') or input_data.get('vehicle_age_months') or input_data.get('vehicle_age_years') or 3.0)
            vehicle_age_months = raw_age * 12.0 if raw_age <= 15.0 else raw_age
        except (ValueError, TypeError):
            vehicle_age_months = 36.0

        # Cycles / Mileage conversion
        raw_mileage = input_data.get('odometer_km')
        if raw_mileage:
            try:
                charging_cycles = float(raw_mileage) / 250.0 # ~250 km per cycle
            except (ValueError, TypeError):
                charging_cycles = (vehicle_age_months / 12.0) * 160.0
        else:
            try:
                charging_cycles = float(input_data.get('charging_cycles') or input_data.get('total_charging_cycles') or (vehicle_age_months * 14.0))
            except (ValueError, TypeError):
                charging_cycles = 500.0

        # Temperature
        try:
            temperature = float(input_data.get('temperature') or input_data.get('avg_temperature_c') or 25.0)
        except (ValueError, TypeError):
            temperature = 25.0

        # Fast Charge Ratio
        fast_charge_habit = input_data.get('charging_habit')
        if fast_charge_habit == 'rare':
            fast_charge_ratio = 0.10
        elif fast_charge_habit == 'frequent':
            fast_charge_ratio = 0.65
        elif fast_charge_habit == 'balanced':
            fast_charge_ratio = 0.30
        else:
            try:
                raw_fcr = float(input_data.get('fast_charge_ratio') or 20.0)
                fast_charge_ratio = raw_fcr / 100.0 if raw_fcr > 1.0 else raw_fcr
            except (ValueError, TypeError):
                fast_charge_ratio = 0.20

        # Driving Style
        raw_driving_style = (input_data.get('driving_style') or 'Moderate').capitalize()
        style_mapping = {
            'Normal': 'Moderate',
            'Eco': 'Conservative',
            'Aggressive': 'Aggressive',
            'Moderate': 'Moderate',
            'Conservative': 'Conservative'
        }
        driving_style = style_mapping.get(raw_driving_style, 'Moderate')

        # Discharge Rate inference if in Simple Mode
        if input_data.get('discharge_rate'):
            try:
                discharge_rate = float(input_data['discharge_rate'])
            except (ValueError, TypeError):
                discharge_rate = 1.5
        else:
            rate_defaults = {'Conservative': 1.05, 'Moderate': 1.50, 'Aggressive': 2.45}
            discharge_rate = rate_defaults.get(driving_style, 1.50)

        # Internal Resistance inference if omitted in Simple Mode
        raw_ir = input_data.get('internal_resistance')
        if raw_ir and float(raw_ir) > 0:
            try:
                val = float(raw_ir)
                internal_resistance = val / 1000.0 if val >= 1.0 else val
            except (ValueError, TypeError):
                internal_resistance = 0.050
        else:
            # Intelligent empirical estimation:
            base_ir = 0.024 if battery_type == 'LFP' else 0.020
            cycle_ir_increase = (charging_cycles / 2000.0) * 0.038
            temp_ir_factor = max(0.0, (temperature - 25.0) * 0.0004)
            internal_resistance = round(float(np.clip(base_ir + cycle_ir_increase + temp_ir_factor, 0.018, 0.088)), 4)

        input_df = pd.DataFrame([{
            'Car_Model': car_model,
            'Battery_Type': battery_type,
            'Driving_Style': driving_style,
            'Battery_Capacity_kWh': battery_capacity,
            'Vehicle_Age_Months': vehicle_age_months,
            'Total_Charging_Cycles': charging_cycles,
            'Avg_Temperature_C': temperature,
            'Fast_Charge_Ratio': fast_charge_ratio,
            'Avg_Discharge_Rate_C': discharge_rate,
            'Internal_Resistance_Ohm': internal_resistance
        }])

        prediction = pipeline.predict(input_df)[0]
        soh = round(float(np.clip(prediction, 40.0, 100.0)), 2)

        # Actionable Health Grade & Summary
        if soh >= 90.0:
            letter_grade = 'A'
            grade_title = 'Grade A — Excellent Condition'
            status = 'Optimal Health'
            status_color = 'success'
            description = (
                f"Battery State of Health is pristine at {soh}%. Capacity retention is near factory specification with negligible cyclic degradation."
            )
        elif soh >= 80.0:
            letter_grade = 'B'
            grade_title = 'Grade B — Good Health'
            status = 'Good Condition'
            status_color = 'success'
            description = (
                f"Battery is in dependable operational condition with {soh}% State of Health. Normal aging observed, providing robust everyday driving range."
            )
        elif soh >= 70.0:
            letter_grade = 'C'
            grade_title = 'Grade C — Fair / Moderate Wear'
            status = 'Moderate Degradation'
            status_color = 'warning'
            description = (
                f"Battery exhibits moderate wear ({soh}% SoH). High temperatures and rapid DC charging should be minimized to protect remaining cell life."
            )
        else:
            letter_grade = 'D'
            grade_title = 'Grade D — Service Recommended'
            status = 'Degradation Alert'
            status_color = 'danger'
            description = (
                f"Battery capacity has degraded to {soh}%, below the standard 70% retention threshold. Noticeable range loss expected. Authorized workshop diagnostics recommended."
            )

        # Usable energy & estimated range
        usable_kwh = round(battery_capacity * (soh / 100.0), 1)
        full_range_km = int(battery_capacity * 5.8)
        current_range_km = int(usable_kwh * 5.8)

        # Actionable Battery Care Tips
        care_tips = []
        if battery_type == 'NMC':
            care_tips.append("Set your daily charge limit to 80% and only charge to 100% for long road trips to protect NMC cathode longevity.")
        else:
            care_tips.append("LFP batteries benefit from periodic 100% charging once a week to calibrate the Battery Management System (BMS).")

        if fast_charge_ratio > 0.35:
            care_tips.append("Frequent DC fast charging accelerates internal resistance growth; utilize AC home charging when parked overnight.")

        if temperature > 30:
            care_tips.append("Avoid parking in direct hot sunlight immediately after fast charging; pre-cool the cabin and pack while plugged in.")

        return {
            'predicted_soh': soh,
            'letter_grade': letter_grade,
            'grade_title': grade_title,
            'model_name': model_name,
            'health_status': status,
            'status_color': status_color,
            'description': description,
            'usable_kwh': usable_kwh,
            'original_kwh': battery_capacity,
            'current_range_km': current_range_km,
            'full_range_km': full_range_km,
            'care_tips': care_tips,
            'inputs': {
                'battery_capacity': battery_capacity,
                'vehicle_age_years': round(vehicle_age_months / 12.0, 1),
                'vehicle_age_months': round(vehicle_age_months),
                'charging_cycles': int(charging_cycles),
                'temperature': temperature,
                'fast_charge_ratio': round(fast_charge_ratio * 100, 1),
                'discharge_rate': discharge_rate,
                'internal_resistance_mohm': round(internal_resistance * 1000, 1),
                'internal_resistance_ohm': round(internal_resistance, 4),
                'driving_style': driving_style,
                'battery_chemistry': battery_type,
                'vehicle_model': car_model
            }
        }
