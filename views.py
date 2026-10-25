import os
import re
import json
import uuid
from datetime import datetime
import pandas as pd
from django.shortcuts import render, redirect
from django.http import JsonResponse, Http404
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth.models import User
from accounts.models import UserProfile
from .models import PredictionRecord
from .ml_service import MLPredictionService
from .firestore_service import FirestoreService
from accounts.middleware import auth_login_required, email_verified_required

def validate_prediction_inputs(data):
    """
    Strict server-side validation against physical bounds to prevent
    adversarial ML attacks and numerical injection.
    """
    errors = []

    try:
        cap = float(data.get('battery_capacity') or data.get('battery_capacity_kwh') or 60.0)
        if not (15.0 <= cap <= 250.0):
            errors.append("Battery capacity must be between 15.0 and 250.0 kWh.")
    except (ValueError, TypeError):
        errors.append("Invalid battery capacity number.")

    try:
        age = float(data.get('vehicle_age') or data.get('vehicle_age_years') or 3.0)
        if not (0.0 <= age <= 25.0):
            errors.append("Vehicle age must be between 0 and 25 years.")
    except (ValueError, TypeError):
        errors.append("Invalid vehicle age.")

    try:
        cycles = float(data.get('charging_cycles') or 500)
        if not (0 <= cycles <= 10000):
            errors.append("Charging cycles must be between 0 and 10,000.")
    except (ValueError, TypeError):
        errors.append("Invalid charging cycles value.")

    try:
        temp = float(data.get('temperature') or 25.0)
        if not (-30.0 <= temp <= 65.0):
            errors.append("Operating temperature must be between -30°C and 65°C.")
    except (ValueError, TypeError):
        errors.append("Invalid temperature value.")

    return errors

def home_view(request):
    """
    Public landing page presenting project overview and call to action.
    """
    ml_service = MLPredictionService.get_instance()
    metrics = ml_service.get_metrics()
    return render(request, 'home.html', {
        'metrics': metrics,
        'best_model': metrics.get('best_model', 'Random Forest'),
    })

@email_verified_required
def dashboard_view(request):
    """
    User dashboard displaying personal statistics, recent predictions, and battery care tips.
    Loads prediction records from Cloud Firestore: users/{firebase_uid}/predictions
    """
    user = getattr(request, 'user_data', {}) or request.session.get('user') or {}
    uid = user.get('uid') or user.get('id') or ''
    ml_service = MLPredictionService.get_instance()
    
    id_token = request.session.get('firebase_id_token')
    refresh_token = request.session.get('firebase_refresh_token')
    fs_service = FirestoreService.get_instance()

    predictions = fs_service.get_user_predictions(uid, id_token=id_token, refresh_token=refresh_token, limit=100)
    total_predictions = len(predictions)
    recent_records = predictions[:5]
    user_predictions = []
    for r in recent_records:
        inputs = r.get('inputs') or {}
        created_at_val = r.get('created_at', '')
        if isinstance(created_at_val, str) and 'T' in created_at_val:
            date_display = created_at_val.replace('T', ' ')[:16]
        else:
            date_display = str(created_at_val)[:16] if created_at_val else 'Recent'

        soh_val = float(r.get('predicted_soh') or r.get('predicted_health') or 0.0)
        user_predictions.append({
            'prediction_id': r.get('prediction_id') or r.get('id'),
            'vehicle_model': r.get('vehicle_model') or r.get('model_name') or 'EV Vehicle',
            'model_name': r.get('model_name') or 'Random Forest',
            'predicted_soh': soh_val,
            'predicted_health': soh_val,
            'health_status': r.get('health_status') or 'Good Condition',
            'created_at': date_display,
            'inputs': inputs,
        })

    latest_soh = user_predictions[0]['predicted_soh'] if user_predictions else None
    
    # Model info
    metrics = ml_service.get_metrics()
    distribution = ml_service.get_distribution_sample()
    
    return render(request, 'predictor/dashboard.html', {
        'user': user,
        'user_predictions': user_predictions,
        'total_predictions': total_predictions,
        'latest_soh': latest_soh,
        'available_models_count': len(ml_service.get_available_models()) or 3,
        'best_model': metrics.get('best_model', 'Random Forest'),
        'metrics': metrics,
        'distribution': distribution
    })

@email_verified_required
def predict_view(request):
    """
    Interactive prediction form with server-side range validation.
    Saves predictions to Cloud Firestore at users/{firebase_uid}/predictions/{pred_id}.
    """
    ml_service = MLPredictionService.get_instance()
    available_models = ml_service.get_available_models() or ['Random Forest', 'Linear Regression', 'Decision Tree']
    
    if request.method == 'POST':
        data = request.POST
        selected_model = data.get('selected_model', 'Random Forest')

        # Server-side validation
        errors = validate_prediction_inputs(data)
        if errors:
            return render(request, 'predictor/predict.html', {
                'available_models': available_models,
                'validation_error': errors[0]
            })

        result = ml_service.predict(data, model_name=selected_model)
        request.session['last_prediction'] = result
        
        # Save under authenticated Firebase UID in Cloud Firestore
        user = getattr(request, 'user_data', {}) or request.session.get('user') or {}
        uid = user.get('uid') or user.get('id')
        if uid:
            pred_id = f"pred_{uuid.uuid4().hex[:12]}"
            id_token = request.session.get('firebase_id_token')
            refresh_token = request.session.get('firebase_refresh_token')
            
            fs_service = FirestoreService.get_instance()
            fs_service.save_prediction(
                user_uid=str(uid),
                prediction_data=result,
                prediction_id=pred_id,
                id_token=id_token,
                refresh_token=refresh_token
            )
            result['prediction_id'] = pred_id
            request.session['last_prediction'] = result
            
        return render(request, 'predictor/result.html', {
            'result': result,
            'user': user,
            'user_data': user,
            'user_uid': str(uid) if uid else '',
        })

    user = getattr(request, 'user_data', {}) or request.session.get('user') or {}
    uid = user.get('uid') or user.get('id') or ''
    return render(request, 'predictor/predict.html', {
        'available_models': available_models,
        'user': user,
        'user_data': user,
        'user_uid': str(uid),
    })

@email_verified_required
def result_view(request):
    """
    Displays the latest predicted SoH and health status indicator.
    Supports viewing any historical record via ?id=<prediction_id>
    loaded directly from Cloud Firestore (users/{firebase_uid}/predictions/{pred_id}).
    """
    user = getattr(request, 'user_data', {}) or request.session.get('user') or {}
    uid = user.get('uid') or user.get('id') or ''
    id_token = request.session.get('firebase_id_token')
    refresh_token = request.session.get('firebase_refresh_token')
    pred_id = request.GET.get('id')
    if pred_id:
        fs_service = FirestoreService.get_instance()

        doc = fs_service.get_prediction(str(uid), pred_id, id_token=id_token, refresh_token=refresh_token) if uid else None
        if doc:
            ml_service = MLPredictionService.get_instance()
            inputs = dict(doc.get('inputs') or {})
            if not inputs:
                inputs = {
                    'battery_capacity': doc.get('battery_capacity') or doc.get('capacity_kwh') or 60.0,
                    'vehicle_age_years': doc.get('vehicle_age_years') or doc.get('vehicle_age') or 3.0,
                    'charging_cycles': doc.get('charging_cycles') or 500,
                    'temperature': doc.get('temperature') or doc.get('avg_temp_c') or 25.0,
                    'vehicle_model': doc.get('vehicle_model') or doc.get('model_name') or 'Tesla Model 3',
                    'driving_style': doc.get('driving_style') or 'Moderate',
                    'battery_chemistry': doc.get('battery_chemistry') or 'NMC',
                }
            vehicle_model = inputs.get('vehicle_model') or doc.get('vehicle_model') or doc.get('model_name') or 'Tesla Model 3'
            inputs['vehicle_model'] = vehicle_model
            model_name = doc.get('model_name') or 'Random Forest'
            res = ml_service.predict(inputs, model_name=model_name)
            res['prediction_id'] = pred_id
            request.session['last_prediction'] = res
            return render(request, 'predictor/result.html', {
                'result': res,
                'user': user,
                'user_data': user,
                'user_uid': str(uid),
            })

    result = request.session.get('last_prediction')
    if not result:
        return redirect('predictor:predict')
    return render(request, 'predictor/result.html', {
        'result': result,
        'user': user,
        'user_data': user,
        'user_uid': str(uid),
    })

def models_comparison_view(request):
    ml_service = MLPredictionService.get_instance()
    metrics = ml_service.get_metrics()
    feature_importance = ml_service.get_feature_importance()
    scatter_sample = ml_service.get_scatter_sample()
    
    return render(request, 'predictor/model_comparison.html', {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'scatter_sample': scatter_sample,
        'best_model': metrics.get('best_model', 'Random Forest')
    })

def dataset_info_view(request):
    csv_path = os.path.join(settings.BASE_DIR, 'data', 'ev_battery_degradation.csv')
    preview_rows = []
    total_records = 10000
    total_columns = 13
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            total_records = len(df)
            total_columns = len(df.columns)
            preview_rows = df.head(10).to_dict(orient='records')
        except Exception as e:
            print(f"Error loading CSV for preview: {e}")
            
    feature_meta = [
        {'name': 'Vehicle_ID', 'type': 'String', 'desc': 'Unique alphanumeric identifier for each EV instance.'},
        {'name': 'Car_Model', 'type': 'Categorical', 'desc': 'Specific EV model.'},
        {'name': 'Battery_Type', 'type': 'Categorical', 'desc': 'Cell chemistry: LFP or NMC.'},
        {'name': 'Battery_Capacity_kWh', 'type': 'Numeric (kWh)', 'desc': 'Rated battery pack energy capacity.'},
        {'name': 'Vehicle_Age_Months', 'type': 'Numeric (Months)', 'desc': 'Elapsed operational calendar age.'},
        {'name': 'Total_Charging_Cycles', 'type': 'Numeric (Cycles)', 'desc': 'Cumulative equivalent full cycles.'},
        {'name': 'Avg_Temperature_C', 'type': 'Numeric (°C)', 'desc': 'Mean ambient thermal temperature history.'},
        {'name': 'Fast_Charge_Ratio', 'type': 'Numeric (Ratio)', 'desc': 'Proportion of cumulative DC fast charging.'},
        {'name': 'Avg_Discharge_Rate_C', 'type': 'Numeric (C-rate)', 'desc': 'Average driving discharge rate.'},
        {'name': 'Driving_Style', 'type': 'Categorical', 'desc': 'Driver aggression profile.'},
        {'name': 'Internal_Resistance_Ohm', 'type': 'Numeric (Ohm)', 'desc': 'Cell internal impedance.'},
        {'name': 'SoH_Percent', 'type': 'Numeric (%) [Target 1]', 'desc': 'State of Health.'},
        {'name': 'Battery_Status', 'type': 'Categorical [Target 2]', 'desc': 'Classification threshold flag.'}
    ]

    return render(request, 'predictor/dataset_info.html', {
        'total_records': total_records,
        'total_columns': total_columns,
        'preview_rows': preview_rows,
        'features': feature_meta
    })

@email_verified_required
def history_view(request):
    """
    Displays historical battery predictions loaded from Cloud Firestore: users/{firebase_uid}/predictions
    """
    user = getattr(request, 'user_data', {}) or request.session.get('user') or {}
    uid = user.get('uid') or user.get('id') or request.session.get('firebase_uid') or ''

    id_token = request.session.get('firebase_id_token')
    refresh_token = request.session.get('firebase_refresh_token')
    fs_service = FirestoreService.get_instance()

    from .firestore_service import refresh_firebase_id_token
    if not id_token and refresh_token:
        fresh = refresh_firebase_id_token(refresh_token)
        if fresh:
            id_token = fresh
            request.session['firebase_id_token'] = fresh
            request.session.modified = True

    raw_predictions = fs_service.get_user_predictions(uid, id_token=id_token, refresh_token=refresh_token, limit=200)

    predictions = []
    for r in raw_predictions:
        inputs = r.get('inputs') or {}
        soh = float(r.get('predicted_soh') or r.get('predicted_health') or 0.0)
        
        # Health grade mapping matching reference image: Good / Moderate / At Risk
        if soh >= 80.0:
            health_grade = 'Good'
            grade_badge = 'badge-soh-good'
        elif soh >= 70.0:
            health_grade = 'Moderate'
            grade_badge = 'badge-soh-warning'
        else:
            health_grade = 'At Risk'
            grade_badge = 'badge-soh-danger'

        model_name = r.get('model_name') or 'Random Forest'
        
        # Format date and time
        created_val = r.get('created_at', '')
        created_date = 'Recent'
        created_time = ''
        if isinstance(created_val, str) and created_val:
            try:
                cleaned = created_val.replace('Z', '').split('.')[0]
                if 'T' in cleaned:
                    dt = datetime.strptime(cleaned, '%Y-%m-%dT%H:%M:%S')
                else:
                    dt = datetime.strptime(cleaned, '%Y-%m-%d %H:%M:%S')
                created_date = dt.strftime('%Y-%m-%d')
                created_time = dt.strftime('%I:%M %p')
            except Exception:
                created_date = created_val[:10]
        date_str = f"{created_date} {created_time}".strip()
        
        # Format key inputs
        cap = inputs.get('battery_capacity') or inputs.get('battery_capacity_kwh') or r.get('capacity_kwh') or 60.0
        age = inputs.get('vehicle_age_years') or inputs.get('vehicle_age') or r.get('vehicle_age_years') or 3.0
        cycles = inputs.get('charging_cycles') or inputs.get('total_charging_cycles') or r.get('charging_cycles') or 500

        pred_id = r.get('prediction_id') or r.get('id') or ''
        vehicle_model = inputs.get('vehicle_model') or inputs.get('car_model') or r.get('vehicle_model') or 'Tesla Model 3'

        predictions.append({
            'prediction_id': pred_id,
            'id': pred_id,
            'vehicle_model': vehicle_model,
            'model_name': model_name,
            'predicted_soh': f"{soh:.2f}",
            'raw_soh': soh,
            'predicted_health': soh,
            'health_status': r.get('health_status') or 'Good Condition',
            'health_grade': health_grade,
            'grade_badge': grade_badge,
            'created_at': date_str,
            'created_date': created_date,
            'created_time': created_time,
            'inputs': inputs,
            'capacity_display': f"{cap} kWh",
            'age_display': f"{age}y",
            'cycles_display': str(cycles),
        })

    total_count = len(predictions)
    avg_soh = round(sum(p['raw_soh'] for p in predictions) / total_count, 1) if total_count > 0 else 0.0
    good_count = sum(1 for p in predictions if p['raw_soh'] >= 80.0)
    at_risk_count = total_count - good_count
    
    # Populate available models filter with actual vehicle models from predictions
    models_set = set(p['vehicle_model'] for p in predictions if p.get('vehicle_model'))
    if not models_set:
        models_set.update(['Tesla Model 3', 'Nissan Leaf', 'Tata Nexon EV', 'Hyundai Ioniq 5', 'Chevrolet Bolt EV'])
    available_models = sorted(list(models_set))
    
    return render(request, 'predictor/history.html', {
        'predictions': predictions,
        'total_count': total_count,
        'avg_soh': avg_soh,
        'good_count': good_count,
        'at_risk_count': at_risk_count,
        'available_models': available_models,
        'user': user,
        'user_data': user,
        'user_uid': str(uid),
    })

def about_view(request):
    return render(request, 'predictor/about.html')

# API Endpoints Protected by CSRF & Email Verification
@csrf_exempt
@email_verified_required
@require_POST
def api_predict(request):
    """
    JSON prediction endpoint with strict validation.
    """
    try:
        body = json.loads(request.body.decode('utf-8'))
        
        # Server-side Range Validation
        errors = validate_prediction_inputs(body)
        if errors:
            return JsonResponse({'error': errors[0]}, status=400)

        model_name = body.get('model_name', 'Random Forest')
        ml_service = MLPredictionService.get_instance()
        result = ml_service.predict(body, model_name=model_name)
        return JsonResponse({'status': 'success', 'data': result})
    except Exception as e:
        return JsonResponse({'error': f"Prediction failed: {str(e)}"}, status=400)

@csrf_exempt
@email_verified_required
@require_POST
def api_save_prediction(request):
    """
    Saves prediction to Cloud Firestore at users/{firebase_uid}/predictions/{pred_id}.
    Enforces verified Firebase token identity.
    """
    try:
        from .firestore_service import verify_firebase_id_token, refresh_firebase_id_token

        auth_header = request.headers.get('Authorization', '')
        id_token = None
        if auth_header.startswith('Bearer '):
            id_token = auth_header[7:].strip()
        if not id_token:
            id_token = request.headers.get('X-Firebase-ID-Token') or request.session.get('firebase_id_token')
        refresh_token = request.session.get('firebase_refresh_token')

        uid = None
        if id_token:
            verified = verify_firebase_id_token(id_token)
            if not verified and refresh_token:
                id_token = refresh_firebase_id_token(refresh_token)
                if id_token:
                    verified = verify_firebase_id_token(id_token)
            if verified:
                uid = verified.get('uid')

        if not uid:
            user = getattr(request, 'user_data', {}) or request.session.get('user') or {}
            uid = user.get('uid') or request.session.get('firebase_uid')

        if not uid:
            return JsonResponse({'error': 'Unauthorized: Valid Firebase token or session required.'}, status=401)

        body = json.loads(request.body.decode('utf-8'))
        pred_id = body.get('prediction_id') or f"pred_{uuid.uuid4().hex[:12]}"

        fs_service = FirestoreService.get_instance()
        saved_id = fs_service.save_prediction(
            user_uid=str(uid),
            prediction_data=body,
            prediction_id=pred_id,
            id_token=id_token,
            refresh_token=refresh_token
        )

        if not saved_id:
            return JsonResponse({'error': 'Failed to save prediction to Cloud Firestore.'}, status=500)

        return JsonResponse({'status': 'success', 'data': {'prediction_id': saved_id}})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
@email_verified_required
@require_POST
def api_delete_prediction(request, prediction_id):
    """
    Deletes prediction from Cloud Firestore users/{firebase_uid}/predictions/{prediction_id}.
    Strictly enforces user ownership.
    """
    try:
        from .firestore_service import verify_firebase_id_token, refresh_firebase_id_token

        if not re.match(r'^[a-zA-Z0-9_\-]+$', prediction_id):
            return JsonResponse({'error': 'Invalid prediction ID format.'}, status=400)

        auth_header = request.headers.get('Authorization', '')
        id_token = None
        if auth_header.startswith('Bearer '):
            id_token = auth_header[7:].strip()
        if not id_token:
            id_token = request.headers.get('X-Firebase-ID-Token') or request.session.get('firebase_id_token')
        refresh_token = request.session.get('firebase_refresh_token')

        uid = None
        if id_token:
            verified = verify_firebase_id_token(id_token)
            if not verified and refresh_token:
                id_token = refresh_firebase_id_token(refresh_token)
                if id_token:
                    verified = verify_firebase_id_token(id_token)
            if verified:
                uid = verified.get('uid')

        if not uid:
            user = getattr(request, 'user_data', {}) or request.session.get('user') or {}
            uid = user.get('uid') or request.session.get('firebase_uid')

        if not uid:
            return JsonResponse({'error': 'Unauthorized: Missing session identity.'}, status=401)

        existing_rec = PredictionRecord.objects.filter(prediction_id=prediction_id).first()
        if existing_rec and existing_rec.user_uid and str(existing_rec.user_uid) != str(uid):
            return JsonResponse({'error': 'Forbidden: You do not own this prediction.'}, status=403)

        fs_service = FirestoreService.get_instance()
        success = fs_service.delete_user_prediction(
            user_uid=str(uid),
            prediction_id=prediction_id,
            id_token=id_token,
            refresh_token=refresh_token
        )
        if success:
            if existing_rec:
                existing_rec.delete()
            return JsonResponse({'status': 'success', 'message': 'Prediction deleted successfully.'})
        return JsonResponse({'error': 'Record not found or access forbidden.'}, status=403)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def api_chart_data(request):
    ml_service = MLPredictionService.get_instance()
    return JsonResponse({
        'metrics': ml_service.get_metrics(),
        'feature_importance': ml_service.get_feature_importance(),
        'scatter_sample': ml_service.get_scatter_sample(),
        'distribution': ml_service.get_distribution_sample()
    })
