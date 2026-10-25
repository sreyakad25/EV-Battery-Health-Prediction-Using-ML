import json
import logging
import requests
from datetime import datetime
from django.conf import settings

logger = logging.getLogger(__name__)

FIRESTORE_REST_BASE = "https://firestore.googleapis.com/v1/projects"


def get_firebase_config():
    return getattr(settings, 'FIREBASE_CONFIG', {})


def refresh_firebase_id_token(refresh_token):
    """
    Exchanges a Firebase refresh token for a fresh Firebase ID token.
    """
    fb_config = get_firebase_config()
    api_key = fb_config.get('apiKey', '').strip()
    if not api_key or not refresh_token:
        return None

    try:
        url = f"https://securetoken.googleapis.com/v1/token?key={api_key}"
        res = requests.post(
            url,
            data={
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token
            },
            timeout=5
        )
        if res.status_code == 200:
            data = res.json()
            return data.get('id_token')
    except Exception as e:
        logger.warning(f"Failed to refresh Firebase ID token: {e}")
    return None


def verify_firebase_id_token(id_token):
    """
    Verifies a Firebase ID token using Google Identity Toolkit and returns user data (localId / uid).
    """
    fb_config = get_firebase_config()
    api_key = fb_config.get('apiKey', '').strip()
    if not api_key or not id_token:
        return None

    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={api_key}"
        res = requests.post(url, json={'idToken': id_token}, timeout=5)
        if res.status_code == 200:
            users = res.json().get('users', [])
            if users:
                user = users[0]
                return {
                    'uid': user.get('localId'),
                    'email': user.get('email', '').strip().lower(),
                    'full_name': user.get('displayName', ''),
                    'email_verified': user.get('emailVerified', False),
                }
    except Exception as e:
        logger.warning(f"Firebase token verification failed: {e}")
    return None


def python_to_firestore_value(val):
    """
    Converts a Python value into Firestore REST API value dictionary.
    """
    if val is None:
        return {"nullValue": None}
    elif isinstance(val, bool):
        return {"booleanValue": val}
    elif isinstance(val, int):
        return {"integerValue": str(val)}
    elif isinstance(val, float):
        return {"doubleValue": float(val)}
    elif isinstance(val, str):
        return {"stringValue": str(val)}
    elif isinstance(val, dict):
        fields = {k: python_to_firestore_value(v) for k, v in val.items()}
        return {"mapValue": {"fields": fields}}
    elif isinstance(val, (list, tuple)):
        values = [python_to_firestore_value(item) for item in val]
        return {"arrayValue": {"values": values}}
    elif isinstance(val, datetime):
        return {"timestampValue": val.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}
    return {"stringValue": str(val)}


def firestore_value_to_python(val_dict):
    """
    Converts a Firestore REST API value dictionary into a standard Python value.
    """
    if not isinstance(val_dict, dict):
        return val_dict

    if "nullValue" in val_dict:
        return None
    if "booleanValue" in val_dict:
        return bool(val_dict["booleanValue"])
    if "integerValue" in val_dict:
        return int(val_dict["integerValue"])
    if "doubleValue" in val_dict:
        return float(val_dict["doubleValue"])
    if "stringValue" in val_dict:
        return str(val_dict["stringValue"])
    if "timestampValue" in val_dict:
        return str(val_dict["timestampValue"])
    if "mapValue" in val_dict:
        fields = val_dict["mapValue"].get("fields", {})
        return {k: firestore_value_to_python(v) for k, v in fields.items()}
    if "arrayValue" in val_dict:
        values = val_dict["arrayValue"].get("values", [])
        return [firestore_value_to_python(v) for v in values]
    return None


def parse_firestore_doc(doc):
    """
    Parses a Firestore document object from REST API into a plain Python dictionary.
    """
    if not doc or not isinstance(doc, dict):
        return {}
    fields = doc.get("fields", {})
    parsed = {}
    for k, v in fields.items():
        parsed[k] = firestore_value_to_python(v)

    # Fallback created_at if missing
    if 'created_at' not in parsed and 'createTime' in doc:
        parsed['created_at'] = doc.get('createTime')
    if 'prediction_id' not in parsed:
        name = doc.get('name', '')
        if name:
            parsed['prediction_id'] = name.split('/')[-1]

    return parsed


def _sort_key_for_prediction(item):
    val = item.get('created_at') or ''
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float)):
        return str(val)
    return str(val)


def save_prediction_to_firestore(user_uid, prediction_data, id_token=None, refresh_token=None):
    """
    Saves a battery prediction to Cloud Firestore at:
    users/{firebase_uid}/predictions/{prediction_id}
    """
    fb_config = get_firebase_config()
    project_id = fb_config.get('projectId', 'ev-battery-health-prediction').strip()
    if not user_uid or not project_id:
        logger.warning("save_prediction_to_firestore: Missing user_uid or project_id")
        return None

    pred_id = prediction_data.get('prediction_id')
    if not pred_id:
        pred_id = f"pred_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{user_uid[:6]}"
        prediction_data['prediction_id'] = pred_id

    # Format fields
    firestore_fields = {}
    for key, value in prediction_data.items():
        if key == 'created_at':
            continue
        firestore_fields[key] = python_to_firestore_value(value)

    # Ensure authentic user UID
    firestore_fields['user_uid'] = python_to_firestore_value(user_uid)

    # Store created_at as real Firestore Timestamp
    raw_created = prediction_data.get('created_at')
    if isinstance(raw_created, datetime):
        firestore_fields['created_at'] = {"timestampValue": raw_created.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}
    elif isinstance(raw_created, str) and raw_created:
        try:
            cleaned = raw_created.replace('Z', '').split('.')[0]
            if 'T' in cleaned:
                dt = datetime.strptime(cleaned, '%Y-%m-%dT%H:%M:%S')
            else:
                dt = datetime.strptime(cleaned, '%Y-%m-%d %H:%M:%S')
            firestore_fields['created_at'] = {"timestampValue": dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}
        except Exception:
            firestore_fields['created_at'] = {"timestampValue": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')}
    else:
        firestore_fields['created_at'] = {"timestampValue": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')}

    url = (
        f"{FIRESTORE_REST_BASE}/{project_id}/databases/(default)/documents/"
        f"users/{user_uid}/predictions/{pred_id}"
    )

    # If id_token is missing, try refreshing proactive
    if not id_token and refresh_token:
        id_token = refresh_firebase_id_token(refresh_token)

    headers = {'Content-Type': 'application/json'}
    if id_token:
        headers['Authorization'] = f"Bearer {id_token}"

    try:
        res = requests.patch(url, headers=headers, json={"fields": firestore_fields}, timeout=6)
        if res.status_code in (401, 403) and refresh_token:
            new_token = refresh_firebase_id_token(refresh_token)
            if new_token:
                headers['Authorization'] = f"Bearer {new_token}"
                res = requests.patch(url, headers=headers, json={"fields": firestore_fields}, timeout=6)

        if res.status_code in (200, 201):
            logger.info(f"Successfully saved prediction {pred_id} to Firestore users/{user_uid}/predictions")
            return pred_id
        else:
            logger.warning(f"Firestore REST write returned {res.status_code}: {res.text}")
            return None
    except Exception as e:
        logger.warning(f"Firestore REST write error: {e}")
        return None


def get_user_predictions_from_firestore(user_uid, id_token=None, refresh_token=None, limit=100):
    """
    Fetches prediction records from Cloud Firestore at:
    users/{firebase_uid}/predictions
    """
    fb_config = get_firebase_config()
    project_id = fb_config.get('projectId', 'ev-battery-health-prediction').strip()
    if not user_uid or not project_id:
        return []

    url = (
        f"{FIRESTORE_REST_BASE}/{project_id}/databases/(default)/documents/"
        f"users/{user_uid}/predictions?pageSize={limit}"
    )

    if not id_token and refresh_token:
        id_token = refresh_firebase_id_token(refresh_token)

    headers = {'Content-Type': 'application/json'}
    if id_token:
        headers['Authorization'] = f"Bearer {id_token}"

    try:
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code in (401, 403) and refresh_token:
            new_token = refresh_firebase_id_token(refresh_token)
            if new_token:
                headers['Authorization'] = f"Bearer {new_token}"
                res = requests.get(url, headers=headers, timeout=6)

        if res.status_code == 200:
            data = res.json()
            documents = data.get("documents", [])
            records = [parse_firestore_doc(doc) for doc in documents]
            # Sort newest first
            records.sort(key=_sort_key_for_prediction, reverse=True)
            return records
        else:
            logger.warning(f"Firestore REST read returned {res.status_code}: {res.text}")
    except Exception as e:
        logger.warning(f"Firestore REST read error: {e}")

    return []


def get_prediction_from_firestore(user_uid, pred_id, id_token=None, refresh_token=None):
    """
    Fetches a single prediction record from Cloud Firestore at:
    users/{firebase_uid}/predictions/{pred_id}
    """
    fb_config = get_firebase_config()
    project_id = fb_config.get('projectId', 'ev-battery-health-prediction').strip()
    if not user_uid or not pred_id or not project_id:
        return None

    url = (
        f"{FIRESTORE_REST_BASE}/{project_id}/databases/(default)/documents/"
        f"users/{user_uid}/predictions/{pred_id}"
    )

    if not id_token and refresh_token:
        id_token = refresh_firebase_id_token(refresh_token)

    headers = {'Content-Type': 'application/json'}
    if id_token:
        headers['Authorization'] = f"Bearer {id_token}"

    try:
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code in (401, 403) and refresh_token:
            new_token = refresh_firebase_id_token(refresh_token)
            if new_token:
                headers['Authorization'] = f"Bearer {new_token}"
                res = requests.get(url, headers=headers, timeout=6)

        if res.status_code == 200:
            doc = res.json()
            return parse_firestore_doc(doc)
        else:
            logger.warning(f"Firestore REST get prediction returned {res.status_code}: {res.text}")
    except Exception as e:
        logger.warning(f"Firestore REST get prediction error: {e}")

    return None


def delete_prediction_from_firestore(user_uid, pred_id, id_token=None, refresh_token=None):
    """
    Deletes a prediction record from Cloud Firestore at:
    users/{firebase_uid}/predictions/{pred_id}
    """
    fb_config = get_firebase_config()
    project_id = fb_config.get('projectId', 'ev-battery-health-prediction').strip()
    if not user_uid or not pred_id or not project_id:
        return False

    url = (
        f"{FIRESTORE_REST_BASE}/{project_id}/databases/(default)/documents/"
        f"users/{user_uid}/predictions/{pred_id}"
    )

    if not id_token and refresh_token:
        id_token = refresh_firebase_id_token(refresh_token)

    headers = {'Content-Type': 'application/json'}
    if id_token:
        headers['Authorization'] = f"Bearer {id_token}"

    try:
        res = requests.delete(url, headers=headers, timeout=6)
        if res.status_code in (401, 403) and refresh_token:
            new_token = refresh_firebase_id_token(refresh_token)
            if new_token:
                headers['Authorization'] = f"Bearer {new_token}"
                res = requests.delete(url, headers=headers, timeout=6)

        return res.status_code in (200, 204)
    except Exception as e:
        logger.warning(f"Firestore REST delete error: {e}")
        return False


class FirestoreService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def save_prediction(self, user_uid, prediction_data, prediction_id=None, id_token=None, refresh_token=None):
        if not id_token and refresh_token:
            id_token = refresh_firebase_id_token(refresh_token)
        if prediction_id and isinstance(prediction_data, dict):
            prediction_data['prediction_id'] = prediction_id
        return save_prediction_to_firestore(
            user_uid=user_uid,
            prediction_data=prediction_data,
            id_token=id_token,
            refresh_token=refresh_token
        )

    def get_user_predictions(self, user_uid, id_token=None, refresh_token=None, limit=100):
        if not id_token and refresh_token:
            id_token = refresh_firebase_id_token(refresh_token)
        return get_user_predictions_from_firestore(
            user_uid=user_uid,
            id_token=id_token,
            refresh_token=refresh_token,
            limit=limit
        )

    def get_prediction(self, user_uid, pred_id, id_token=None, refresh_token=None):
        if not id_token and refresh_token:
            id_token = refresh_firebase_id_token(refresh_token)
        return get_prediction_from_firestore(
            user_uid=user_uid,
            pred_id=pred_id,
            id_token=id_token,
            refresh_token=refresh_token
        )

    def delete_prediction(self, user_uid, pred_id, id_token=None, refresh_token=None):
        if not id_token and refresh_token:
            id_token = refresh_firebase_id_token(refresh_token)
        return delete_prediction_from_firestore(
            user_uid=user_uid,
            pred_id=pred_id,
            id_token=id_token,
            refresh_token=refresh_token
        )

    def delete_user_prediction(self, user_uid, prediction_id, id_token=None, refresh_token=None):
        return self.delete_prediction(
            user_uid=user_uid,
            pred_id=prediction_id,
            id_token=id_token,
            refresh_token=refresh_token
        )

