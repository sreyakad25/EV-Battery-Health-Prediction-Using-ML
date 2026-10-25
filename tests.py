from django.test import TestCase, Client
from django.urls import reverse
from django.conf import settings
from accounts.models import UserProfile
from predictor.models import PredictionRecord
from predictor.ml_service import MLPredictionService

class EVBatteryHealthTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.ml_service = MLPredictionService.get_instance()

    def set_session_user(self, user_data):
        session = self.client.session
        session['user'] = user_data
        session.save()
        self.client.cookies[settings.SESSION_COOKIE_NAME] = session._session_key

    def test_public_pages_accessible(self):
        """Verify public pages render with HTTP 200 without authentication."""
        urls = [
            reverse('home'),
            reverse('accounts:login'),
            reverse('accounts:register'),
        ]
        for url in urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, f"Failed for {url}")

    def test_security_headers_enforced(self):
        """Verify essential defensive security headers are attached."""
        response = self.client.get(reverse('home'))
        self.assertEqual(response.headers.get('X-Frame-Options'), 'DENY')
        self.assertEqual(response.headers.get('X-Content-Type-Options'), 'nosniff')
        self.assertEqual(response.headers.get('Referrer-Policy'), 'strict-origin-when-cross-origin')

    def test_protected_routes_require_authentication(self):
        """Verify protected routes redirect unauthenticated users to login."""
        protected_urls = [
            reverse('predictor:dashboard'),
            reverse('predictor:predict'),
            reverse('predictor:history'),
            reverse('accounts:profile'),
        ]
        for url in protected_urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302)
            self.assertIn(reverse('accounts:login'), response.url)

    def test_unverified_email_redirects_to_verification_page(self):
        """Verify unverified users are gated and directed to verification screen."""
        self.set_session_user({
            'uid': 'test_unverified_uid',
            'email': 'unverified@test.com',
            'name': 'Unverified Tester',
            'email_verified': False
        })

        response = self.client.get(reverse('predictor:dashboard'))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('accounts:verify_email'), response.url)

    def test_verified_user_can_access_dashboard_and_prediction(self):
        """Verify verified users can access dashboard and prediction form."""
        self.set_session_user({
            'uid': 'test_verified_uid',
            'email': 'verified@test.com',
            'name': 'Verified Tester',
            'email_verified': True
        })

        dash_res = self.client.get(reverse('predictor:dashboard'))
        self.assertEqual(dash_res.status_code, 200)

        pred_res = self.client.get(reverse('predictor:predict'))
        self.assertEqual(pred_res.status_code, 200)

    def test_ml_prediction_service(self):
        """Verify ML prediction produces valid numerical SoH within range."""
        test_input = {
            'battery_capacity': 60.0,
            'vehicle_age': 3.0,
            'charging_cycles': 500,
            'temperature': 25.0,
            'fast_charge_ratio': 20.0,
            'discharge_rate': 1.5,
            'internal_resistance': 50.0,
            'driving_style': 'Moderate',
            'battery_chemistry': 'NMC',
            'vehicle_model': 'Tesla Model 3'
        }
        res = self.ml_service.predict(test_input, model_name='Random Forest')
        self.assertIn('predicted_soh', res)
        self.assertGreaterEqual(res['predicted_soh'], 40.0)
        self.assertLessEqual(res['predicted_soh'], 100.0)
        self.assertIn('health_status', res)

    def test_malformed_adversarial_input_rejected(self):
        """Verify server-side validation rejects maliciously extreme input values."""
        self.set_session_user({
            'uid': 'test_verified_uid',
            'email': 'verified@test.com',
            'name': 'Verified Tester',
            'email_verified': True
        })

        # Negative capacity should fail validation
        res = self.client.post(
            reverse('predictor:api_predict'),
            data='{"battery_capacity": -9999}',
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 400)

    def test_idor_protection_for_prediction_records(self):
        """Verify users cannot delete predictions belonging to another user."""
        # Record owned by User A
        rec = PredictionRecord.objects.create(
            prediction_id='pred_user_a_001',
            user_uid='user_a_uid',
            model_name='Random Forest',
            predicted_soh=84.5,
            health_status='Good Health',
            inputs={'battery_capacity': 60}
        )

        # Authenticate as User B
        self.set_session_user({
            'uid': 'user_b_uid',
            'email': 'user_b@test.com',
            'name': 'User B',
            'email_verified': True
        })

        # User B attempts to delete User A's prediction
        res = self.client.post(
            reverse('predictor:api_delete_prediction', kwargs={'prediction_id': 'pred_user_a_001'}),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 403)
        self.assertTrue(PredictionRecord.objects.filter(prediction_id='pred_user_a_001').exists())
