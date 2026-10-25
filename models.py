from django.db import models

class PredictionRecord(models.Model):
    prediction_id = models.CharField(max_length=64, primary_key=True)
    user_uid = models.CharField(max_length=128, db_index=True)
    model_name = models.CharField(max_length=64)
    predicted_soh = models.FloatField()
    health_status = models.CharField(max_length=64)
    inputs = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.prediction_id} - User: {self.user_uid} - SoH: {self.predicted_soh}%"

    class Meta:
        db_table = 'prediction_records'
        ordering = ['-created_at']
 