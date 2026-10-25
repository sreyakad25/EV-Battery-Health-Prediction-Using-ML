from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('predict/', views.predict_view, name='predict'),
    path('result/', views.result_view, name='result'),
    path('models/', views.models_comparison_view, name='models_comparison'),
    path('dataset/', views.dataset_info_view, name='dataset_info'),
    path('history/', views.history_view, name='history'),
    path('about/', views.about_view, name='about'),

    # APIs
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/save-prediction/', views.api_save_prediction, name='api_save_prediction'),
    path('api/delete-prediction/<str:prediction_id>/', views.api_delete_prediction, name='api_delete_prediction'),
    path('api/chart-data/', views.api_chart_data, name='api_chart_data'),
]
