from django.urls import path
from . import views

urlpatterns = [
    path('', views.subir_archivo, name='subir_archivo'),
    path('clasificar/<str:nombre_archivo>/', views.clasificar_columnas, name='clasificar_columnas'),
    path('resumen/<str:nombre_archivo>/', views.resumen_filas, name='resumen_filas'),
    path('obtener_describe/<str:nombre_archivo>/', views.obtener_describe, name='obtener_describe'),
    path('graficos/<str:nombre_archivo>/', views.graficos, name='graficos'),
    path('eliminar_archivo/', views.eliminar_archivo, name='eliminar_archivo'),
    path('estandarizacion/<str:nombre_archivo>/', views.estandarizacion, name='estandarizacion'),
    path('numero_cluster/<str:nombre_archivo>/', views.numero_cluster, name='numero_cluster'),
    path('clusterizar/<str:nombre_archivo>/', views.clusterizar, name='clusterizar'),

]