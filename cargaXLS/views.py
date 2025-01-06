from django.shortcuts import render, redirect , get_object_or_404
from django.http import HttpResponse
from .models import DataRow, ColumnClassification , ModeloEstandarizado
from datetime import datetime
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from django.contrib import messages
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def subir_archivo(request):
    mensaje = None
    if request.method == 'POST':
        nombre_archivo = request.POST.get('nombre_archivo')
        archivo = request.FILES.get('archivo_xls')

        if not archivo or not nombre_archivo:
            mensaje = "Por favor, proporciona un nombre y selecciona un archivo."

        if archivo and nombre_archivo:
            if DataRow.objects.filter(archivo_nombre=nombre_archivo).exists():
                mensaje = "Ya existe un archivo con ese nombre. Por favor, elige otro nombre."
            else:
                try:
                    if not archivo.name.endswith(('.xls', '.xlsx')):
                        raise ValueError("El archivo debe ser un documento Excel (.xls, .xlsx).")

                    df = pd.read_excel(archivo)  # Leer el archivo XLS

                    if df.isnull().values.any():
                        raise ValueError("El archivo contiene datos nulos, lo que no es permitido.")         
                    
                    filas = [
                        DataRow(archivo_nombre=nombre_archivo, data=row.to_dict())
                        for _, row in df.iterrows()
                    ]
                    DataRow.objects.bulk_create(filas)
                    
                    # Redirigir a la vista de clasificación con un mensaje
                    return redirect('clasificar_columnas', nombre_archivo=nombre_archivo)
                except Exception as e:
                    mensaje = f'Error al procesar el archivo: {e}'
        else:
            mensaje = "Por favor, proporciona un nombre y selecciona un archivo."
    
    return render(request, 'cargaXLS/subir.html', {'mensaje': mensaje})

def clasificar_columnas(request, nombre_archivo):
    archivo = DataRow.objects.filter(archivo_nombre=nombre_archivo).first()
    if not archivo:
        return HttpResponse("Archivo no encontrado.")

    # Obtener las columnas originales desde los datos
    datos = archivo.data
    columnas_originales = list(datos.keys()) if datos else []

    if not columnas_originales:
        return HttpResponse("No hay columnas disponibles para clasificar.")

    if request.method == 'POST':
        nuevos_nombres = {}
        clasificaciones = {}

        # Procesar los cambios enviados desde el formulario
        for columna in columnas_originales:
            nuevo_nombre = request.POST.get(f'nombre_{columna}', columna).strip()
            tipo_dato = request.POST.get(f'tipo_{columna}', 'categorico')
            
            if nuevo_nombre:
                nuevos_nombres[columna] = nuevo_nombre
            clasificaciones[columna] = tipo_dato

        # Actualizar las filas en DataRow
        filas = DataRow.objects.filter(archivo_nombre=nombre_archivo)
        for fila in filas:
            data_actualizada = {
                nuevos_nombres.get(columna, columna): valor
                for columna, valor in fila.data.items()
            }
            fila.data = data_actualizada
            fila.save()

        # Guardar las clasificaciones en ColumnClassification
        ColumnClassification.objects.filter(archivo=archivo).delete()  # Limpiar clasificaciones anteriores
        for columna, tipo_dato in clasificaciones.items():
            ColumnClassification.objects.create(
                archivo=archivo,
                columna_nombre=nuevos_nombres.get(columna, columna),
                tipo_dato=tipo_dato
            )

        # Redirigir al resumen después de guardar
        return redirect('resumen_filas', nombre_archivo=nombre_archivo)

    # Renderizar las columnas originales para editar
    contexto = {
        'archivo_nombre': nombre_archivo,
        'columnas': columnas_originales,
    }
    return render(request, 'cargaXLS/clasificar.html', contexto)


def resumen_filas(request, nombre_archivo):
    # Filtrar todos los registros asociados al archivo_nombre
    archivos = DataRow.objects.filter(archivo_nombre=nombre_archivo)
    if not archivos.exists():
        return HttpResponse("Archivo no encontrado.")

    # Consolidar los datos en un DataFrame
    data_list = [archivo.data for archivo in archivos]
    df = pd.DataFrame(data_list)

    # Verificar si hay columnas numéricas
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    describe_data = df[numeric_columns].describe().to_dict() if numeric_columns else {}

    return render(request, 'cargaXLS/resumen_filas.html', {
        'nombre_archivo': nombre_archivo,
        'total_filas': len(df),
        'columnas_numericas': numeric_columns,
        'describe_data': describe_data,
    })

def obtener_describe(request, nombre_archivo):
    if request.method == 'GET':
        columna = request.GET.get('columna')

        # Obtener los datos del archivo
        archivo = get_object_or_404(DataRow, archivo_nombre=nombre_archivo)
        data_rows = DataRow.objects.filter(archivo_nombre=nombre_archivo).values_list('data', flat=True)
        df = pd.DataFrame(list(data_rows))

        # Generar describe para la columna solicitada
        if columna in df.columns:
            describe_col = df[columna].describe().to_dict()
            return JsonResponse({'describe': describe_col})
        return JsonResponse({'error': 'Columna no encontrada'}, status=400)
    

def graficos(request, nombre_archivo):
    archivos = DataRow.objects.filter(archivo_nombre=nombre_archivo)
    if not archivos.exists():
        return HttpResponse("Archivo no encontrado.")

    datos_archivo = [archivo.data for archivo in archivos]

    try:
        data = pd.DataFrame(datos_archivo)
    except ValueError as e:
        return HttpResponse("Error al procesar los datos del archivo.")

    columnas_numericas = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if not data.empty:
        columna_nombre = data.columns[0]  # Usamos la primera columna como nombre de empresa

        datos_columnas = {
            col: data[col].dropna().tolist() for col in columnas_numericas
        }

        # Obtener los nombres de las empresas como una lista para mostrarlos como etiquetas
        nombres_empresas = data[columna_nombre].tolist()

        return render(request, 'cargaXLS/graficos.html', {
            'nombre_archivo': nombre_archivo,
            'columnas_numericas': columnas_numericas,
            'datos_columnas': datos_columnas,
            'columna_nombre': columna_nombre,
            'nombres_empresas': nombres_empresas,  # Pasar los nombres de las empresas
        })
    else:
        return HttpResponse("No hay datos disponibles en el archivo.")

def archivos_disponibles(request):
    archivos = DataRow.objects.values('archivo_nombre').distinct()
    datos = ModeloEstandarizado.objects.values('nombre_archivo').distinct()

    return {'archivos_disponibles': archivos ,
            'datos_estandarizados':datos}

def datos_estandarizados(request):
    datos = ModeloEstandarizado.objects.values('nombre_archivo').distinct()
    return {'datos_estandarizados': datos}


def eliminar_archivo(request):
    if request.method == 'POST':
        archivo_a_eliminar = request.POST.get('archivo_a_eliminar')
        if archivo_a_eliminar:
            DataRow.objects.filter(archivo_nombre=archivo_a_eliminar).delete()
    return redirect('/')



def estandarizacion(request, nombre_archivo):
    # Recuperar todas las filas con el mismo archivo_nombre
    data_rows = DataRow.objects.filter(archivo_nombre=nombre_archivo)
    if not data_rows.exists():
        return JsonResponse({"error": "No se encontró ningún archivo con ese nombre."}, status=404)

    # Verificar si ya existe un modelo estandarizado con el mismo nombre
    if ModeloEstandarizado.objects.filter(nombre_archivo=nombre_archivo).exists():
        messages.warning(request, "Ya existe un modelo estandarizado con este nombre.")
        return redirect('graficos', nombre_archivo=nombre_archivo)

    # Juntar los datos en un solo DataFrame
    registros = []
    for data_row in data_rows:
        data = data_row.data  # Cada fila es un diccionario
        if isinstance(data, dict):  # Confirmar que es un diccionario
            registros.append(data)
        else:
            return JsonResponse({"error": "Formato de datos inválido en una fila."}, status=400)

    # Crear un DataFrame con todos los registros
    data = pd.DataFrame(registros)

    # Recuperar las clasificaciones de columnas asociadas al archivo
    clasificaciones = ColumnClassification.objects.filter(archivo__in=data_rows)
    clasificacion_dict = {c.columna_nombre: c.tipo_dato for c in clasificaciones}

    # Separar columnas numéricas y categóricas
    columnas_numericas = [col for col, tipo in clasificacion_dict.items() if tipo in ['real', 'entero']]
    columnas_categoricas = [col for col, tipo in clasificacion_dict.items() if tipo == 'categorico']

    try:
        # Estandarizar las columnas numéricas
        if columnas_numericas:
            scaler = StandardScaler()
            data[columnas_numericas] = scaler.fit_transform(data[columnas_numericas])

        # Convertir columnas categóricas a índices numéricos
        for col in columnas_categoricas:
            if col in data.columns:
                data[col] = data[col].astype('category').cat.codes

    except ValueError as e:
        messages.error(request, f"Las columnas fueron clasificadas incorrectamente , error: {e}")
        return redirect('clasificar_columnas', nombre_archivo=nombre_archivo)

    # Convertir el DataFrame a una lista de diccionarios serializables
    fila_estandarizada = data.to_dict(orient="records")

    # Guardar los datos estandarizados en el modelo ModeloEstandarizado
    ModeloEstandarizado.objects.create(
        nombre_archivo=nombre_archivo,
        data=fila_estandarizada
    )       

    messages.success(request, "Datos estandarizados y guardados correctamente.")

    return render(request, 'cargaXLS/estandarizacion.html', {
        'nombre_archivo': nombre_archivo,
        'clasificaciones': clasificaciones,
    })



def numero_cluster(request, nombre_archivo):
    # Obtener datos estandarizados del modelo
    data_entries = ModeloEstandarizado.objects.filter(nombre_archivo=nombre_archivo)

    if not data_entries.exists():
        return render(request, 'error.html', {
            'mensaje': 'No se encontraron datos estandarizados para este archivo.'
        })

    # Extraer todas las columnas y valores directamente
    scaled_data = []
    for entry in data_entries:
        for fila in entry.data:  # Recorrer cada diccionario en la lista
            valores = list(fila.values())  # Tomar todos los valores
            scaled_data.append(valores)

    # Convertir a numpy array
    try:
        scaled_data = np.array(scaled_data, dtype=float)
    except ValueError as e:
        return render(request, 'error.html', {
            'mensaje': f'Error al procesar los datos: {e}'
        })

    # Calcular SSE para diferentes valores de k
    sse = []
    k_values = range(1, 9)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_data)
        sse.append({'k': k, 'sse': kmeans.inertia_})

    # Enviar datos al template
    return render(request, 'cargaXLS/numero_cluster.html', {
        'nombre_archivo': nombre_archivo,
        'sse_data': json.dumps(sse)
    })


def clusterizar(request, nombre_archivo):
    # Obtener el número de clusters desde la URL
    clusters = int(request.GET.get('clusters', 3))  # Por defecto, 3 clusters

    # Recuperar el DataRow usando nombre_archivo
    data_row = DataRow.objects.filter(archivo_nombre=nombre_archivo).first()
    if not data_row:
        messages.error(request, "No se han encontrado datos para este archivo.")
        return redirect('graficos', nombre_archivo=nombre_archivo)

    # Recuperar los datos estandarizados
    modelo_estandarizado = ModeloEstandarizado.objects.filter(nombre_archivo=nombre_archivo).first()
    if not modelo_estandarizado:
        messages.error(request, "No se han encontrado datos estandarizados para este archivo.")
        return redirect('graficos', nombre_archivo=nombre_archivo)

    # Convertir el JSON de los datos estandarizados de nuevo a un DataFrame
    data = pd.DataFrame(modelo_estandarizado.data)

    # Recuperar la clasificación de las columnas (qué tipo de dato tiene cada columna)
    clasificacion_columnas = ColumnClassification.objects.filter(archivo=data_row)

    # Crear un diccionario para clasificar las columnas
    columnas_numericas = []
    for clasificacion in clasificacion_columnas:
        if clasificacion.tipo_dato in ['real', 'entero']:  # Filtrar columnas numéricas
            columnas_numericas.append(clasificacion.columna_nombre)

    # Seleccionar solo las columnas numéricas de la data
    data_numerica = data[columnas_numericas]

    # Normalizar los datos numéricos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_numerica)

    # Aplicar PCA (Reducción de dimensiones)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])

    # Aplicar KMeans para la clusterización
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    data['KMeans_Cluster'] = kmeans_labels

    # Mostrar el número de instancias por cluster
    cluster_counts = data['KMeans_Cluster'].value_counts()

    # Redirigir a una nueva vista con los resultados
    return render(request, 'cargaXLS/clusterizacion_resultados.html', {
        'nombre_archivo': nombre_archivo,
        'cluster_counts': cluster_counts,
        'data': data,
    })
