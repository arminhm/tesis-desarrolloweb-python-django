from django.db import models

class DataRow(models.Model):
    archivo_nombre = models.CharField(max_length=255)
    data = models.JSONField()
    fecha_subida = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.archivo_nombre


class ColumnClassification(models.Model):
    archivo = models.ForeignKey(DataRow, on_delete=models.CASCADE, related_name='columnas')
    columna_nombre = models.CharField(max_length=255)
    tipo_dato = models.CharField(max_length=50, choices=[('real', 'Real'), ('categorico', 'Categórico'),('entero','Entero')])

    def __str__(self):
        return f"{self.columna_nombre} ({self.tipo_dato}) - {self.archivo.archivo_nombre}"


class ModeloEstandarizado(models.Model):
    nombre_archivo = models.CharField(max_length=255)
    data = models.JSONField()  # Almacena un diccionario de datos codificados

    def __str__(self):
        return f'Modelo de {self.nombre_archivo}'