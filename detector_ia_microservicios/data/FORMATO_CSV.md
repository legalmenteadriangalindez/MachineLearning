# Formato del conjunto de entrenamiento

Guarda aquí `dataset.csv` en UTF-8 con encabezados:

```csv
texto,etiqueta
"[Coloca aquí un texto humano de al menos 50 caracteres con autoría verificable]",0
"[Coloca aquí un texto generado con IA de al menos 50 caracteres y origen verificable]",1
```

**No uses esas dos filas ilustrativas como datos de entrenamiento.**

- `0` = humano, `1` = IA.
- Necesitas 10 textos diferentes de cada clase para probar técnicamente el proceso.
- Para obtener un detector defendible necesitas muchos más textos, múltiples géneros,
  fuentes y modelos de IA, anotaciones correctas y una prueba externa independiente.
- No cargues textos privados o sensibles sin autorización.
- El programa elimina duplicados normalizados antes de separar entrenamiento y prueba.
