# Notebooks

## `demo.ipynb` — Demo end-to-end del proyecto

Recorre el pipeline completo en ~5 minutos:

1. Inspecciona el dataset (2 250 chunks + manifest + histograma de duraciones).
2. Carga el modelo XTTS-v2 fine-tuneado (`models/finetuned/<run>/best_model.pth`).
3. Genera audio inline para 5 prompts inéditos.
4. Compara 3 configuraciones de hiperparámetros (conservadora / default / expresiva) sobre el mismo prompt.
5. Carga `evaluation/metrics.csv` (similitud 0.846, WER 11.4 % / 27.8 %) y los grafica.
6. Lee la curva `loss_mel_ce` del run de TensorBoard si está disponible.

### Cómo correrlo

```bash
conda activate aivoice
jupyter lab notebooks/demo.ipynb
```

Requiere los mismos archivos que la app: `data/manifest.csv`, `data/chunks/*.wav`,
`models/finetuned/<run>/best_model.pth`, y `models/pretrained/XTTS-v2/{vocab.json,config.json,mel_stats.pth}`.

> El notebook se commitea **con outputs** (audio embebido + plots) para que pueda verse en GitHub
> sin necesidad de ejecutarlo. Esto es intencional para la presentación / entrega.

### Para presentar en vivo

- Reproducir las celdas de la **sección 4** (audio inline en el navegador del notebook).
- Mostrar la sección **5** (gráfica de barras de métricas).
- Cerrar con la sección **7** (tabla resumen).
- Cambiar a `python app/gradio_app.py` para tomar input del público en vivo.
