# Future Improvements — AIVoiceMaker

Ideas concretas para mejorar el modelo de voz, ordenadas por **impacto vs. esfuerzo**.
Cada tarea incluye el porqué, los pasos sugeridos y una estimación de ganancia esperada.

> Estado base actual: similitud de hablante **0.846**, WER held-out **11.4 %**,
> WER validación **27.8 %**. Modelo subjetivamente "decente pero ligeramente sintético".

---

## 🟢 Quick wins (alto impacto, bajo esfuerzo)

### 1. Resume training desde `best_model.pth` con LR bajo (polish pass)

- **Por qué:** 10 épocas es corto para XTTS-v2; el modelo aún tiene margen para
  ajustar prosodia sin sobreentrenar. Un "polish pass" con LR muy bajo es la
  receta clásica para exprimir los últimos puntos de calidad.
- **Cómo:**
  - En PC con la RTX 3070, modificar `src/train.py`:
    - `lr=1e-6` (10× más bajo que el actual `5e-6`).
    - `epochs=5`.
    - Cargar `best_model.pth` como restore_path en el trainer.
  - Correr `python -m src.train --epochs 5 --batch-size 2 --grad-accum 8`.
- **Esfuerzo:** ~1 h en 3070.
- **Ganancia esperada:** +0.02–0.05 en similitud, voz menos "rígida" en frases largas.

### 2. Filtrado más estricto del dataset (drop chunks malos)

- **Por qué:** la similitud del hablante se ve dominada por los peores chunks, no por
  los mejores. Tienes 2 250 muestras pero seguramente hay 100–300 con: silencio
  largo, música de fondo, risa cortando palabras, o transcripción Whisper con
  baja confianza.
- **Cómo:**
  - Agregar a `src/preprocess.py` o un nuevo `src/filter_dataset.py`:
    - Drop chunks con duración < 3.0 s o > 10 s.
    - Drop chunks cuyo SNR estimado sea < 15 dB (`librosa.feature.rms` + percentiles).
    - Drop chunks con texto muy corto (< 10 caracteres después del strip).
  - Re-ejecutar `transcribe.py` capturando además `avg_logprob` por chunk; descartar
    los que tengan `avg_logprob < -1.0` (Whisper "no muy seguro").
  - Re-correr `split_dataset.py` y re-entrenar.
- **Esfuerzo:** 2 h de código + 1 h de re-entrenamiento.
- **Ganancia esperada:** +0.03–0.06 en similitud, menos mumbles, WER puede bajar a ~9 %.

### 3. Más datos de YouTube (más videos del mismo hablante)

- **Por qué:** 3 h es el mínimo para XTTS fine-tuning; 10 h es donde la calidad
  se acerca a "profesional". Más datos = más diversidad prosódica.
- **Cómo:**
  - Descargar 5–10 videos adicionales del mismo canal con `yt-dlp`.
  - Ejecutar el pipeline existente: `preprocess.py` → `transcribe.py` → `split_dataset.py`.
  - Idealmente elegir variedad: streams largos (cadencia conversacional) +
    videos editados (energía variable) + colabs (hablar con otros, pausas).
- **Esfuerzo:** 1 h descarga + 4 h transcripción + 4 h re-entrenamiento.
- **Ganancia esperada:** mayor naturalidad, especialmente en frases largas; voz
  más expresiva. Es la mejora con mayor techo.

---

## 🟡 Mejoras medias (impacto medio o esfuerzo medio)

### 4. Limpieza de audio con denoising / dereverb antes del chunking

- **Por qué:** YouTube audio frecuentemente tiene reverb del cuarto, ruido de
  ventilador, o compresión MP3 audible. XTTS entrena el timbre con todos esos
  artefactos incluidos.
- **Cómo:**
  - Pipeline pre-VAD: pasar cada MP3 por **DeepFilterNet 3** (`pip install deepfilternet`)
    o **Demucs** para separar voz del fondo.
  - Opcional: dereverb con **MossFormer2** (modelo open-source).
  - Renombrar el directorio: `data/raw_audio/` → procesar → `data/clean_audio/` →
    correr el pipeline normal sobre el limpio.
- **Esfuerzo:** 4 h (DeepFilterNet es directo; Demucs requiere afinar parámetros).
- **Ganancia esperada:** voz sintetizada más nítida, sin "halo" de fondo, mejor
  timbre. Reportado como una de las mejoras más subjetivamente notables.

### 5. Data augmentation en tiempo de entrenamiento

- **Por qué:** XTTS suele generalizar mejor cuando los chunks tienen variación
  en pitch, velocidad y volumen.
- **Cómo:**
  - En el dataset loader del trainer, aplicar con probabilidad 0.3:
    - Pitch shift de ±2 semitones (`librosa.effects.pitch_shift`).
    - Time stretch ±10 % (`librosa.effects.time_stretch`).
    - Gain de ±3 dB.
  - **Importante:** NO augmentar las `gpt_cond` references; solo el target del loss.
- **Esfuerzo:** 3 h (modificar `XttsDataset` en TTS).
- **Ganancia esperada:** modelo más robusto a referencias variadas, menos
  diferencia entre el sonido de las samples generadas y la voz real.

### 6. Validar y limpiar las pseudo-labels de Whisper

- **Por qué:** entrenamos con texto generado por Whisper, que tiene WER ~5 % de
  base. Esos errores se propagan al modelo TTS, que aprende asociaciones
  texto-audio incorrectas.
- **Cómo:**
  - Re-transcribir el dataset con un modelo distinto (e.g. **NVIDIA Parakeet**
    o **Whisper large-v3-turbo**) y conservar solo los chunks donde ambos
    transcriptores coinciden (CER < 0.05).
  - O: usar **forced alignment** con `wav2vec2` para verificar que las palabras
    están en los timestamps esperados.
- **Esfuerzo:** 4 h (transcripción dual + script de comparación).
- **Ganancia esperada:** WER held-out baja a 8–9 %; pronunciación más clara de
  palabras técnicas y nombres propios.

### 7. Ajustar `gpt_cond_len` al máximo permitido

- **Por qué:** XTTS condiciona la generación con hasta 6 s del clip de
  referencia. Pasamos clips de 11 s pero el modelo solo "ve" los primeros 6 s.
- **Cómo:**
  - En `src/infer.py`, cambiar el llamado a `model.synthesize(...)` para pasar:
    - `gpt_cond_len=6`
    - `gpt_cond_chunk_len=6`
    - `max_ref_len=12`
  - Probar con varios chunks de referencia distintos por sintesis (ya hacemos esto)
    y comparar.
- **Esfuerzo:** 30 min.
- **Ganancia esperada:** marginal pero gratis. Vale la pena medirlo.

---

## 🔴 Mejoras grandes (alto esfuerzo, opcional)

### 8. Entrenar un modelo de habla emocional sobre el mismo dataset

- **Por qué:** XTTS-v2 es agnóstico a emoción. Si etiquetas chunks por emoción
  (alegre, neutral, sorpresa, etc.) puedes condicionar la generación.
- **Cómo:**
  - Clasificar cada chunk con **wav2vec2-emotion** o **SpeechBrain emotion**.
  - Agregar la columna `emotion` al manifest.
  - Modificar el trainer para añadir un embedding de emoción al input del GPT.
  - O alternativa zero-shot: pasar referencias diferentes según la emoción
    deseada (ej: clip donde se ríe → genera frases más alegres).
- **Esfuerzo:** 1–2 días.
- **Ganancia esperada:** voz expresiva controlable; punto fuerte para demo.

### 9. Cuantización del modelo a INT8 / GGUF para inferencia rápida

- **Por qué:** el checkpoint de 2 GB tarda 25 s en cargar en MPS y 5–15 s por
  oración. Cuantizar lo aceleraría 2–3× y haría posible correr en navegador con WebGPU.
- **Cómo:**
  - `bitsandbytes` con `load_in_8bit=True` para inferencia.
  - O exportar a **ONNX** + cuantizar con **onnxruntime-tools**.
  - Probar con `optimum-quanto` (Hugging Face).
- **Esfuerzo:** 1 día (PyTorch → ONNX no es trivial para modelos GPT autorregresivos).
- **Ganancia esperada:** 2–3× speedup en MPS; abre la puerta a deploy en GitHub
  Pages real (con WebGPU + ONNX runtime web).

### 10. Multi-speaker fine-tuning (entrenar con varios hablantes y usar speaker IDs)

- **Por qué:** entrenar con un solo hablante limita la diversidad prosódica del
  modelo. Con 5–10 hablantes distintos + speaker embeddings, XTTS aprende
  patrones generales de habla y puede transferirlos al hablante objetivo.
- **Cómo:**
  - Recopilar 30 min–1 h de cada uno de 5–10 hablantes distintos.
  - Etiquetar `speaker_id` en el manifest.
  - Re-entrenar con `use_speaker_embedding=True` en el config.
  - En inferencia, condicionar con el embedding del hablante objetivo
    + las referencias WAV.
- **Esfuerzo:** 2–3 días (datos + entrenamiento más largo).
- **Ganancia esperada:** mejor generalización, menos overfitting al
  estilo conversacional único de Amelia, modelo "más versátil".

### 11. Reemplazar el vocoder HiFiGAN por uno más nuevo (BigVGAN, Vocos)

- **Por qué:** XTTS-v2 usa HiFiGAN para el último paso (mel → waveform).
  HiFiGAN es de 2020; modelos más recientes como **BigVGAN-v2** o **Vocos**
  producen audio más claro y menos artefactual.
- **Cómo:**
  - Entrenar/cargar BigVGAN-v2 (`pip install bigvgan`).
  - Modificar `src/infer.py` para que después de obtener el mel del GPT,
    pase por BigVGAN en lugar del HiFiGAN integrado.
  - Cuidado: requiere reescribir parte del flujo de XTTS.
- **Esfuerzo:** 2 días (es invasivo).
- **Ganancia esperada:** audio más limpio en agudos y consonantes sibilantes;
  diferencia menor en voz conversacional pero notable en frases con muchas "s".

---

## ⚙️ Mejoras de infraestructura (no de modelo, pero útiles)

### 12. Métricas humanas (MOS) sobre las muestras generadas

- **Por qué:** todas las métricas actuales son automáticas (Resemblyzer, WER).
  El estándar académico para TTS es **MOS** (Mean Opinion Score) de 1 a 5.
- **Cómo:**
  - Generar 20 muestras (mezcla de prompts inéditos y validation).
  - Pedir a 5–10 personas que las califiquen 1–5 en naturalidad y similitud.
  - Promediar y reportar como tabla en el reporte.
- **Esfuerzo:** 4 h (más coordinación con voluntarios).
- **Ganancia esperada:** métrica reportable que añade rigor académico
  significativo al proyecto.

### 13. CI/CD que regenere muestras automáticamente al hacer push

- **Por qué:** ahora mismo `docs/samples/*.wav` se generan manualmente. Un GitHub
  Action podría regenerarlos cada vez que cambies el modelo o `infer.py`.
- **Cómo:**
  - GitHub Action que: (a) descargue el checkpoint de un release, (b) corra
    el pipeline de generación, (c) commitee los nuevos samples al branch
    `gh-pages`.
- **Esfuerzo:** 4 h.
- **Ganancia esperada:** sitio siempre actualizado, demos reproducibles.

### 14. Versionado del modelo con DVC o Git LFS

- **Por qué:** el `best_model.pth` de 2 GB no está en el repo. Si en el futuro
  entrenas múltiples versiones, perderás trazabilidad.
- **Cómo:**
  - `dvc init` + `dvc add models/finetuned/`.
  - O crear releases en GitHub con los `.pth` adjuntos.
- **Esfuerzo:** 2 h.
- **Ganancia esperada:** historial limpio de modelos, fácil rollback.

### 15. Detector de spoofing / marca de agua de IA

- **Por qué:** clonar voces conlleva riesgos éticos (deepfakes). Como contramedida
  académica, agregar una marca de agua audible o inaudible al audio generado
  sería un punto interesante para el reporte/presentación.
- **Cómo:**
  - **Inaudible:** insertar una señal en el espectro de alta frecuencia
    (>16 kHz) que un detector pueda recuperar.
  - **Audible:** prefijar todos los outputs con un beep corto o un "Generated
    by AIVoiceMaker" sintetizado.
  - Investigar **AudioSeal** (Meta, 2024) — biblioteca open-source para
    watermarking de audio neural.
- **Esfuerzo:** 1 día.
- **Ganancia esperada:** punto importante para discusión ética; demuestra
  responsabilidad del autor.

---

## Roadmap sugerido (si tuviera que elegir 3)

1. **Tarea 2** (filtrado del dataset) — la más barata con mayor ganancia objetiva.
2. **Tarea 4** (denoising/dereverb del audio fuente) — cambio subjetivo más notable.
3. **Tarea 1** (polish pass con LR bajo) — fácil de demostrar en la presentación
   como "v2 del modelo".

Si tienes tiempo extra: **Tarea 3** (más videos) tiene el techo más alto a largo plazo.

---

## Cómo trackear el progreso

Cada mejora debería ir acompañada de:

1. **Re-ejecutar `python -m src.evaluate`** y comparar similitud + WER.
2. **Regenerar `docs/samples/`** con el modelo nuevo.
3. **Actualizar `PROJECT_LOG.md`** con un nuevo §6.X que documente el cambio.
4. **Anotar en `docs/REPORTE.md`** §10 cualquier resultado relevante para defender en la presentación.

---

## Referencias útiles

- Coqui XTTS-v2 docs: https://docs.coqui.ai/en/latest/models/xtts.html
- BigVGAN-v2: https://github.com/NVIDIA/BigVGAN
- DeepFilterNet 3: https://github.com/Rikorose/DeepFilterNet
- AudioSeal (watermarking): https://github.com/facebookresearch/audioseal
- Whisper large-v3-turbo: https://github.com/openai/whisper
- Recipe oficial XTTS-v2: https://github.com/idiap/coqui-ai-TTS/tree/main/recipes/ljspeech/xtts_v2
