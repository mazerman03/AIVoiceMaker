# Reporte de Proyecto — AIVoiceMaker
### Clonación de voz mediante fine-tuning de XTTS-v2 a partir de grabaciones de YouTube

**Autor:** Max Zemeno  
**Repositorio:** https://github.com/mazerman03/AIVoiceMaker  
**Sitio demo (GitHub Pages):** https://mazerman03.github.io/AIVoiceMaker/  
**Documentos complementarios en `docs/`:** `THEORY.md` (teoría), `PROJECT_LOG.md` (bitácora técnica detallada).

---

## 1. Área de aplicación

Este proyecto se inscribe en el **Procesamiento de Lenguaje Natural (PLN) hablado**, específicamente en dos sub-áreas estrechamente relacionadas:

1. **Síntesis de voz neural (Text-to-Speech, TTS):** convertir texto escrito en una señal de audio que suena como una voz humana.
2. **Clonación de voz (Voice Cloning):** condicionar el sintetizador para que la voz generada imite el timbre, prosodia y estilo de un hablante específico, a partir de unas pocas muestras de su audio.

Adicionalmente se utiliza **Reconocimiento Automático de Voz (Automatic Speech Recognition, ASR)** como herramienta auxiliar en dos puntos del pipeline: (a) generar transcripciones aproximadas (*pseudo-labels*) cuando no se dispone de transcripciones manuales, y (b) evaluar el modelo midiendo qué tan bien un ASR independiente entiende el audio sintetizado.

---

## 2. Relevancia del área

La síntesis de voz personalizada es una de las áreas de PLN con mayor crecimiento práctico en los últimos años:

- **Accesibilidad:** lectores de pantalla, audiolibros y asistentes para personas con afasia o pérdida de voz (por cirugía o ELA) pueden recuperar la voz de la persona a partir de grabaciones antiguas.
- **Industrias creativas:** doblaje, videojuegos, narración de pódcast, traducción cross-lingual manteniendo el timbre del actor original.
- **Asistentes virtuales y agentes conversacionales:** voces más naturales aumentan la confianza y la usabilidad.
- **Educación e investigación:** reproducir conferencias en distintos idiomas con la voz del autor.
- **Frente ético:** dado el riesgo de *deepfakes* de audio, entender cómo funciona internamente esta tecnología es indispensable para diseñar contramedidas (detección, marcas de agua, consentimiento informado).

Las técnicas modernas de **deep learning** han transformado el área en dos sentidos:
1. **Calidad:** los sistemas neurales (Tacotron, FastSpeech, VITS, XTTS, Bark) producen audio prácticamente indistinguible del humano en muchos contextos, algo inalcanzable para los TTS concatenativos o paramétricos clásicos (formantes, HMM).
2. **Datos requeridos:** mediante *transfer learning* y modelos *zero-shot* como XTTS-v2, ya no se necesitan decenas de horas grabadas en estudio: con 1–10 minutos de referencia se logra una voz reconocible, y con 2–5 horas el resultado se acerca a calidad profesional.

La relevancia del proyecto está en demostrar, con un caso real y reproducible, que un usuario con una sola GPU de consumo (NVIDIA RTX 3070, 8 GB) puede entrenar un modelo de clonación de voz utilizando únicamente material disponible públicamente en YouTube, sin transcripciones manuales y sin equipos profesionales.

---

## 3. Objetivo

> **Construir un sistema completo, end-to-end y reproducible, que tome MP3s de YouTube como única entrada, y produzca una interfaz web donde el usuario pueda escribir cualquier texto y recibir audio sintetizado en la voz clonada del hablante original.**

Sub-objetivos verificables:

| # | Sub-objetivo | Métrica de éxito |
|---|--------------|------------------|
| 1 | Recuperar transcripciones automáticamente sin texto manual | ≥ 95 % de los chunks transcritos |
| 2 | Fine-tunear XTTS-v2 sobre el dataset propio sin colapso de entrenamiento | Loss de evaluación ↓ vs. step 0 |
| 3 | El modelo resultante imita la voz original | Similitud de hablante (cosine, embeddings Resemblyzer) > 0.80 |
| 4 | El audio sintetizado es inteligible | WER (Word Error Rate) < 15 % en prompts no vistos |
| 5 | Demo accesible para un usuario no técnico | App web local + sitio estático con muestras |

---

## 4. Datos

- **Fuente:** 4 videos del canal "Amelia Watson Ch. hololive-EN" descargados como MP3 (`yt-dlp`).
- **Duración total:** ~3 h de habla espontánea conversacional.
- **Idioma:** inglés.
- **Sin transcripciones manuales** — punto central del problema, ya que el TTS supervisado requiere pares (audio, texto). Se resuelve con *pseudo-labeling*: usar un ASR potente (Whisper) para generar transcripciones automáticas y entrenar el TTS sobre ellas.
- **Política de privacidad / licencias:** los MP3s están en `data/raw_audio/` y excluidos del repositorio mediante `.gitignore` para evitar redistribución de material con derechos de autor. El repositorio sólo contiene código y los 10 audios sintéticos de muestra.

---

## 5. Arquitectura del sistema

El proyecto está estructurado en seis fases, separadas por scripts:

```
data/raw_audio/*.mp3
        │
        │  (1) src/preprocess.py  ── VAD + chunking 24 kHz mono
        ▼
data/chunks/*.wav  (2 250 clips, ~3 s a 12 s)
        │
        │  (2) src/transcribe.py  ── faster-whisper large-v3
        ▼
data/manifest.csv  (audio_path | text | duration)
        │
        │  (3) src/split_dataset.py  ── 90 % train / 10 % val
        ▼
data/manifest_train.csv  +  data/manifest_val.csv
        │
        │  (4) src/train.py  ── fine-tune XTTS-v2 GPT (498.7 M params)
        ▼
models/finetuned/<run>/best_model.pth  (2 GB)
        │
        │  (5) src/evaluate.py  ── similitud de hablante + WER
        ▼
evaluation/{metrics.csv, synth/, val_synth/}
        │
        │  (6) app/gradio_app.py  ── demo interactiva
        ▼
http://127.0.0.1:7860
```

**Hardware split (división Mac/PC):** preprocesamiento + evaluación + demo en MacBook (Apple Silicon, dispositivo `mps`); transcripción con Whisper + fine-tuning en PC con GPU NVIDIA RTX 3070 (CUDA 12.1, fp16). El handoff entre máquinas se hizo por archivo zip.

---

## 6. Detalles del modelo

### 6.1 ¿Qué es XTTS-v2?

**XTTS-v2** (Coqui, 2024) es un modelo TTS multi-hablante, multilingüe y *zero-shot* basado en arquitectura Transformer. Está compuesto por tres bloques principales:

1. **GPT autorregresivo (498.7 M de parámetros):** dada la secuencia de texto + un *embedding* del hablante de referencia, predice una secuencia de "tokens de audio" discretos (códigos de un VQ-VAE).
2. **DVAE (Discrete VAE):** codifica audio en tokens discretos durante el entrenamiento, y los decodifica en mel-espectrogramas durante la inferencia. Sus pesos quedaron *congelados* en el fine-tuning.
3. **HiFiGAN vocoder:** convierte el mel-espectrograma final en una forma de onda a 24 kHz.

El modelo está **pre-entrenado en ~14 idiomas y miles de hablantes**, lo que le permite imitar voces nuevas con muy pocas muestras. En este proyecto se aplica **transfer learning**: partimos de los pesos públicos `coqui/XTTS-v2` y *fine-tuneamos* sólo el módulo GPT sobre los 2,250 chunks de Amelia Watson, lo que toma ~2 horas en una RTX 3070 (vs. semanas que tomaría entrenar desde cero).

### 6.2 Por qué fine-tuning y no clonación zero-shot

XTTS-v2 ya soporta clonación zero-shot pasando un solo clip de referencia a inferencia. Sin embargo:

- **Zero-shot:** captura el timbre pero no la prosodia ni los manerismos del hablante.
- **Fine-tuning sobre 3 h:** el modelo aprende patrones de entonación, ritmo y vocabulario propios del hablante, y la voz suena considerablemente más fiel.

### 6.3 Pipeline de código (resumen por archivo)

| Archivo | Responsabilidad | Decisiones técnicas clave |
|---------|-----------------|---------------------------|
| `src/utils.py` | Constantes, logging, detección de dispositivo (`cuda` / `mps` / `cpu`), `KMP_DUPLICATE_LIB_OK` para Windows | Centralizar paths para que cada script funcione independientemente |
| `src/preprocess.py` | MP3 → WAV mono 24 kHz → VAD (`webrtcvad`) → chunking de 3–12 s | VAD con agresividad 2; padding de 200 ms; 24 kHz porque es la frecuencia nativa de XTTS |
| `src/transcribe.py` | Para cada chunk, generar texto con `faster-whisper large-v3 int8_float16` | Cuantización int8_float16 cabe en 4–5 GB de VRAM; ~40 min para los 2 250 chunks |
| `src/split_dataset.py` | 90/10 train/val estratificado por duración | Semilla fija (`seed=42`) para reproducibilidad |
| `src/train.py` | Configura `GPTArgs` + `XttsAudioConfig` + `GPTTrainerConfig` y llama a `Trainer.fit()` | 10 épocas, batch 2, grad-accum 8 (batch efectivo 16), fp16, LR 5e-6 con scheduler MultiStepLR. Flags críticos descubiertos durante el debugging: `gpt_use_perceiver_resampler=True`, `gpt_use_masking_gt_prompt_approach=True` (sin ellos el optimizador falla porque parámetros instanciados nunca participan en el forward) |
| `src/infer.py` | Carga `best_model.pth` y expone `synthesize(text, references, **gen_params)` | Soporta múltiples referencias (XTTS promedia el embedding del hablante → mejor calidad subjetiva); todos los hiperparámetros de generación de XTTS expuestos |
| `src/evaluate.py` | Sintetiza prompts de control + chunks de validación; calcula similitud (Resemblyzer) y WER (Whisper sobre la re-síntesis) | 5 prompts inéditos + 10 muestras de validación |
| `app/gradio_app.py` | UI web local: textbox + dropdown de referencias + sliders de generación + 12 ejemplos | Multi-select de hasta 20 clips de referencia, default = los 3 más largos |
| `docs/index.html` | Página estática para GitHub Pages que reproduce los 10 audios de muestra | Carga `samples/index.json` por fetch |

### 6.4 Hiperparámetros del fine-tuning

```python
# src/train.py — extracto
GPTTrainerConfig(
    output_path=...,
    epochs=10,
    batch_size=2,
    grad_accum_steps=8,         # batch efectivo = 16
    optimizer="AdamW",
    optimizer_wd_only_on_weights=True,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-6,                    # LR baja: ya partimos de pesos buenos
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5},
    mixed_precision=True,       # fp16 para ahorrar VRAM
    save_step=1000,
    save_n_checkpoints=2,
    save_checkpoints=True,
)
```

Y los parámetros del módulo GPT (heredados del recipe oficial):

```python
GPTArgs(
    max_conditioning_length=132300,
    min_conditioning_length=66150,
    max_text_length=200,
    gpt_max_audio_tokens=605,
    gpt_use_masking_gt_prompt_approach=True,   # crítico
    gpt_use_perceiver_resampler=True,          # crítico
    ...
)
```

### 6.5 Inferencia: parámetros que afectan la naturalidad

| Parámetro | Default Coqui | Usado aquí | Efecto |
|-----------|---------------|-----------|--------|
| `temperature` | 0.65 | 0.70 | Más variación entonacional |
| `repetition_penalty` | 2.0 | 5.0 | Suprime tartamudeos / loops |
| `top_p` | 0.85 | 0.85 | Sampling nucleus |
| `top_k` | 50 | 50 | Limita vocabulario por paso |
| `enable_text_splitting` | False | True | Suaviza salidas multi-oración |
| `speaker_wav` (referencias) | 1 clip | 3 clips largos | Embedding promediado → voz más fiel |

---

## 7. Resultados — métricas de validación

El modelo se entrenó por 10 épocas (10,120 pasos efectivos) en una NVIDIA RTX 3070, fp16. Tiempo total: **~2 h 13 min**.

### 7.1 Curva de pérdida (loss)

| Punto | `loss_text_ce` | `loss_mel_ce` | `loss` total |
|-------|---------------:|--------------:|-------------:|
| Step 0 (inicio) | 0.0685 | 5.939 | 0.7509 |
| Eval final | 0.0660 | 5.095 | **5.161** |

La pérdida total que reporta el trainer cambia de escala porque incorpora pesos distintos cuando entra en evaluación; lo relevante es la reducción consistente de `loss_mel_ce` (5.94 → 5.10), que es la pérdida que mide qué tan bien el modelo predice los códigos de audio. El comportamiento es estable, sin signos de divergencia ni overfitting.

### 7.2 Métricas objetivas sobre el modelo final

Calculadas con `python -m src.evaluate`:

| Métrica | Valor | Interpretación |
|---------|------:|----------------|
| **Similitud de hablante** (cosine, embeddings Resemblyzer) | **0.846** | 1.0 = idéntico, 0.5 ≈ no relacionado. > 0.8 se considera "claramente el mismo hablante". |
| **WER** (5 prompts de control, no vistos en entrenamiento) | **11.4 %** | Whisper transcribe la voz sintetizada y se compara contra el texto fuente. El benchmark publicado de XTTS-v2 fine-tuneado oscila entre 8 % y 15 %. |
| **WER** (10 muestras del split de validación) | **27.8 %** | Más alto porque las propias pseudo-labels de Whisper sobre la validación contienen ruido; el WER aquí mezcla errores del TTS con errores de la etiqueta automática original. |

### 7.3 Métricas de proceso

| Categoría | Valor |
|-----------|------:|
| Chunks generados (preprocesamiento) | 2,250 |
| Chunks transcritos por Whisper (success rate) | 2,250 / 2,250 = **100 %** |
| Parámetros entrenables (sólo el GPT) | 498,699,671 |
| Pasos de optimización | 10,120 |
| Tamaño del checkpoint final (`best_model.pth`) | 2.08 GB |
| Memoria VRAM en pico durante entrenamiento | ~7 GB / 8 GB disponibles |

### 7.4 Tabla de cumplimiento de objetivos

| # | Objetivo | Meta | Resultado | ✓ / ✗ |
|---|----------|------|-----------|------|
| 1 | Transcripción automática sin texto manual | ≥ 95 % | 100 % | ✅ |
| 2 | Fine-tuning sin colapso | Loss ↓ | 5.94 → 5.10 (`mel_ce`) | ✅ |
| 3 | Imita la voz original | Similitud > 0.80 | 0.846 | ✅ |
| 4 | Audio inteligible | WER < 15 % | 11.4 % (held-out) | ✅ |
| 5 | Demo accesible | App + sitio web | Gradio + GH Pages funcionando | ✅ |

---

## 8. Resultados sobre datos diferentes a los de entrenamiento

Para probar generalización se usan **dos conjuntos de prompts que el modelo nunca vio durante el fine-tuning**:

### 8.1 Prompts inéditos (held-out)

Cinco oraciones genéricas en inglés diseñadas para ser foneticamente diversas (pangramas, números, signos de puntuación, palabras extranjeras). Resultados:

- **WER promedio: 11.4 %** (calculado por `evaluate.py` reproduciendo cada prompt y pasándolo de vuelta por Whisper).
- **Audios disponibles:**
  - Localmente: `evaluation/synth/prompt_00.wav` … `prompt_04.wav`
  - Públicamente (GitHub Pages): https://mazerman03.github.io/AIVoiceMaker/
  - En el repo (commiteados): `docs/samples/prompt_00.wav` … `prompt_09.wav` — 10 muestras frescas re-generadas con la mejora de multi-referencia + parámetros tuneados.

Lista de prompts incluidos en el sitio estático:

| # | Prompt |
|---|--------|
| 00 | Hello! This is my AI cloned voice, fine-tuned on three hours of YouTube audio. |
| 01 | Artificial intelligence is reshaping how we create digital media. |
| 02 | The quick brown fox jumps over the lazy dog. |
| 03 | She sells seashells by the seashore on sunny summer afternoons. |
| 04 | Once upon a time, in a land far, far away, a small fox went on an adventure. |
| 05 | Did you know that Hololive is a Japanese virtual YouTuber agency? |
| 06 | Today's weather is unusually pleasant for this time of year. |
| 07 | I can read full sentences with natural rhythm and intonation. |
| 08 | Please save your work before closing the application. |
| 09 | Goodbye, and thank you for trying out the AI voice maker demo. |

### 8.2 Split de validación

10 muestras del 10 % retenido del manifest. Sirve como control interno con la misma distribución del dominio de origen (charla informal de Amelia Watson). WER 27.8 % — mayor pero esperable, ya que se compara contra pseudo-labels de Whisper que tienen ~5 % de error de base.

### 8.3 Demo en vivo

- **Aplicación local (Gradio):** `python app/gradio_app.py` → http://127.0.0.1:7860. Cualquier texto en inglés; el usuario puede ajustar referencias y hiperparámetros de generación en tiempo real.
- **Sitio público (GitHub Pages):** sólo audios pre-generados (no se puede correr un modelo de 2 GB en el navegador).

---

## 9. Limitaciones y trabajo futuro

1. **Calidad subjetiva:** la voz aún se percibe ligeramente "TTS-y" en frases largas. Causas:
   - Dataset modesto (~3 h vs. 10–20 h ideales para fine-tuning de XTTS).
   - Sólo 10 épocas; un *polish pass* con LR 1e-6 por 5 épocas adicionales reduciría el error residual.
2. **Pseudo-labels imperfectas:** Whisper se equivoca en términos propios (nombres de juegos, expresiones japonesas), lo que introduce ruido en el target del entrenamiento.
3. **Inglés únicamente:** XTTS soporta multilingual zero-shot, pero el fine-tuning hecho aquí está sesgado al inglés.
4. **Cuestión ética:** clonar la voz de una persona pública sin consentimiento expreso plantea dudas — el proyecto se documenta como ejercicio académico y los audios fuente no se redistribuyen (`.gitignore`).
5. **Hardware:** el inferencia en MacBook (`mps`) toma 5–15 s por oración; en una GPU NVIDIA con CUDA serían < 2 s.

---

## 10. Conclusión personal

Lo que más me llevo de este proyecto no es el resultado final (que también — escuchar mi propio modelo decir frases inventadas en una voz reconocible es genuinamente sorprendente), sino **lo que aprendí sobre el flujo real de un proyecto de deep learning fuera del aula**.

Cuando uno aprende redes neuronales en clase suele ver el proceso como "arma el modelo, entrena, evalúa". En la práctica, descubrí que esos tres pasos representan tal vez el 30 % del trabajo. El otro 70 % se dividió en:

1. **Preparación de datos** (preprocesamiento, VAD, chunking, generación de pseudo-labels) — sin esto no hay nada que entrenar.
2. **Compatibilidad de versiones** — esto fue lo más sorprendente: pasé prácticamente una sesión entera de trabajo resolviendo el rompecabezas de versiones de PyTorch, CUDA, transformers y la librería de Coqui (que está abandonada por sus creadores originales y mantenida por terceros). La sección 6.7 de `PROJECT_LOG.md` documenta nueve crashes encadenados, cada uno con su commit. Aprendí que un proyecto moderno de ML invierte tanto esfuerzo en *infraestructura* como en *modelado*.
3. **Validación y demo** — entrenar un modelo que nadie puede usar no sirve. Construir el pipeline de evaluación, el script de inferencia, la app de Gradio y el sitio estático le da al modelo una "superficie" donde puede ser tocado por un humano. Esto cambió mi perspectiva: el código de "demo" no es trivial, es parte del producto.

Otras lecciones específicas:

- **Transfer learning es transformador.** Fine-tunear 500 M parámetros en 2 horas con una GPU de gaming es algo impensable hace 5 años. El acceso a pesos pre-entrenados (XTTS-v2, Whisper) democratizó por completo este tipo de proyectos.
- **El pseudo-labeling funciona.** No tener transcripciones manuales fue inicialmente lo que me preocupaba; resultó ser el problema más fácil del proyecto, gracias a Whisper.
- **División de hardware:** trabajar entre Mac (cómodo, sin GPU CUDA) y PC (con la 3070 pero más engorroso) me forzó a diseñar el pipeline en archivos discretos comunicados por CSV y zip, lo que terminó siendo más limpio que un único notebook gigante.
- **Las métricas no lo cuentan todo.** Una similitud de 0.846 y un WER de 11.4 % son buenos números, pero la "naturalidad" subjetiva sigue siendo difícil de capturar y depende fuertemente de los hiperparámetros de generación, no sólo del entrenamiento. La mejora más grande en cómo *suena* el modelo no vino de re-entrenar sino de pasar de 1 clip de referencia a 3 clips largos.

Si volviera a empezar, dedicaría más tiempo desde el inicio a la **calidad y diversidad del dataset** (más videos, mejor VAD, filtrado por confianza de Whisper) y menos al ajuste fino de hiperparámetros: el cuello de botella claramente está en los datos, no en el optimizador.

---

## 11. Guion para la presentación oral

Sugerencia de estructura para la presentación (≈ 10 min):

1. **(1 min) Apertura — el problema y por qué importa**
   - Demo en vivo del Gradio: escribir una oración cualquiera y dejar que el público la escuche en la voz clonada *antes* de explicar nada.
   - Conectar con accesibilidad / industrias creativas / riesgos éticos.
2. **(1 min) Área de aplicación**
   - PLN hablado: TTS + Voice Cloning + ASR.
   - Por qué los modelos neurales superan a los métodos clásicos (concatenativos, paramétricos).
3. **(2 min) Modelo y código**
   - Diagrama del pipeline (sección 5 del reporte).
   - XTTS-v2: GPT autorregresivo + DVAE + HiFiGAN.
   - Transfer learning: pre-entrenado en miles de hablantes → fine-tuning sobre 3 h propias.
   - Pseudo-labeling con Whisper para resolver la falta de transcripciones.
   - Mostrar el bloque crítico de `train.py` con los flags `gpt_use_perceiver_resampler` (porque fue lo que costó más debuggear).
4. **(2 min) Resultados con métricas**
   - Tabla de objetivos (sección 7.4).
   - Curva de pérdida (5.94 → 5.10).
   - Similitud 0.846, WER 11.4 %.
5. **(2 min) Resultados sobre datos no vistos**
   - Reproducir 2–3 audios de `docs/samples/` que el modelo nunca vio.
   - Idealmente: que alguien del público escriba una frase en el Gradio en vivo.
6. **(1 min) Limitaciones y ética**
   - Voz aún ligeramente sintética; data y épocas como cuello de botella.
   - Riesgos de deepfake → necesidad de detección y consentimiento.
7. **(1 min) Conclusión personal**
   - "El 70 % del proyecto fue infraestructura, no modelado."
   - Transfer learning + pseudo-labeling democratizan voice cloning.
   - Lo aprendido se transfiere a cualquier proyecto futuro de deep learning.

---

## Anexos

- **Código completo:** https://github.com/mazerman03/AIVoiceMaker
- **Bitácora técnica detallada:** `docs/PROJECT_LOG.md` (incluye el chain de 9 commits que resolvió todos los crashes durante el fine-tuning).
- **Documento teórico introductorio:** `docs/THEORY.md`.
- **Muestras públicas:** `docs/samples/prompt_00.wav` … `prompt_09.wav` (también accesibles desde el GitHub Pages del proyecto).
- **Métricas crudas:** `evaluation/metrics.csv` (excluido del repo, regenerable con `python -m src.evaluate`).

### Dependencias principales

| Paquete | Versión usada | Rol |
|---------|---------------|-----|
| `coqui-tts` (fork de Idiap) | ≥ 0.24 | Modelo XTTS-v2 + entrenador |
| `torch` | 2.5.1+cu121 (PC) / 2.9 (Mac) | Backend de cómputo |
| `faster-whisper` | última | ASR para pseudo-labels |
| `webrtcvad` | última | Voice Activity Detection |
| `transformers` | ≥ 4.46 | Generación autorregresiva del GPT |
| `gradio` | última | UI web |
| `resemblyzer` | última | Embeddings de hablante para evaluar similitud |
| `jiwer` | última | Cálculo de WER |
| `huggingface_hub` | última | Descarga de pesos pre-entrenados de XTTS-v2 |
