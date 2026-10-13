---
title: Triaje manual de correos
emoji: 📧
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: other
---

# Thesis-Triage · Plataforma de triaje manual de correos

Repositorio de la plataforma web (Hugging Face Space con Gradio). Es el tercer componente de la tesis, junto a `Thesis-Document` (documento en LaTeX) y `Thesis-Project` (código del pipeline y cuaderno de Colab).

Versión web del triaje manual cronometrado de la tesis *Sistema de clasificación automática de correos
electrónicos por nivel de urgencia en el sector de la energía usando aprendizaje profundo* (Andrés Giovanny
Rubiano Muñoz, Maestría en Inteligencia Artificial, Universidad de La Salle).

Presenta el lote de simulación **a ciegas** y **en orden de llegada**, cronometra cada decisión y guarda
`manual/<anotador>.csv` (columnas `id, orden_llegada, etiqueta_humana, segundos`) en el repositorio de
datos privado de la tesis, de donde el cuaderno de Colab recupera los triajes para la comparativa de
tiempos y el análisis de errores.

## Configuración del Space

| Ajuste | Dónde | Valor |
|---|---|---|
| `HF_TOKEN` | Settings → Variables and secrets → **Secret** | Token de Hugging Face con permiso *write* |
| `DATASET_REPO` | Settings → Variables and secrets → **Variable** | `<usuario>/thesis-simulation` |

El repositorio de datos debe contener `simulation.csv` en su raíz (lo publica la sección 5.2 del cuaderno).
Los triajes y las constancias de consentimiento se sincronizan automáticamente cada minuto en las carpetas
`manual/` y `consent/` del mismo repositorio.

## Publicación en Hugging Face

1. Crear el Space (New → Space, SDK **Gradio**, CPU basic, público) y subir `app.py`, `requirements.txt` y `README.md`.
2. Configurar el secreto `HF_TOKEN` y la variable `DATASET_REPO` (tabla anterior).
3. El Space se construye solo; los anotadores entran por su URL, aceptan el consentimiento, se identifican y clasifican.

## Estructura

```
Thesis-Triage/
├── app.py             # aplicación Gradio (misma lógica que src/simulation/3-manual_triage.py de Thesis-Project)
├── requirements.txt   # dependencias adicionales a Gradio
├── README.md          # cabecera de configuración del Space + esta guía
└── data/              # carpeta local sincronizada con el dataset (se crea en ejecución, no se versiona)
```
