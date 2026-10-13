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

<div align="center">
    <img src="assets/images/Logo.png" width="250" alt="Logo Universidad de La Salle">
</div>

# Sistema de Clasificación Automática de Correos Electrónicos por Nivel de Urgencia en el Sector de la Energía usando Aprendizaje Profundo — *Plataforma de Triaje*

## 📋 Información General

<div align="center">
    <img src="assets/images/author/Andy Rubiano.png" width="200" alt="Foto de Andrés Giovanny Rubiano Muñoz" style="border-radius: 10px;">
</div>

| Aspecto | Detalles |
|--------|----------|
| **Autor** | Andrés Giovanny Rubiano Muñoz "Andy Rubiano" |
| **Correo** | arubiano67@unisalle.edu.co |
| **Director** | Paola Andrea Zambrano Garcia |
| **Programa** | Maestría en Inteligencia Artificial |
| **Universidad** | Universidad de La Salle |
| **Líneas de Investigación** | Procesamiento del Lenguaje Natural (NLP) y Aprendizaje Profundo (Deep Learning) |
| **Año** | 2026 |
| **Estado** | En desarrollo |

> Este repositorio contiene la **plataforma web de triaje manual** del proyecto de grado: un *Space* de Hugging Face (Gradio) donde los anotadores humanos clasifican el lote de simulación del objetivo 4. Es el tercer componente de la tesis, junto a **Thesis-Document** (documento en LaTeX) y **Thesis-Project** (código del pipeline y cuaderno de Colab).

---

## 🎯 Descripción del Proyecto

La validación en **entorno operativo simulado** (objetivo 4) mide cuánto tiempo ahorra el modelo seleccionado frente al triaje manual sobre un lote de correos inédito. Para ello hace falta una **referencia humana**: personas que clasifiquen el mismo lote, a ciegas y cronometradas. Esta plataforma reemplaza, para la web, la aplicación de escritorio en Tkinter del proyecto ([`3-manual_triage.py`](https://github.com/RubianoAndy/Thesis-Project/blob/main/src/simulation/3-manual_triage.py)), conservando **la misma lógica** y añadiendo lo necesario para que varios anotadores participen desde cualquier dispositivo y los resultados lleguen solos al cuaderno de Colab.

### Qué hace

- Presenta cada correo **a ciegas** (sin la predicción del modelo, la etiqueta de oro, la categoría ni el subconjunto) y **en orden de llegada** (FIFO).
- **Cronometra** cada decisión desde que el correo queda visible hasta que el anotador elige Alta, Media o Baja.
- Es **multi-anotador** (un archivo por persona) y **reanudable** desde cualquier dispositivo con el mismo identificador.
- Recoge la aceptación del **consentimiento informado** antes de empezar, con el mismo contenido del formato del apéndice de la tesis.
- Sincroniza los resultados con el **repositorio de datos privado** de Hugging Face que consume el cuaderno de Colab.

### Flujo de datos

```
Cuaderno de Colab (Thesis-Project)            Hugging Face                         Anotadores
──────────────────────────────────            ─────────────────────────────        ──────────
5.1 genera simulation.csv  ──publica──▶  Dataset privado  <usuario>/thesis-simulation
                                           ├── simulation.csv  ───lee───▶  Space Thesis-Triage ◀── clasifican
                                           ├── manual/<anotador>.csv  ◀──sincroniza cada minuto──┘
5.4 recupera manual/*.csv  ◀──descarga──   └── consent/<anotador>.json
5.5 comparativa de tiempos · 5.6 análisis de errores
```

---

## 📁 Estructura del Repositorio

```
.
├── app.py               # Aplicación Gradio: pantallas de bienvenida, triaje y resumen; persistencia y sincronización
├── requirements.txt     # Dependencias adicionales a Gradio (huggingface_hub)
├── README.md            # Este archivo (su cabecera YAML configura el Space)
├── assets/
│   └── images/          # Logo institucional y foto del autor
└── data/                # ⛔ se crea en ejecución y no se versiona: carpeta local sincronizada con el dataset
    ├── manual/          #    <anotador>.csv  → id, orden_llegada, etiqueta_humana, segundos
    └── consent/         #    <anotador>.json → constancia de aceptación del consentimiento
```

---

## 🖥️ Pantallas de la Plataforma

| Pantalla | Qué muestra | Equivalente en la versión de escritorio |
|---|---|---|
| **1. Bienvenida** | Consentimiento informado, casilla de aceptación, identificador del anotador y guía de los tres niveles de urgencia | `_showWelcome` |
| **2. Triaje** | Progreso (`Correo n de N`), tarjeta con asunto y mensaje, botones **Alta / Media / Baja**, **Saltar** y **Guardar y salir** | `_showTriage` + `_decide` |
| **3. Resumen** | Clasificados en la sesión, progreso acumulado, tiempo total y promedio por correo, botones **Continuar** y **Otro anotador** | `_showSummary` |

Atajos de teclado: **A** (Alta), **M** (Media), **B** (Baja) y **S** (Saltar), activos solo cuando no se está escribiendo en un campo de texto.

### Guía de clasificación mostrada al anotador

| Nivel | Criterio |
|---|---|
| 🔴 **Alta** | Riesgo a personas o infraestructura, corte total, emergencia que requiere atención inmediata. |
| 🟡 **Media** | Afectación parcial del servicio o inconformidad operativa que debe atenderse en el corto plazo. |
| 🟢 **Baja** | Consulta, trámite administrativo o solicitud sin afectación del servicio. |

---

## 🔒 Persistencia y Ética

- **Archivo por anotador:** cada decisión se agrega de inmediato a `data/manual/<anotador>.csv`; el identificador se normaliza a un *slug* seguro (sin tildes, minúsculas, guiones bajos), p. ej. `Andrés Rubiano` → `andres_rubiano.csv`.
- **Sincronización:** un `CommitScheduler` de `huggingface_hub` sube la carpeta `data/` al dataset privado **cada minuto**; al iniciar sesión se recupera el archivo del anotador si ya existe en el repositorio, lo que permite reanudar desde otro dispositivo.
- **Consentimiento informado:** no se puede comenzar sin aceptarlo; la aceptación queda en `data/consent/<anotador>.json` con fecha y hora. El texto reproduce el formato de consentimiento del apéndice de la tesis (objetivo del estudio, naturaleza sintética de los correos, datos que se registran, uso académico y seudonimizado, voluntariedad y retiro), en conformidad con la **Ley 1581 de 2012** de Protección de Datos Personales de Colombia.
- **Cronómetro:** se mide en el servidor entre la presentación del correo y el clic de decisión; incluye la latencia de red de cada clic, despreciable frente a decisiones humanas de varios segundos.

---

## ⚙️ Requisitos

| Requisito | Versión / Detalle |
|-----------|-------------------|
| **Gradio** | 5.29.0 (fijada en la cabecera YAML de este README; la instala el Space) |
| **huggingface_hub** | ≥ 0.30 ([`requirements.txt`](requirements.txt)) |
| **Hardware del Space** | CPU basic (gratuito); la aplicación no ejecuta modelos |
| **Dataset** | Repositorio de datos privado con `simulation.csv` en la raíz (lo publica la sección 5.2 del cuaderno de Colab) |

### Configuración del Space

| Ajuste | Dónde | Valor |
|---|---|---|
| `HF_TOKEN` | Settings → Variables and secrets → **Secret** | Token de Hugging Face con permiso *write* |
| `DATASET_REPO` | Settings → Variables and secrets → **Variable** | `<usuario>/thesis-simulation` |

---

## 🚀 Publicación en Hugging Face

1. **Dataset:** crear el repositorio de datos `thesis-simulation` como **privado** y publicar `simulation.csv` (desde la celda 5.2 del cuaderno o subiéndolo a mano).
2. **Token:** Settings → Access Tokens → *Create new token* de tipo **Write**.
3. **Space:** *New → Space*, SDK **Gradio**, hardware **CPU basic**, visibilidad **pública** (los anotadores entran sin cuenta; los datos viven en el dataset privado).
4. **Archivos:** subir `app.py`, `requirements.txt`, `README.md` y la carpeta `assets/` (o hacer `git push` al remoto del Space).
5. **Secretos:** configurar `HF_TOKEN` y `DATASET_REPO` según la tabla anterior.
6. **Arranque:** el Space se construye solo en uno o dos minutos. Si el registro muestra `Falta el secreto HF_TOKEN` o no encuentra `simulation.csv`, revisar los pasos 5 y 1.
7. **Anotadores:** compartir la URL del Space; cada persona acepta el consentimiento, se identifica y clasifica. Puede cerrar y volver después con el mismo identificador.

### Ejecución local (opcional)

```bash
pip install "gradio==5.29.0" "huggingface_hub>=0.30"
set HF_TOKEN=hf_...                      # Windows (PowerShell: $env:HF_TOKEN="hf_...")
set DATASET_REPO=<usuario>/thesis-simulation
python app.py
```

---

## 🔗 Recursos

- [Thesis-Project — código del pipeline y cuaderno de Colab](https://github.com/RubianoAndy/Thesis-Project)
- [Gradio — documentación](https://www.gradio.app/docs)
- [huggingface_hub — `CommitScheduler`](https://huggingface.co/docs/huggingface_hub/guides/upload#scheduled-uploads)
- [Hugging Face Spaces — configuración del README](https://huggingface.co/docs/hub/spaces-config-reference)

---

## 🔑 Palabras Clave

`Triaje Manual` · `Anotación Humana` · `Entorno Operativo Simulado` · `Gradio` · `Hugging Face Spaces` · `Clasificación de Correos Electrónicos` · `Priorización de Incidentes` · `Sector Energético` · `Consentimiento Informado`

---

## 📧 Contacto

**Andrés Giovanny Rubiano Muñoz**
Maestría en Inteligencia Artificial · Universidad de La Salle
arubiano67@unisalle.edu.co

---

## 📄 Derechos Reservados

© 2026 Andrés Giovanny Rubiano Muñoz (Andy Rubiano). Todos los derechos reservados.

Este proyecto de grado y su contenido —investigación, código, plantillas, metodologías y documentación— son propiedad intelectual conjunta de:

- **Andrés Giovanny Rubiano Muñoz** (Andy Rubiano) — Autor
- **Universidad de La Salle** — Institución académica

El uso, reproducción o distribución requiere autorización previa escrita de los titulares de derechos.

---

<div align="center">
  Universidad de La Salle | Bogotá D. C., Colombia
</div>
