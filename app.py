"""
=============================================================================
Plataforma web de triaje manual cronometrado (Gradio · desplegada en Render)
=============================================================================
Versión web de `src/simulation/3-manual_triage.py`: presenta el MISMO lote de
PQRs que procesó el modelo, A CIEGAS (sin predicción del modelo ni etiqueta
de oro), en ORDEN DE LLEGADA (FIFO), y cronometra el tiempo que tarda la
persona en decidir el nivel de urgencia de cada mensaje.

Diferencias con la versión de escritorio:
  - El lote se lee del repositorio de datos privado de Hugging Face
    (`simulation.csv`, publicado desde el cuaderno de Colab).
  - Cada registro se escribe en `manual/<anotador>.csv` y un `CommitScheduler`
    sincroniza esa carpeta con el mismo repositorio de datos cada minuto, de
    donde el cuaderno de Colab recupera los triajes.
  - Antes de empezar, el anotador lee y acepta el consentimiento informado; la
    aceptación queda registrada en `consent/<anotador>.json`.
  - Reanudable desde cualquier dispositivo: al iniciar sesión se recupera el
    archivo del anotador si ya existe en el repositorio.
  - Al terminar el lote, el anotador responde el cuestionario de uso y
    apropiación de la interfaz (SUS, nueve ítems adaptados de TAM y dos
    preguntas abiertas), que queda en `questionnaire/<anotador>.json`.

Variables de entorno del servicio (Render):
  HF_TOKEN      secreto · token de Hugging Face con permiso de escritura.
  DATASET_REPO  variable · p. ej. "RubianoAndy/thesis-simulation".
  PORT          la asigna Render; en local se usa 7860.

Salidas: manual/<anotador>.csv         (columnas: id, orden_llegada, etiqueta_humana, segundos)
         consent/<anotador>.json       (constancia de aceptación del consentimiento)
         questionnaire/<anotador>.json (respuestas del cuestionario y puntajes)
=============================================================================
"""
import base64
import csv
import json
import os
import re
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from huggingface_hub import CommitScheduler, hf_hub_download

# --- Configuración -----------------------------------------------------------
HF_TOKEN     = os.environ.get("HF_TOKEN")
DATASET_REPO = os.environ.get("DATASET_REPO", "RubianoAndy/thesis-simulation")
LOT_FILE     = os.environ.get("LOT_FILE")   # solo para pruebas locales: ruta a un simulation.csv (sin sincronizar)

DATA_DIR          = Path("data")            # carpeta local sincronizada con el dataset
MANUAL_DIR        = DATA_DIR / "manual"
CONSENT_DIR       = DATA_DIR / "consent"
QUESTIONNAIRE_DIR = DATA_DIR / "questionnaire"
for _d in (MANUAL_DIR, CONSENT_DIR, QUESTIONNAIRE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

KEYS = {"a": "Alta", "m": "Media", "b": "Baja"}
OUTPUT_FIELDS = ["id", "orden_llegada", "etiqueta_humana", "segundos"]

PROGRAM = "Maestría en Inteligencia Artificial"
FACULTY = "Facultad de Ingeniería · Universidad de La Salle"
AUTHOR  = "Andrés Giovanny Rubiano Muñoz"
TITLE   = ("Triage automático de PQRs por nivel de urgencia en el sector energético "
           "con aprendizaje profundo a partir de un dataset sintético")

# --- Instrumento de uso y apropiación de la interfaz (sección 6.6.4 de la tesis) ---
# Escala de respuesta común a las partes A y B: Likert de 5 puntos.
LIKERT = ["1", "2", "3", "4", "5"]
LIKERT_LEGEND = [("1", "Totalmente en desacuerdo"), ("2", "En desacuerdo"), ("3", "Ni de acuerdo ni en desacuerdo"),
                 ("4", "De acuerdo"), ("5", "Totalmente de acuerdo")]

# Parte A · Escala de Usabilidad del Sistema (SUS): impares en sentido positivo, pares en sentido negativo.
SUS_ITEMS = [
    "Creo que me gustaría utilizar esta plataforma con frecuencia.",
    "Encontré la plataforma innecesariamente compleja.",
    "Pensé que la plataforma era fácil de usar.",
    "Creo que necesitaría el apoyo de una persona con conocimientos técnicos para poder utilizar esta plataforma.",
    "Encontré que las diversas funciones de la plataforma estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en la plataforma.",
    "Imagino que la mayoría de las personas aprenderían a utilizar esta plataforma muy rápidamente.",
    "Encontré la plataforma muy incómoda de utilizar.",
    "Me sentí muy seguro(a) utilizando la plataforma.",
    "Necesité aprender muchas cosas antes de poder empezar a utilizar esta plataforma.",
]

# Parte B · Cuestionario de apropiación (adaptado de TAM): (código, dimensión, afirmación).
TAM_DIMENSIONS = {"UP": "Utilidad percibida", "FU": "Facilidad de uso percibida", "IU": "Intención de uso"}
TAM_ITEMS = [
    ("UP1", "Usar esta herramienta me permitiría identificar más rápido las PQRs críticas."),
    ("UP2", "Contar con una cola de PQRs priorizada por nivel de urgencia mejoraría mi desempeño al atender las solicitudes."),
    ("UP3", "Considero que esta herramienta es útil para el triage de PQRs del sector energético."),
    ("FU1", "Aprender a usar la plataforma me resultó fácil."),
    ("FU2", "La información que presenta la plataforma es clara y fácil de entender."),
    ("FU3", "Me resultó sencillo clasificar las PQRs con la plataforma."),
    ("IU1", "Usaría esta herramienta en mi trabajo si estuviera disponible."),
    ("IU2", "Recomendaría esta herramienta a otras personas que gestionan PQRs."),
    ("IU3", "Estaría dispuesto(a) a apoyarme en una priorización automática de las PQRs, siempre que una persona "
            "supervise los casos críticos."),
]

# Parte C · Preguntas abiertas: (clave en el JSON, pregunta).
OPEN_QUESTIONS = [
    ("mejoras", "¿Qué aspectos de la plataforma mejoraría?"),
    ("desconfianza", "¿En qué situaciones no confiaría en la priorización de PQRs propuesta por el modelo?"),
]


# --- Utilidades del identificador (idénticas a la versión de escritorio) -----
def format_display_name(raw):
    """Inicial de cada palabra en mayúscula, conservando tildes."""
    return raw.strip().title()


def slugify(name):
    """Identificador seguro para el archivo: sin tildes, minúsculas, guiones bajos."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")


# --- Lote --------------------------------------------------------------------
def load_lot():
    """Descarga el lote del repositorio de datos y lo carga en orden FIFO."""
    if LOT_FILE:
        path = LOT_FILE          # modo local: lote desde un archivo, sin tocar el repositorio
    elif not HF_TOKEN:
        raise RuntimeError("Falta la variable de entorno HF_TOKEN en la configuración del servicio.")
    else:
        path = hf_hub_download(DATASET_REPO, "simulation.csv", repo_type="dataset", token=HF_TOKEN)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, 1):
        r["orden_llegada"] = i
    return rows


LOT = load_lot()

# --- Sincronización con el repositorio de datos -------------------------------
class _LocalScheduler:
    """Sustituto del CommitScheduler en modo local: solo aporta el candado, no sincroniza nada."""
    def __init__(self):
        import threading
        self.lock = threading.Lock()


scheduler = _LocalScheduler() if LOT_FILE else CommitScheduler(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    folder_path=DATA_DIR,
    path_in_repo="",
    every=1,                 # minutos entre sincronizaciones
    private=True,
    token=HF_TOKEN,
)


def fetch_existing(slug):
    """Recupera del repositorio el CSV del anotador si existe (reanudación entre dispositivos)."""
    local = MANUAL_DIR / f"{slug}.csv"
    if local.exists() or LOT_FILE:
        return
    try:
        remote = hf_hub_download(DATASET_REPO, f"manual/{slug}.csv", repo_type="dataset",
                                 token=HF_TOKEN, force_download=True)
        shutil.copy(remote, local)
    except Exception:
        pass                 # no existe todavía: anotador nuevo


def load_done(slug):
    """Ids ya clasificados por el anotador (para reanudar sin repetir)."""
    path = MANUAL_DIR / f"{slug}.csv"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def append_record(slug, record):
    """Agrega un registro al CSV del anotador (bajo el candado del sincronizador)."""
    path = MANUAL_DIR / f"{slug}.csv"
    with scheduler.lock:
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(record)


def record_consent(display, slug):
    """Deja constancia de la aceptación del consentimiento informado (una vez por anotador)."""
    path = CONSENT_DIR / f"{slug}.json"
    if path.exists():
        return
    with scheduler.lock:
        path.write_text(json.dumps({
            "anotador": display,
            "identificador": slug,
            "consentimiento_aceptado": True,
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_correos_lote": len(LOT),
        }, ensure_ascii=False, indent=2), encoding="utf-8")


# --- Cuestionario de uso y apropiación ----------------------------------------
def fetch_existing_questionnaire(slug):
    """Recupera del repositorio el cuestionario del anotador si ya lo respondió (p. ej. tras un reinicio)."""
    local = QUESTIONNAIRE_DIR / f"{slug}.json"
    if local.exists() or LOT_FILE:
        return
    try:
        remote = hf_hub_download(DATASET_REPO, f"questionnaire/{slug}.json", repo_type="dataset",
                                 token=HF_TOKEN, force_download=True)
        shutil.copy(remote, local)
    except Exception:
        pass                 # todavía no lo ha respondido


def questionnaire_done(slug):
    """True si el anotador ya respondió el cuestionario."""
    return (QUESTIONNAIRE_DIR / f"{slug}.json").exists()


def sus_score(answers):
    """Puntaje SUS (0–100): a las impares se les resta 1, las pares se restan de 5; suma × 2,5."""
    contributions = [(a - 1) if n % 2 == 1 else (5 - a) for n, a in enumerate(answers, 1)]
    return round(sum(contributions) * 2.5, 1)


def record_questionnaire(state, sus_answers, tam_answers, open_answers):
    """Guarda las respuestas y los puntajes en questionnaire/<anotador>.json (una vez por anotador)."""
    path = QUESTIONNAIRE_DIR / f"{state['slug']}.json"
    if path.exists():
        return
    sus = {str(n): a for n, a in enumerate(sus_answers, 1)}
    tam = {code: a for (code, _), a in zip(TAM_ITEMS, tam_answers)}
    dims = {}
    for prefix, name in TAM_DIMENSIONS.items():
        values = [a for code, a in tam.items() if code.startswith(prefix)]
        dims[name] = round(sum(values) / len(values), 2)
    with scheduler.lock:
        path.write_text(json.dumps({
            "anotador": state["display"],
            "identificador": state["slug"],
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "escala": "Likert de 5 puntos (1 = totalmente en desacuerdo, 5 = totalmente de acuerdo)",
            "sus": sus,
            "puntaje_sus": sus_score(sus_answers),
            "apropiacion": tam,
            "promedio_dimension": dims,
            "abiertas": {key: text for (key, _), text in zip(OPEN_QUESTIONS, open_answers)},
        }, ensure_ascii=False, indent=2), encoding="utf-8")


# --- Textos ------------------------------------------------------------------
CONSENT_MD = f"""
### Consentimiento informado para anotadores

**Proyecto de grado:** {TITLE}.
**Investigador:** {AUTHOR}. **Directora:** Paola Andrea Zambrano García.

**Propósito.** El ejercicio forma parte de una investigación que compara la clasificación automática de
PQRs, realizada por un modelo de inteligencia artificial, frente a la clasificación hecha por personas. Su
tarea consiste en leer, una a una, las PQRs de un lote y asignar a cada una un nivel de urgencia (Alta,
Media o Baja). La plataforma presenta las PQRs en orden de llegada y mide el tiempo empleado en cada
decisión. Al finalizar, usted responde un cuestionario breve sobre su experiencia de uso y la utilidad que
percibe de la plataforma, compuesto por una escala de usabilidad de diez afirmaciones, nueve afirmaciones
valoradas en una escala de cinco puntos y dos preguntas abiertas.

**Naturaleza de las PQRs.** Todas las PQRs son hipotéticas y fueron generadas de forma sintética para
esta investigación; no corresponden a comunicaciones reales de ninguna empresa ni contienen datos
personales de clientes, usuarios o terceros. La participación es individual y a ciegas: usted no conoce la
predicción del modelo ni la etiqueta correcta, y no existen respuestas que lo afecten; el objetivo es
evaluar el sistema, no al participante. El cuestionario se diligencia una vez finalizada la clasificación
y antes de conocer el desempeño del modelo.

**Datos que se registran.** El identificador que usted proporcione (para guardar y reanudar la sesión),
el nivel de urgencia asignado a cada PQR, el tiempo empleado en cada decisión, el tiempo total de la
sesión y las respuestas al cuestionario de usabilidad y apropiación, incluidos los comentarios en texto
libre. No se recolectan datos sensibles; el identificador se seudonimiza antes de reportar cualquier
resultado.

**Uso de la información.** Exclusivamente académico, dentro de este proyecto de grado y las publicaciones
que de él se deriven. Las decisiones y las respuestas se presentan siempre de forma agregada y bajo
seudónimo, con acceso restringido al investigador y a la directora, y pueden eliminarse a solicitud del
participante.

**Voluntariedad.** La participación es libre y voluntaria; puede abstenerse de responder o retirarse en
cualquier momento sin justificación ni consecuencia alguna, y puede pedir que sus decisiones y respuestas
se excluyan del análisis aun después de finalizar la clasificación y el cuestionario. No hay compensación
económica ni riesgos más allá del tiempo dedicado. Este tratamiento se ajusta a la Ley 1581 de 2012 de
Protección de Datos Personales de Colombia.
"""

# --- Recursos visuales (misma identidad que la app de escritorio) ------------
def _logo_data_uri():
    """Logo institucional (versión para fondo oscuro) incrustado como data URI."""
    try:
        raw = (Path(__file__).parent / "assets" / "images" / "UnisalleDarkLogo.png").read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    except Exception:
        return ""


LOGO_URI = _logo_data_uri()


STEPS = ["Consentimiento", "Triaje", "Cuestionario", "Resumen"]


def steps_html(current):
    """Indicador de pasos de la cabecera: el paso activo en dorado, los completados con marca."""
    items = []
    for n, label in enumerate(STEPS, 1):
        cls = "done" if n < current else ("active" if n == current else "")
        mark = "✓" if n < current else str(n)
        items.append(f"<div class='step {cls}'><span class='step-n'>{mark}</span><span class='step-l'>{label}</span></div>")
    return "<div class='steps'>" + "<div class='step-sep'></div>".join(items) + "</div>"


def header_html(title, subtitle, step=1):
    """Cabecera institucional de borde a borde: logo, bloque académico, título y pasos del proceso."""
    logo = f'<img class="logo" src="{LOGO_URI}" alt="Universidad de La Salle">' if LOGO_URI else ""
    return f"""
<div class="header">
  <div class="header-top">
    <div class="header-logo">{logo}</div>
    <div class="header-meta">
      <div class="program">{PROGRAM}</div>
      <div class="faculty">{FACULTY}</div>
    </div>
  </div>
  <div class="header-main">
    <div class="header-text">
      <div class="eyebrow">Proyecto de grado &nbsp;·&nbsp; {AUTHOR}</div>
      <div class="title">{escape(title)}</div>
      <div class="subtitle">{escape(subtitle)}</div>
    </div>
    {steps_html(step)}
  </div>
</div>"""


GUIDE_HTML = """
<div class="guide">
  <div class="guide-title">Niveles de urgencia</div>
  <div class="guide-sub">Clasifique cada PQR según el impacto que describe.</div>
  <div class="guide-row alta"><div class="guide-head"><span class="lvl">Alta</span><kbd>A</kbd></div>
    <div class="guide-desc">Riesgo a personas o infraestructura, corte total, emergencia que requiere atención inmediata.</div></div>
  <div class="guide-row media"><div class="guide-head"><span class="lvl">Media</span><kbd>M</kbd></div>
    <div class="guide-desc">Afectación parcial del servicio o inconformidad operativa que debe atenderse en el corto plazo.</div></div>
  <div class="guide-row baja"><div class="guide-head"><span class="lvl">Baja</span><kbd>B</kbd></div>
    <div class="guide-desc">Consulta, trámite administrativo o solicitud sin afectación del servicio.</div></div>
</div>
<div class="note"><span class="note-icon">⏱</span> El tiempo se cronometra desde que aparece cada PQR hasta su decisión.
  Las PQRs se presentan a ciegas y en orden de llegada; puede cerrar y reanudar después con el mismo identificador.</div>
"""

FOOTER_HTML = ("<div class='footer'><span class='footer-brand'>Universidad de La Salle</span>"
               "<span class='footer-sep'></span>Bogotá D. C., Colombia"
               "<span class='footer-sep'></span>Andrés Giovanny Rubiano Muñoz"
               "<span class='footer-sep'></span>2026 © Todos los derechos reservados</div>")


SURVEY_INTRO_HTML = """
<div class='section-label'>Cuestionario de uso y apropiación de la interfaz
  <small>Su opinión sobre la plataforma con la que acaba de clasificar las PQRs.</small></div>
<div class='survey-intro'>
  Indique, para cada afirmación, su grado de acuerdo en una escala de 1 a 5. No hay respuestas correctas ni
  incorrectas: se evalúa la plataforma, no al participante. Las respuestas se guardan bajo el mismo
  identificador de sus decisiones y se amparan en el consentimiento informado que aceptó al inicio.
</div>
<div class='scale-legend'>""" + "".join(
    f"<div class='scale-item'><span class='scale-n'>{n}</span><span class='scale-l'>{label}</span></div>"
    for n, label in LIKERT_LEGEND) + "</div>"


def part_html(letter, title, subtitle):
    """Encabezado de cada parte del cuestionario."""
    return (f"<div class='part'><span class='part-n'>Parte {letter}</span>"
            f"<span class='part-t'>{title}</span><span class='part-s'>{subtitle}</span></div>")


def dimension_html(name):
    """Rótulo de dimensión dentro de la parte B."""
    return f"<div class='dimension'>{name}</div>"


def progress_html(pos, total):
    """«PQR n de N», porcentaje y barra de progreso dorada."""
    pct = pos / total * 100 if total else 0
    return (f"<div class='progress'><div class='progress-head'><div class='progress-text'>PQR "
            f"<strong>{pos}</strong> de {total}</div><div class='progress-pct'>{pct:.0f} %</div></div>"
            f"<div class='progress-track'><div class='progress-bar' style='width:{pct:.1f}%'></div></div></div>")


# Tipografías institucionales: serif para títulos (tono académico) y sans para la lectura.
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
"""

CSS = """
:root {
  --navy:#002E5B; --navy-deep:#001F40; --navy-soft:#0B3F76; --gold:#F2A900; --gold-soft:#FFF4D6;
  --bg:#F4F6F9; --card:#FFFFFF; --ink:#13263B; --muted:#5B6B7E; --faint:#8A97A8; --border:#D9E0E8; --track:#E3E8EF;
  --alta:#C8102E; --alta-soft:#FBE9EC; --media:#E09A00; --media-soft:#FFF4D6; --baja:#1B7F4A; --baja-soft:#E6F4EC;
  --serif:"Source Serif 4", Georgia, "Times New Roman", serif;
  --sans:"Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}
body, .gradio-container, .gradio-container .main { background:var(--bg) !important; color:var(--ink);
       font-family:var(--sans); }
.gradio-container { max-width:100% !important; width:100% !important; margin:0 !important; padding:0 !important;
                    overflow-x:hidden; }
.gradio-container *, .gradio-container *::before, .gradio-container *::after { border-radius:0 !important; }
footer { display:none !important; }
/* El contenedor principal de Gradio limita el ancho a 1280 px y añade relleno: se anula para ir de borde a borde */
.gradio-container > main, main.app, .gradio-container .main { max-width:100% !important; width:100% !important;
       margin:0 !important; padding:0 !important; }
.gradio-container .main > .wrap, .gradio-container .contain { padding:0 !important; }
.html-container.padding { padding:0 !important; }
.gap { gap:0 !important; }
.block, .form { background:transparent !important; border:none !important; box-shadow:none !important; }

/* ---- Ajuste al alto de la ventana ---------------------------------------
   Toda la cadena de contenedores de Gradio es una columna flex: se fija la raíz a 100vh y se deja que la
   pantalla activa crezca; dentro de ella, solo el consentimiento y el cuerpo de la PQR se desplazan. */
html, body { height:100%; overflow:hidden; }
gradio-app, .gradio-container { height:100vh !important; min-height:100vh !important; max-height:100vh !important;
                                overflow:hidden !important; }
.gradio-container > main, main.app, .gradio-container .main, .gradio-container .main > .wrap,
.gradio-container .contain, .gradio-container .contain > .column { flex:1 1 auto !important; min-height:0 !important;
                                                                   height:auto !important; }
#hdr, #ftr { flex:none !important; }                       /* cabecera y pie nunca se encogen */
.welcome, .wide, .narrow { flex:1 1 0px !important; min-height:0 !important; }
.welcome > .row { flex:1 1 0px !important; min-height:0 !important; align-items:stretch !important;
                  flex-wrap:nowrap !important; }   /* con wrap, la fila crecería hasta el contenido más alto */
.welcome > .row > .column { min-height:0 !important; overflow:hidden; flex-wrap:nowrap !important; }
.welcome > .row > .column > .block:not(.consent-box) { flex:none !important; overflow:visible !important; }  /* solo cede el consentimiento */
#email-box, #email-box .html-container, #email-box .prose { flex:1 1 auto !important; min-height:0 !important;
       display:flex !important; flex-direction:column !important; max-height:none !important; }

/* ---- Cabecera institucional ------------------------------------------------ */
.header { background:linear-gradient(135deg, var(--navy-deep) 0%, var(--navy) 60%, var(--navy-soft) 100%);
          color:#F0F4F8; border-bottom:4px solid var(--gold); }
.header-top { display:flex; align-items:center; justify-content:space-between; gap:24px; padding:10px 40px 9px;
              border-bottom:1px solid rgba(255,255,255,.12); }
.header-logo .logo { height:52px; object-fit:contain; display:block; }
.header-meta { text-align:right; }
.header .program { font-family:var(--serif); font-weight:600; font-size:15px; color:#FFFFFF; letter-spacing:.01em; }
.header .faculty { font-size:11.5px; color:#A8BDD3; margin-top:2px; }
.header-main { display:flex; align-items:center; justify-content:space-between; gap:32px; padding:14px 40px 16px; }
.header .eyebrow { font-size:11px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--gold); }
.header .title { font-family:var(--serif); font-size:26px; font-weight:700; color:#FFFFFF; margin-top:4px; line-height:1.15; }
.header .subtitle { font-size:13px; color:#A8BDD3; margin-top:6px; }

.steps { display:flex; align-items:center; gap:10px; flex:none; }
.step { display:flex; align-items:center; gap:9px; font-size:12.5px; font-weight:500; white-space:nowrap; }
.step .step-l { color:#8DA4BD !important; }
.step-n { width:28px; height:28px; display:inline-flex; align-items:center; justify-content:center;
          border:1.5px solid #4B6A8E; color:#A8BDD3 !important; font-size:12px; font-weight:700; }
.step.active .step-l { color:#FFFFFF !important; font-weight:600; }
.step.active .step-n { background:var(--gold); border-color:var(--gold); color:var(--navy-deep) !important; }
.step.done .step-n { background:transparent; border-color:var(--gold); color:var(--gold) !important; }
.step.done .step-l { color:#DCE6F2 !important; }
.step-sep { width:28px; height:1px; background:#4B6A8E; }
@media (max-width: 900px) { .steps { display:none; } .header-main { padding:16px 24px; } .header-top { padding:12px 24px; } }

/* ---- Contenedores de pantalla ------------------------------------------- */
.narrow { max-width:760px; margin:0 auto; padding:32px 24px 12px; }
.welcome { max-width:1320px; margin:0 auto; padding:24px 40px 8px; }
.wide { max-width:1320px; margin:0 auto; padding:20px 40px 8px; }
.section-label { font-family:var(--serif); font-size:18px; font-weight:700; color:var(--navy); margin:4px 0 10px;
                 padding-left:12px; border-left:4px solid var(--gold); line-height:1.2; }
.section-label small { display:block; font-family:var(--sans); font-size:12px; font-weight:400; color:var(--muted); margin-top:2px; }
.hint { color:var(--muted); font-size:12.5px; margin:8px 0 10px; }
.ok { color:var(--baja); font-size:12.5px; font-weight:500; margin:8px 0 10px; }
.ok code { background:var(--baja-soft); color:var(--baja); padding:1px 6px; font-size:12px; }

/* Consentimiento: hoja blanca con sombra suave, como un documento */
/* Gradio aplica la clase tanto al bloque exterior como al <div class="prose"> interior; el texto vive en <span class="md">.
   Toda la cadena es una columna flex y SOLO el <span class="md"> se desplaza. */
.block.consent-box { flex:1 1 0px !important; height:auto !important; max-height:none !important; min-height:0 !important;
               overflow:hidden !important; border:1px solid var(--border) !important; border-top:3px solid var(--navy) !important;
               background:var(--card) !important; padding:0 !important; box-shadow:0 1px 2px rgba(0,46,91,.06) !important;
               display:flex !important; flex-direction:column !important; }
.block.consent-box > div:not(.wrap), .prose.consent-box { flex:1 1 0px !important; min-height:0 !important; height:auto !important;
               max-height:none !important; display:flex !important; flex-direction:column !important; padding:0 !important;
               overflow:hidden !important; }
.consent-box span.md { flex:1 1 0px !important; min-height:0 !important; height:auto !important; max-height:none !important;
               display:block !important; overflow-y:auto !important; padding:8px 22px 14px !important; box-sizing:border-box; }
.consent-box * { color:var(--ink) !important; font-size:13.5px; line-height:1.6; }
.consent-box h3 { font-family:var(--serif) !important; font-size:17px !important; color:var(--navy) !important;
                  margin:12px 0 6px !important; }
.consent-box strong { color:var(--navy) !important; }
.consent-check { flex:none !important; }
.consent-check label { font-size:13.5px !important; color:var(--ink) !important; font-weight:500 !important; line-height:1.3; }
/* Solo se cambia el color de fondo: la imagen del chulito la pone Gradio (background-image) y debe conservarse */
.consent-check input[type=checkbox] { accent-color:var(--navy); width:18px !important; height:18px !important;
       border:1.5px solid var(--navy) !important; background-color:#FFFFFF; flex:none; }
.consent-check input[type=checkbox]:checked { background-color:var(--navy) !important; border-color:var(--navy) !important; }

/* Identificador */
#name-box textarea, #name-box input { font-size:15px !important; border:1.5px solid var(--border) !important;
       background:var(--card) !important; color:var(--ink) !important; padding:12px 14px !important; }
#name-box textarea:focus, #name-box input:focus { border-color:var(--navy) !important; box-shadow:0 0 0 3px rgba(0,46,91,.12) !important; }

/* Guía de niveles */
.guide { background:var(--card); border:1px solid var(--border); padding:18px 22px 10px; margin-top:6px;
         box-shadow:0 1px 2px rgba(0,46,91,.06); }
.guide-title { font-family:var(--serif); font-weight:700; font-size:16px; color:var(--navy); }
.guide-sub { font-size:12.5px; color:var(--muted); margin:2px 0 14px; }
.guide-row { border-left:5px solid; padding:10px 14px; margin:0 0 10px; background:var(--bg); }
.guide-row.alta { border-color:var(--alta); } .guide-row.media { border-color:var(--media); } .guide-row.baja { border-color:var(--baja); }
.guide-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:3px; }
.guide-head .lvl { font-weight:700; font-size:14px; letter-spacing:.06em; text-transform:uppercase; }
.guide-row.alta .lvl { color:var(--alta); } .guide-row.media .lvl { color:var(--media); } .guide-row.baja .lvl { color:var(--baja); }
.guide-desc { font-size:13px; color:var(--ink); line-height:1.5; }
kbd { display:inline-flex; align-items:center; justify-content:center; min-width:24px; height:22px; padding:0 6px;
      font-family:var(--sans); font-size:11.5px; font-weight:700; color:var(--navy); background:#FFFFFF;
      border:1px solid var(--border); border-bottom-width:2px; }
.note { display:flex; gap:10px; align-items:flex-start; background:var(--gold-soft); border-left:4px solid var(--gold);
        color:var(--ink); font-size:12.5px; line-height:1.5; padding:10px 14px; margin-top:12px; }
.note-icon { font-size:15px; line-height:1.3; }

/* ---- Triaje -------------------------------------------------------------- */
.progress { margin:0 0 14px; }
.progress-head { display:flex; justify-content:space-between; align-items:baseline; }
.progress-text { font-size:14px; color:var(--muted); }
.progress-text strong { font-family:var(--serif); font-size:22px; color:var(--navy); margin-right:2px; }
.progress-pct { font-size:12.5px; font-weight:600; color:var(--navy); letter-spacing:.04em; }
.progress-track { height:8px; background:var(--track); margin-top:8px; }
.progress-bar { height:100%; background:linear-gradient(90deg, var(--gold), #FFC43D); transition:width .3s ease; }

.email-card, .email-card * { color:var(--ink) !important; }
.email-card { background:var(--card); border:1px solid var(--border); box-shadow:0 2px 8px rgba(0,46,91,.08); overflow-wrap:anywhere;
              display:flex; flex-direction:column; flex:1 1 auto; min-height:0; }
.email-head { background:var(--navy); color:#FFFFFF !important; padding:10px 24px; display:flex; justify-content:space-between;
              align-items:center; font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; font-weight:600; }
.email-head, .email-head * { color:#DCE6F2 !important; }
.email-head .tag { color:var(--gold) !important; }
.email-meta { padding:18px 24px 14px; border-bottom:1px solid var(--border); }
.email-card .label { font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--faint) !important; }
.email-card .subject { font-family:var(--serif); font-size:23px; font-weight:700; line-height:1.3; margin:6px 0 0; color:var(--navy) !important; }
.email-card .body { white-space:pre-wrap; font-size:15.5px; line-height:1.7; padding:18px 24px 22px;
                    flex:1 1 auto; min-height:0; overflow-y:auto; }

/* ---- Cuestionario -------------------------------------------------------- */
/* La pantalla ocupa el alto disponible y se desplaza por dentro, como el consentimiento */
.survey { max-width:1000px; margin:0 auto; padding:24px 40px 8px; width:100%; box-sizing:border-box;
          flex:1 1 0px !important; min-height:0 !important; overflow-y:auto !important; }
/* Los hijos no se encogen ni se desplazan por separado: todo el contenido crece y se desplaza la columna */
.survey > * , .survey .form, .survey .block { flex:none !important; overflow:visible !important; max-height:none !important; }
.survey .form { border:none !important; background:transparent !important; }
.survey-intro { font-size:13.5px; line-height:1.6; color:var(--ink); margin:0 0 14px; }
.scale-legend { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px 12px; background:var(--card);
                border:1px solid var(--border); border-left:4px solid var(--gold); padding:10px 16px; margin:0 0 6px; }
.scale-item { display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--ink); }
.scale-n { width:24px; height:24px; display:inline-flex; align-items:center; justify-content:center; font-weight:700;
           font-size:12px; color:var(--navy); background:var(--gold-soft); border:1px solid var(--gold); }
.part { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin:22px 0 4px; padding-bottom:8px;
        border-bottom:2px solid var(--navy); }
.part-n { font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--gold); }
.part-t { font-family:var(--serif); font-size:18px; font-weight:700; color:var(--navy); }
.part-s { font-size:12.5px; color:var(--muted); flex-basis:100%; }
.dimension { font-size:11.5px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
             margin:14px 0 0; padding:0 0 0 12px; border-left:4px solid var(--gold); }
/* Cada afirmación es una fila: enunciado a la izquierda, las cinco opciones a la derecha */
.block.likert { display:flex !important; flex-direction:row !important; align-items:center !important;
                justify-content:space-between !important; gap:20px !important; padding:10px 0 !important;
                border-bottom:1px solid var(--border) !important; }
.likert > span[data-testid="block-info"] { flex:1 1 auto; font-size:14px !important; color:var(--ink) !important;
                font-weight:500 !important; line-height:1.45 !important; margin:0 !important; padding:0 !important;
                display:block !important; background:transparent !important; }
.likert .wrap { display:flex !important; flex:none !important; gap:6px !important; flex-wrap:nowrap !important; }
.likert .wrap label { display:inline-flex !important; align-items:center; justify-content:center; width:42px; height:38px;
                      margin:0 !important; padding:0 !important; background:var(--card) !important; color:var(--navy) !important;
                      border:1.5px solid var(--border) !important; font-weight:700 !important; font-size:13.5px !important;
                      cursor:pointer; box-shadow:none !important; }
.likert .wrap label:hover { border-color:var(--navy) !important; }
.likert .wrap label.selected { background:var(--navy) !important; border-color:var(--navy) !important; color:#FFFFFF !important; }
.likert .wrap label input[type=radio] { display:none !important; }
.likert .wrap label span { margin:0 !important; color:inherit !important; }
/* Preguntas abiertas */
.open-q label > span { font-size:14px !important; color:var(--ink) !important; font-weight:500 !important; margin-bottom:6px !important;
                       background:transparent !important; padding:0 !important; }
.open-q textarea { font-size:14px !important; line-height:1.5 !important; border:1.5px solid var(--border) !important;
                   background:var(--card) !important; color:var(--ink) !important; padding:10px 12px !important; }
.open-q textarea:focus { border-color:var(--navy) !important; box-shadow:0 0 0 3px rgba(0,46,91,.12) !important; }
#btn-submit { background:var(--gold) !important; color:var(--navy-deep) !important; font-weight:700; font-size:15px;
              border:none !important; min-height:48px; max-width:320px; box-shadow:0 2px 0 #C98A00; }
#btn-submit:hover { filter:brightness(1.05); }
#btn-later { background:var(--card) !important; color:var(--muted) !important; font-weight:600;
             border:1.5px solid var(--border) !important; max-width:260px; min-height:44px; }
#btn-later:hover { border-color:var(--navy) !important; color:var(--navy) !important; }
.survey-actions { justify-content:space-between; margin:20px 0 8px !important; }
@media (max-width: 760px) {
  .block.likert { flex-direction:column !important; align-items:flex-start !important; gap:8px !important; }
  .survey { padding:16px 20px 8px; }
}

/* ---- Resumen ------------------------------------------------------------- */
.complete { display:flex; gap:12px; align-items:center; background:var(--baja-soft); border-left:4px solid var(--baja);
            color:var(--baja); font-weight:600; font-size:14px; padding:12px 16px; margin:0 0 18px; }
.stats { display:grid; grid-template-columns:repeat(2, 1fr); gap:14px; margin:4px 0 16px; }
.stat { border:1px solid var(--border); border-top:4px solid var(--gold); background:var(--card); padding:16px 20px 14px;
        box-shadow:0 1px 2px rgba(0,46,91,.06); }
.stat .value { font-family:var(--serif); font-size:32px; font-weight:700; color:var(--navy) !important; line-height:1.1; }
.stat .label { font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--muted) !important; margin-top:8px; }
.file-card { border:1px solid var(--border); border-left:4px solid var(--navy); background:var(--card); padding:14px 18px; }
.file-card .label { font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--muted) !important; }
.file-card .path { font-size:13px; color:var(--ink) !important; margin-top:6px; line-height:1.5; }
.file-card code { background:var(--bg); border:1px solid var(--border); padding:1px 6px; font-size:12.5px; color:var(--navy); }

/* ---- Pie ------------------------------------------------------------------ */
.footer { display:flex; justify-content:center; align-items:center; gap:14px; flex-wrap:wrap; color:var(--muted);
          font-size:12px; padding:12px 24px 14px; margin-top:10px; border-top:1px solid var(--border); flex:none; }
.footer-brand { font-family:var(--serif); font-weight:700; color:var(--navy); font-size:13px; }
.footer-sep { width:4px; height:4px; background:var(--gold); display:inline-block; }

/* ---- Botones (planos, esquinas rectas) ----------------------------------- */
button.lg, button.primary, button.secondary { transition:filter .15s ease, transform .05s ease; font-family:var(--sans) !important; }
button:active { transform:translateY(1px); }
#btn-start { background:var(--gold) !important; color:var(--navy-deep) !important; font-weight:700; font-size:15px;
             letter-spacing:.02em; border:none !important; min-height:50px; max-width:320px; margin:10px auto 0; display:block;
             box-shadow:0 2px 0 #C98A00; }
#btn-start:hover { filter:brightness(1.05); }
#btn-alta, #btn-media, #btn-baja { min-height:58px; font-size:16px; font-weight:700; letter-spacing:.04em; text-transform:uppercase;
             border:none !important; display:flex; align-items:center; justify-content:center; gap:12px; }
#btn-alta::after, #btn-media::after, #btn-baja::after { display:inline-flex; align-items:center; justify-content:center;
             min-width:24px; height:22px; padding:0 6px; font-size:11.5px; font-weight:700; border:1px solid rgba(255,255,255,.55);
             background:rgba(255,255,255,.14); }
#btn-alta::after  { content:"A"; } #btn-media::after { content:"M"; } #btn-baja::after  { content:"B"; }
#btn-alta  { background:var(--alta) !important; color:#FFFFFF !important; }
#btn-media { background:var(--media) !important; color:#FFFFFF !important; }
#btn-baja  { background:var(--baja) !important; color:#FFFFFF !important; }
#btn-alta:hover, #btn-media:hover, #btn-baja:hover { filter:brightness(1.08); }
#btn-skip  { background:var(--card) !important; color:var(--muted) !important; font-weight:600;
             border:1.5px solid var(--border) !important; max-width:240px; min-height:44px; white-space:nowrap; }
#btn-skip:hover { border-color:var(--navy) !important; color:var(--navy) !important; }
#btn-quit, #btn-exit { background:var(--navy) !important; color:#FFFFFF !important; font-weight:600; border:none !important; min-height:44px; }
#btn-quit { max-width:250px; margin-left:auto; }
#btn-exit { max-width:150px; margin-left:auto; }
#btn-continue { background:var(--gold) !important; color:var(--navy-deep) !important; font-weight:700; border:none !important;
                max-width:300px; min-height:44px; box-shadow:0 2px 0 #C98A00; }
.actions-row { justify-content:space-between; margin-top:16px !important; }
.decision-row { gap:12px !important; margin-top:16px !important; }

/* ---- Alturas de ventana reducidas (van al final para prevalecer sobre las reglas anteriores) ------------- */
/* Alto medio (portátiles de 768 px): versión compacta de la bienvenida para que siga cabiendo sin desplazamiento */
@media (max-height: 860px) {
  .welcome { padding-top:14px !important; }
  .section-label { font-size:16px; margin-bottom:6px; }
  .section-label small { display:none !important; }
  .hint, .ok { margin:6px 0 6px; }
  .consent-check label { font-size:12.5px !important; }
  .guide { padding:12px 18px 4px; }
  .guide-sub { display:none !important; }
  .guide-title { margin-bottom:8px; }
  .guide-row { padding:7px 12px; margin-bottom:7px; }
  .guide-desc { font-size:12.5px; }
  .note { display:none !important; }
  #btn-start { min-height:44px; margin-top:6px; }
  .header-top { padding:8px 40px 7px; }
  .header-main { padding:10px 40px 12px; }
  .header .title { font-size:23px; }
  .email-meta { padding:12px 24px 10px; }
  .email-card .subject { font-size:20px; }
}
/* Muy bajas: se vuelve al desplazamiento normal de la página */
@media (max-height: 700px) {
  html, body { overflow:auto; }
  gradio-app, .gradio-container { height:auto !important; max-height:none !important; overflow:visible !important; }
  .block.consent-box { flex:none !important; height:380px !important; }
  .email-card { flex:none !important; }
  .email-card .body { flex:none !important; height:260px !important; }
}
"""

# Al cargar: fuerza el tema claro (interfaz blanca aunque el navegador esté en modo oscuro) y activa los
# atajos de teclado A/M/B/S/Q, que pulsan los botones del triaje (solo si no se está escribiendo).
SHORTCUTS_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.toString());
    return;
  }
  document.addEventListener('keydown', (e) => {
    const tag = (document.activeElement && document.activeElement.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    const map = {a:'btn-alta', m:'btn-media', b:'btn-baja', s:'btn-skip', q:'btn-quit'};
    const id = map[e.key.toLowerCase()];
    if (!id) return;
    const btn = document.getElementById(id);
    if (btn && btn.offsetParent !== null) { e.preventDefault(); btn.click(); }
  });
}
"""


# --- Lógica de sesión ----------------------------------------------------------
def new_session(display, slug):
    """Estado de una sesión de triaje: pendientes en orden FIFO y contadores."""
    fetch_existing(slug)
    fetch_existing_questionnaire(slug)
    done = load_done(slug)
    pending = [i for i, e in enumerate(LOT) if e["id"] not in done]
    return {"display": display, "slug": slug, "pending": pending, "index": 0,
            "start": 0.0, "classified": 0, "total_seconds": 0.0}


def escape(text):
    """Escapa el texto para incrustarlo en HTML."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_email(state):
    """HTML de la PQR actual (a ciegas) y texto de progreso; arranca el cronómetro."""
    email = LOT[state["pending"][state["index"]]]
    pos, total = state["index"] + 1, len(state["pending"])
    state["start"] = time.perf_counter()   # el cronómetro arranca cuando la PQR queda visible
    subject = email["asunto"] or "(sin asunto)"
    body = email["cuerpo"] or "(sin cuerpo)"
    html = (f"<div class='email-card'>"
            f"<div class='email-head'><span>PQR entrante</span>"
            f"<span class='tag'>Orden de llegada nº {email['orden_llegada']}</span></div>"
            f"<div class='email-meta'><div class='label'>Asunto</div>"
            f"<div class='subject'>{escape(subject)}</div></div>"
            f"<div class='body'>{escape(body)}</div></div>")
    return html, progress_html(pos, total)


def load_records(slug):
    """Registros ya guardados por el anotador (para las estadísticas del resumen)."""
    path = MANUAL_DIR / f"{slug}.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_html(state, already_complete=False):
    """Resumen de la sesión: tarjetas de estadística y archivo del anotador (como en la app de escritorio)."""
    records = load_records(state["slug"])
    done, total = len(records), len(LOT)
    stats = [("PQRs clasificadas", str(state["classified"])),
             ("Progreso acumulado", f"{done} / {total}")]
    if state["classified"]:
        stats.append(("Tiempo total", f"{state['total_seconds']:.1f} s"))
        stats.append(("Promedio por PQR", f"{state['total_seconds'] / state['classified']:.2f} s"))
    if records:
        secs = sorted(float(r["segundos"]) for r in records)
        mid = len(secs) // 2
        median = secs[mid] if len(secs) % 2 else (secs[mid - 1] + secs[mid]) / 2
        counts = {k: sum(1 for r in records if r["etiqueta_humana"] == k) for k in ("Alta", "Media", "Baja")}
        stats.append(("Mediana por PQR", f"{median:.2f} s"))
        stats.append(("Distribución (Alta / Media / Baja)", f"{counts['Alta']} / {counts['Media']} / {counts['Baja']}"))
    cards = "".join(f"<div class='stat'><div class='value'>{v}</div><div class='label'>{k}</div></div>"
                    for k, v in stats)
    head = (f"<div class='complete'>✓&nbsp; El anotador «{escape(state['display'])}» ya clasificó las {total} PQRs. ¡Gracias por su participación!</div>"
            if already_complete or done >= total else "")
    if questionnaire_done(state["slug"]):
        survey = (f"<div class='file-card'><div class='label'>Cuestionario de uso y apropiación</div>"
                  f"<div class='path'>Respondido &nbsp;·&nbsp; <code>questionnaire/{state['slug']}.json</code>. ¡Gracias por su opinión!</div></div>")
    elif done >= total:
        survey = ("<div class='file-card'><div class='label'>Cuestionario de uso y apropiación</div>"
                  "<div class='path'>Pendiente &nbsp;·&nbsp; puede responderlo ahora con el botón <strong>Responder cuestionario</strong> "
                  "o más tarde ingresando con el mismo identificador.</div></div>")
    else:
        survey = ""
    return (f"{head}<div class='stats'>{cards}</div>"
            f"<div class='file-card'><div class='label'>Archivo del anotador</div>"
            f"<div class='path'><code>manual/{state['slug']}.csv</code> &nbsp;·&nbsp; se sincroniza con el repositorio de datos "
            f"en menos de un minuto; sus decisiones quedan guardadas aunque cierre esta ventana.</div></div>{survey}")


N_SURVEY_INPUTS = len(SUS_ITEMS) + len(TAM_ITEMS) + len(OPEN_QUESTIONS)


def show(screen, state, email="", progress="", summary=""):
    """Arma la tupla de salidas: estado, cabecera, visibilidad de las pantallas y contenidos."""
    survey_reset = [gr.update()] * N_SURVEY_INPUTS
    if screen == 1:
        header = header_html("Triaje manual de PQRs",
                             "Clasificación humana por nivel de urgencia · comparativa de tiempos", 1)
        continue_lbl = gr.update()
    elif screen == 2:
        header = header_html("Triaje en curso",
                             f"Anotador: {state['display']}  ·  clasificación a ciegas en orden de llegada", 2)
        continue_lbl = gr.update()
    elif screen == 3:
        header = header_html("Cuestionario de uso y apropiación",
                             f"Anotador: {state['display']}  ·  su valoración de la plataforma", 3)
        continue_lbl = gr.update()
        # Formulario en blanco para cada anotador
        survey_reset = ([gr.update(value=None)] * (len(SUS_ITEMS) + len(TAM_ITEMS))
                        + [gr.update(value="")] * len(OPEN_QUESTIONS))
    else:
        header = header_html("Sesión finalizada", f"Anotador: {state['display']}", 4)
        remaining = len(LOT) - len(load_done(state["slug"]))
        if remaining > 0:
            continue_lbl = gr.update(visible=True, value=f"↻  Continuar ({remaining} pendientes)")
        elif not questionnaire_done(state["slug"]):
            continue_lbl = gr.update(visible=True, value="Responder cuestionario  →")
        else:
            continue_lbl = gr.update(visible=False)
    return (state, header,
            gr.update(visible=(screen == 1)), gr.update(visible=(screen == 2)),
            gr.update(visible=(screen == 3)), gr.update(visible=(screen == 4)),
            email, progress, summary, continue_lbl, *survey_reset)


def finish(state, already_complete=False):
    """Cierre del lote: cuestionario si aún no lo respondió; si ya lo hizo, resumen."""
    if not questionnaire_done(state["slug"]):
        return show(3, state)
    return show(4, state, summary=summary_html(state, already_complete=already_complete))


def preview_name(raw):
    """Retroalimentación en vivo del identificador: cómo se guardará el archivo."""
    slug = slugify(format_display_name(raw or ""))
    if slug:
        return f"<div class='ok'>✔&nbsp; Sus decisiones se guardarán en <code>{slug}.csv</code></div>"
    return "<div class='hint'>Escriba su nombre o un identificador: con él podrá reanudar la sesión desde cualquier dispositivo.</div>"


def start_session(name, consent):
    """Valida identificador y consentimiento, crea la sesión y muestra la primera PQR."""
    display = format_display_name(name or "")
    slug = slugify(display)
    if not slug:
        raise gr.Error("Escriba su nombre o identificador para continuar.")
    if not consent:
        raise gr.Error("Debe leer y aceptar el consentimiento informado para participar.")
    record_consent(display, slug)
    state = new_session(display, slug)
    if not state["pending"]:
        return finish(state, already_complete=True)
    html, progress = render_email(state)
    return show(2, state, email=html, progress=progress)


def decide(state, choice):
    """Procesa la decisión (a/m/b/s/q) con la misma semántica que la versión de escritorio."""
    if not state or not state["pending"]:
        return show(1, state)
    seconds = time.perf_counter() - state["start"]
    if choice == "q":
        return show(4, state, summary=summary_html(state))
    if choice != "s":
        email = LOT[state["pending"][state["index"]]]
        append_record(state["slug"], {
            "id": email["id"],
            "orden_llegada": email["orden_llegada"],
            "etiqueta_humana": KEYS[choice],
            "segundos": round(seconds, 2),
        })
        state["classified"] += 1
        state["total_seconds"] += seconds
    state["index"] += 1
    if state["index"] >= len(state["pending"]):
        # Lote agotado en esta sesión: si todo quedó clasificado (sin saltos) sigue el cuestionario
        if len(load_done(state["slug"])) >= len(LOT):
            return finish(state)
        return show(4, state, summary=summary_html(state))
    html, progress = render_email(state)
    return show(2, state, email=html, progress=progress)


def submit_questionnaire(state, *answers):
    """Valida que las 19 afirmaciones tengan respuesta, guarda el cuestionario y muestra el resumen."""
    n_sus, n_tam = len(SUS_ITEMS), len(TAM_ITEMS)
    sus_raw, tam_raw, open_raw = answers[:n_sus], answers[n_sus:n_sus + n_tam], answers[n_sus + n_tam:]
    missing = [f"A{n}" for n, a in enumerate(sus_raw, 1) if not a]
    missing += [code for (code, _), a in zip(TAM_ITEMS, tam_raw) if not a]
    if missing:
        # Aviso sin lanzar error: así Gradio no marca cada componente de la pantalla como fallido
        gr.Warning(f"Faltan {len(missing)} afirmaciones por responder: {', '.join(missing)}.")
        return (state, *[gr.update()] * (9 + N_SURVEY_INPUTS))
    record_questionnaire(state, [int(a) for a in sus_raw], [int(a) for a in tam_raw],
                         [(t or "").strip() for t in open_raw])
    return show(4, state, summary=summary_html(state))


def answer_later(state):
    """Pospone el cuestionario: pasa al resumen y queda pendiente para la próxima sesión."""
    return show(4, state, summary=summary_html(state))


def continue_session(state):
    """Recarga los pendientes y reanuda el triaje; si no quedan, sigue con el cuestionario o el resumen."""
    fresh = new_session(state["display"], state["slug"])
    if not fresh["pending"]:
        return finish(fresh, already_complete=True)
    html, progress = render_email(fresh)
    return show(2, fresh, email=html, progress=progress)


def restart():
    """Vuelve a la pantalla de bienvenida para otro anotador."""
    return show(1, None)


# --- Interfaz -----------------------------------------------------------------
THEME = gr.themes.Soft(primary_hue="blue", radius_size=gr.themes.sizes.radius_none)

with gr.Blocks(title="Triaje manual de PQRs", css=CSS, js=SHORTCUTS_JS, head=HEAD, theme=THEME) as demo:
    state = gr.State(None)
    header = gr.HTML(header_html("Triaje manual de PQRs",
                                 "Clasificación humana por nivel de urgencia · comparativa de tiempos", 1),
                     elem_id="hdr")

    # Pantalla 1 · bienvenida: consentimiento (izquierda), identificador y guía (derecha).
    # Dos columnas anchas para que toda la pantalla quepa en el alto de un monitor sin desplazamiento.
    with gr.Column(visible=True, elem_classes=["welcome"]) as welcome:
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=360):
                gr.HTML("<div class='section-label'>Consentimiento informado"
                        "<small>Lea el documento completo y acepte para continuar.</small></div>")
                gr.Markdown(CONSENT_MD, elem_classes=["consent-box"])
                consent = gr.Checkbox(label="He leído y acepto el consentimiento informado y participo de forma libre y voluntaria.",
                                      elem_classes=["consent-check"])
            with gr.Column(scale=1, min_width=360):
                gr.HTML("<div class='section-label'>Identificador del anotador"
                        "<small>Con él se guarda y se reanuda su sesión.</small></div>")
                name = gr.Textbox(show_label=False, placeholder="Nombre o identificador (p. ej. Andrés Rubiano)",
                                  elem_id="name-box")
                name_hint = gr.HTML(preview_name(""))
                gr.HTML(GUIDE_HTML)
        start_btn = gr.Button("Comenzar triaje  →", elem_id="btn-start")

    # Pantalla 2 · triaje (ancho completo)
    with gr.Column(visible=False, elem_classes=["wide"]) as triage:
        progress = gr.HTML()
        email_html = gr.HTML(elem_id="email-box")
        with gr.Row(elem_classes=["decision-row"]):
            alta_btn = gr.Button("Alta", elem_id="btn-alta")
            media_btn = gr.Button("Media", elem_id="btn-media")
            baja_btn = gr.Button("Baja", elem_id="btn-baja")
        with gr.Row(elem_classes=["actions-row"]):
            skip_btn = gr.Button("Saltar esta PQR  (S)", elem_id="btn-skip")
            quit_btn = gr.Button("Guardar y salir  (Q)", elem_id="btn-quit")

    # Pantalla 3 · cuestionario de uso y apropiación (se desplaza por dentro)
    with gr.Column(visible=False, elem_classes=["survey"]) as survey:
        gr.HTML(SURVEY_INTRO_HTML)
        gr.HTML(part_html("A", "Escala de Usabilidad del Sistema (SUS)",
                          "Diez afirmaciones sobre la usabilidad general de la plataforma."))
        sus_inputs = [gr.Radio(choices=LIKERT, label=f"{n}. {text}", elem_classes=["likert"])
                      for n, text in enumerate(SUS_ITEMS, 1)]
        gr.HTML(part_html("B", "Cuestionario de apropiación",
                          "Nueve afirmaciones sobre la utilidad, la facilidad de uso y la intención de uso en el triage de PQRs."))
        tam_inputs = []
        current_dim = None
        for code, text in TAM_ITEMS:
            if code[:2] != current_dim:
                current_dim = code[:2]
                gr.HTML(dimension_html(TAM_DIMENSIONS[current_dim]))
            tam_inputs.append(gr.Radio(choices=LIKERT, label=f"{code}. {text}", elem_classes=["likert"]))
        gr.HTML(part_html("C", "Preguntas abiertas", "Opcionales: escriba con sus palabras lo que considere."))
        open_inputs = [gr.Textbox(label=question, lines=3, elem_classes=["open-q"]) for _, question in OPEN_QUESTIONS]
        with gr.Row(elem_classes=["survey-actions"]):
            later_btn = gr.Button("Responder más tarde", elem_id="btn-later")
            submit_btn = gr.Button("Enviar cuestionario  →", elem_id="btn-submit")

    # Pantalla 4 · resumen (columna central compacta)
    with gr.Column(visible=False, elem_classes=["narrow"]) as summary:
        summary_text = gr.HTML()
        with gr.Row(elem_classes=["actions-row"]):
            continue_btn = gr.Button("↻  Continuar", elem_id="btn-continue")
            exit_btn = gr.Button("Salir", elem_id="btn-exit")

    gr.HTML(FOOTER_HTML, elem_id="ftr")

    survey_inputs = sus_inputs + tam_inputs + open_inputs
    outputs = [state, header, welcome, triage, survey, summary, email_html, progress, summary_text, continue_btn,
               *survey_inputs]
    name.change(preview_name, [name], [name_hint])
    # Tras cada cambio de pantalla, el cuestionario vuelve a mostrarse desde el principio
    scroll_top_js = "() => { const s = document.querySelector('.survey'); if (s) s.scrollTop = 0; }"
    events = [
        start_btn.click(start_session, [name, consent], outputs),
        alta_btn.click(lambda s: decide(s, "a"), [state], outputs),
        media_btn.click(lambda s: decide(s, "m"), [state], outputs),
        baja_btn.click(lambda s: decide(s, "b"), [state], outputs),
        skip_btn.click(lambda s: decide(s, "s"), [state], outputs),
        quit_btn.click(lambda s: decide(s, "q"), [state], outputs),
        submit_btn.click(submit_questionnaire, [state, *survey_inputs], outputs),
        later_btn.click(answer_later, [state], outputs),
        continue_btn.click(continue_session, [state], outputs),
        exit_btn.click(restart, None, outputs),
    ]
    for event in events:
        event.then(None, None, None, js=scroll_top_js)

if __name__ == "__main__":
    # Render asigna el puerto en la variable de entorno PORT; en local se usa 7860.
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
