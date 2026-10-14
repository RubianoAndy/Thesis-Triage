"""
=============================================================================
Plataforma web de triaje manual cronometrado (Gradio · desplegada en Render)
=============================================================================
Versión web de `src/simulation/3-manual_triage.py`: presenta el MISMO lote de
correos que procesó el modelo, A CIEGAS (sin predicción del modelo ni etiqueta
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

Variables de entorno del servicio (Render):
  HF_TOKEN      secreto · token de Hugging Face con permiso de escritura.
  DATASET_REPO  variable · p. ej. "RubianoAndy/thesis-simulation".
  PORT          la asigna Render; en local se usa 7860.

Salida: manual/<anotador>.csv  (columnas: id, orden_llegada, etiqueta_humana, segundos)
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

DATA_DIR    = Path("data")            # carpeta local sincronizada con el dataset
MANUAL_DIR  = DATA_DIR / "manual"
CONSENT_DIR = DATA_DIR / "consent"
MANUAL_DIR.mkdir(parents=True, exist_ok=True)
CONSENT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = {"a": "Alta", "m": "Media", "b": "Baja"}
OUTPUT_FIELDS = ["id", "orden_llegada", "etiqueta_humana", "segundos"]

PROGRAM = "Maestría en Inteligencia Artificial"
FACULTY = "Facultad de Ingeniería · Universidad de La Salle"
AUTHOR  = "Andrés Giovanny Rubiano Muñoz"
TITLE   = ("Sistema de clasificación automática de correos electrónicos por nivel de "
           "urgencia en el sector de la energía usando aprendizaje profundo")


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


# --- Textos ------------------------------------------------------------------
CONSENT_MD = f"""
### Consentimiento informado para anotadores

**Proyecto de grado:** {TITLE}.
**Investigador:** {AUTHOR}. **Directora:** Paola Andrea Zambrano García.

**Propósito.** Esta prueba forma parte de una investigación que compara la clasificación automática de
correos electrónicos, realizada por un modelo de inteligencia artificial, frente a la clasificación hecha
por personas. Su tarea consiste en leer, uno a uno, los correos de un lote y asignar a cada uno un nivel
de urgencia (Alta, Media o Baja). La plataforma presenta los correos en orden de llegada y mide el tiempo
empleado en cada decisión.

**Naturaleza de los correos.** Todos los correos son hipotéticos y fueron generados de forma sintética
para esta investigación; no corresponden a comunicaciones reales de ninguna empresa ni contienen datos
personales de clientes, usuarios o terceros. La participación es individual y a ciegas: usted no conoce la
predicción del modelo ni la etiqueta correcta, y no existen respuestas que lo afecten; el objetivo es
evaluar el sistema, no al participante.

**Datos que se registran.** El identificador que usted proporcione (para guardar y reanudar la sesión),
el nivel de urgencia asignado a cada correo, el tiempo empleado en cada decisión y el tiempo total de la
sesión. No se recolectan datos sensibles.

**Uso de la información.** Exclusivamente académico, dentro de este proyecto de grado y las publicaciones
que de él se deriven. Las decisiones se presentan siempre de forma agregada y bajo seudónimo, con acceso
restringido al investigador y a la directora, y pueden eliminarse a solicitud del participante.

**Voluntariedad.** La participación es libre y voluntaria; puede abstenerse de responder o retirarse en
cualquier momento sin justificación ni consecuencia alguna, y puede pedir que sus decisiones se excluyan
del análisis aun después de finalizar. No hay compensación económica ni riesgos más allá del tiempo
dedicado. Este tratamiento se ajusta a la Ley 1581 de 2012 de Protección de Datos Personales de Colombia.
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


STEPS = ["Consentimiento", "Triaje", "Resumen"]


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
  <div class="guide-sub">Clasifique cada correo según el impacto que describe.</div>
  <div class="guide-row alta"><div class="guide-head"><span class="lvl">Alta</span><kbd>A</kbd></div>
    <div class="guide-desc">Riesgo a personas o infraestructura, corte total, emergencia que requiere atención inmediata.</div></div>
  <div class="guide-row media"><div class="guide-head"><span class="lvl">Media</span><kbd>M</kbd></div>
    <div class="guide-desc">Afectación parcial del servicio o inconformidad operativa que debe atenderse en el corto plazo.</div></div>
  <div class="guide-row baja"><div class="guide-head"><span class="lvl">Baja</span><kbd>B</kbd></div>
    <div class="guide-desc">Consulta, trámite administrativo o solicitud sin afectación del servicio.</div></div>
</div>
<div class="note"><span class="note-icon">⏱</span> El tiempo se cronometra desde que aparece cada correo hasta su decisión.
  Los correos se presentan a ciegas y en orden de llegada; puede cerrar y reanudar después con el mismo identificador.</div>
"""

FOOTER_HTML = ("<div class='footer'><span class='footer-brand'>Universidad de La Salle</span>"
               "<span class='footer-sep'></span>Bogotá D. C., Colombia"
               "<span class='footer-sep'></span>Andrés Giovanny Rubiano Muñoz"
               "<span class='footer-sep'></span>2026 © Todos los derechos reservados</div>")


def progress_html(pos, total):
    """«Correo n de N», porcentaje y barra de progreso dorada."""
    pct = pos / total * 100 if total else 0
    return (f"<div class='progress'><div class='progress-head'><div class='progress-text'>Correo "
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

/* ---- Cabecera institucional ------------------------------------------------ */
.header { background:linear-gradient(135deg, var(--navy-deep) 0%, var(--navy) 60%, var(--navy-soft) 100%);
          color:#F0F4F8; border-bottom:4px solid var(--gold); }
.header-top { display:flex; align-items:center; justify-content:space-between; gap:24px; padding:14px 40px 12px;
              border-bottom:1px solid rgba(255,255,255,.12); }
.header-logo .logo { height:52px; object-fit:contain; display:block; }
.header-meta { text-align:right; }
.header .program { font-family:var(--serif); font-weight:600; font-size:15px; color:#FFFFFF; letter-spacing:.01em; }
.header .faculty { font-size:11.5px; color:#A8BDD3; margin-top:2px; }
.header-main { display:flex; align-items:center; justify-content:space-between; gap:32px; padding:18px 40px 20px; }
.header .eyebrow { font-size:11px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--gold); }
.header .title { font-family:var(--serif); font-size:30px; font-weight:700; color:#FFFFFF; margin-top:4px; line-height:1.15; }
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
.consent-box { height:calc(100vh - 400px) !important; max-height:calc(100vh - 400px) !important; min-height:220px;
               overflow:hidden !important; border:1px solid var(--border); border-top:3px solid var(--navy); background:var(--card);
               padding:0 !important; box-shadow:0 1px 2px rgba(0,46,91,.06); }
.consent-box * { color:var(--ink) !important; font-size:13.5px; line-height:1.6; }
.consent-box h3 { font-family:var(--serif) !important; font-size:17px !important; color:var(--navy) !important;
                  margin:12px 0 6px !important; }
.consent-box strong { color:var(--navy) !important; }
/* Solo el texto se desplaza: el bloque exterior queda fijo para evitar la doble barra */
.consent-box .prose { height:calc(100vh - 403px) !important; max-height:none !important; overflow-y:auto !important;
                      padding:8px 22px 14px !important; box-sizing:border-box; }
.consent-check label { font-size:13.5px !important; color:var(--ink) !important; font-weight:500 !important; }
.consent-check input[type=checkbox] { accent-color:var(--navy); width:18px !important; height:18px !important;
       border:1.5px solid var(--navy) !important; background:#FFFFFF; }
.consent-check input[type=checkbox]:checked { background:var(--navy) !important; border-color:var(--navy) !important; }

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
.email-card { background:var(--card); border:1px solid var(--border); box-shadow:0 2px 8px rgba(0,46,91,.08); overflow-wrap:anywhere; }
.email-head { background:var(--navy); color:#FFFFFF !important; padding:10px 24px; display:flex; justify-content:space-between;
              align-items:center; font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; font-weight:600; }
.email-head, .email-head * { color:#DCE6F2 !important; }
.email-head .tag { color:var(--gold) !important; }
.email-meta { padding:18px 24px 14px; border-bottom:1px solid var(--border); }
.email-card .label { font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--faint) !important; }
.email-card .subject { font-family:var(--serif); font-size:23px; font-weight:700; line-height:1.3; margin:6px 0 0; color:var(--navy) !important; }
.email-card .body { white-space:pre-wrap; font-size:15.5px; line-height:1.7; padding:18px 24px 22px;
                    height:calc(100vh - 600px); min-height:160px; overflow-y:auto; }

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
          font-size:12px; padding:22px 24px 26px; margin-top:18px; border-top:1px solid var(--border); }
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
    done = load_done(slug)
    pending = [i for i, e in enumerate(LOT) if e["id"] not in done]
    return {"display": display, "slug": slug, "pending": pending, "index": 0,
            "start": 0.0, "classified": 0, "total_seconds": 0.0}


def escape(text):
    """Escapa el texto para incrustarlo en HTML."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_email(state):
    """HTML del correo actual (a ciegas) y texto de progreso; arranca el cronómetro."""
    email = LOT[state["pending"][state["index"]]]
    pos, total = state["index"] + 1, len(state["pending"])
    state["start"] = time.perf_counter()   # el cronómetro arranca cuando el correo queda visible
    subject = email["asunto"] or "(sin asunto)"
    body = email["cuerpo"] or "(sin cuerpo)"
    html = (f"<div class='email-card'>"
            f"<div class='email-head'><span>Correo entrante</span>"
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
    stats = [("Correos clasificados", str(state["classified"])),
             ("Progreso acumulado", f"{done} / {total}")]
    if state["classified"]:
        stats.append(("Tiempo total", f"{state['total_seconds']:.1f} s"))
        stats.append(("Promedio por correo", f"{state['total_seconds'] / state['classified']:.2f} s"))
    if records:
        secs = sorted(float(r["segundos"]) for r in records)
        mid = len(secs) // 2
        median = secs[mid] if len(secs) % 2 else (secs[mid - 1] + secs[mid]) / 2
        counts = {k: sum(1 for r in records if r["etiqueta_humana"] == k) for k in ("Alta", "Media", "Baja")}
        stats.append(("Mediana por correo", f"{median:.2f} s"))
        stats.append(("Distribución (Alta / Media / Baja)", f"{counts['Alta']} / {counts['Media']} / {counts['Baja']}"))
    cards = "".join(f"<div class='stat'><div class='value'>{v}</div><div class='label'>{k}</div></div>"
                    for k, v in stats)
    head = (f"<div class='complete'>✓&nbsp; El anotador «{escape(state['display'])}» ya clasificó los {total} correos. ¡Gracias por su participación!</div>"
            if already_complete or done >= total else "")
    return (f"{head}<div class='stats'>{cards}</div>"
            f"<div class='file-card'><div class='label'>Archivo del anotador</div>"
            f"<div class='path'><code>manual/{state['slug']}.csv</code> &nbsp;·&nbsp; se sincroniza con el repositorio de datos "
            f"en menos de un minuto; sus decisiones quedan guardadas aunque cierre esta ventana.</div></div>")


def show(screen, state, email="", progress="", summary=""):
    """Arma la tupla de salidas: estado, cabecera, visibilidad de las pantallas y contenidos."""
    if screen == 1:
        header = header_html("Triaje manual de correos",
                             "Clasificación humana por nivel de urgencia · comparativa de tiempos", 1)
        continue_lbl = gr.update()
    elif screen == 2:
        header = header_html("Triaje en curso",
                             f"Anotador: {state['display']}  ·  clasificación a ciegas en orden de llegada", 2)
        continue_lbl = gr.update()
    else:
        header = header_html("Sesión finalizada", f"Anotador: {state['display']}", 3)
        remaining = len(LOT) - len(load_done(state["slug"]))
        continue_lbl = gr.update(visible=remaining > 0, value=f"↻  Continuar ({remaining} pendientes)")
    return (state, header,
            gr.update(visible=(screen == 1)), gr.update(visible=(screen == 2)), gr.update(visible=(screen == 3)),
            email, progress, summary, continue_lbl)


def preview_name(raw):
    """Retroalimentación en vivo del identificador: cómo se guardará el archivo."""
    slug = slugify(format_display_name(raw or ""))
    if slug:
        return f"<div class='ok'>✔&nbsp; Sus decisiones se guardarán en <code>{slug}.csv</code></div>"
    return "<div class='hint'>Escriba su nombre o un identificador: con él podrá reanudar la sesión desde cualquier dispositivo.</div>"


def start_session(name, consent):
    """Valida identificador y consentimiento, crea la sesión y muestra el primer correo."""
    display = format_display_name(name or "")
    slug = slugify(display)
    if not slug:
        raise gr.Error("Escriba su nombre o identificador para continuar.")
    if not consent:
        raise gr.Error("Debe leer y aceptar el consentimiento informado para participar.")
    record_consent(display, slug)
    state = new_session(display, slug)
    if not state["pending"]:
        return show(3, state, summary=summary_html(state, already_complete=True))
    html, progress = render_email(state)
    return show(2, state, email=html, progress=progress)


def decide(state, choice):
    """Procesa la decisión (a/m/b/s/q) con la misma semántica que la versión de escritorio."""
    if not state or not state["pending"]:
        return show(1, state)
    seconds = time.perf_counter() - state["start"]
    if choice == "q":
        return show(3, state, summary=summary_html(state))
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
        return show(3, state, summary=summary_html(state))
    html, progress = render_email(state)
    return show(2, state, email=html, progress=progress)


def continue_session(state):
    """Recarga los pendientes y reanuda el triaje."""
    fresh = new_session(state["display"], state["slug"])
    if not fresh["pending"]:
        return show(3, fresh, summary=summary_html(fresh, already_complete=True))
    html, progress = render_email(fresh)
    return show(2, fresh, email=html, progress=progress)


def restart():
    """Vuelve a la pantalla de bienvenida para otro anotador."""
    return show(1, None)


# --- Interfaz -----------------------------------------------------------------
THEME = gr.themes.Soft(primary_hue="blue", radius_size=gr.themes.sizes.radius_none)

with gr.Blocks(title="Triaje manual de correos", css=CSS, js=SHORTCUTS_JS, head=HEAD, theme=THEME) as demo:
    state = gr.State(None)
    header = gr.HTML(header_html("Triaje manual de correos",
                                 "Clasificación humana por nivel de urgencia · comparativa de tiempos", 1))

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
        email_html = gr.HTML()
        with gr.Row(elem_classes=["decision-row"]):
            alta_btn = gr.Button("Alta", elem_id="btn-alta")
            media_btn = gr.Button("Media", elem_id="btn-media")
            baja_btn = gr.Button("Baja", elem_id="btn-baja")
        with gr.Row(elem_classes=["actions-row"]):
            skip_btn = gr.Button("Saltar este correo  (S)", elem_id="btn-skip")
            quit_btn = gr.Button("Guardar y salir  (Q)", elem_id="btn-quit")

    # Pantalla 3 · resumen (columna central compacta)
    with gr.Column(visible=False, elem_classes=["narrow"]) as summary:
        summary_text = gr.HTML()
        with gr.Row(elem_classes=["actions-row"]):
            continue_btn = gr.Button("↻  Continuar", elem_id="btn-continue")
            exit_btn = gr.Button("Salir", elem_id="btn-exit")

    gr.HTML(FOOTER_HTML)

    outputs = [state, header, welcome, triage, summary, email_html, progress, summary_text, continue_btn]
    name.change(preview_name, [name], [name_hint])
    start_btn.click(start_session, [name, consent], outputs)
    alta_btn.click(lambda s: decide(s, "a"), [state], outputs)
    media_btn.click(lambda s: decide(s, "m"), [state], outputs)
    baja_btn.click(lambda s: decide(s, "b"), [state], outputs)
    skip_btn.click(lambda s: decide(s, "s"), [state], outputs)
    quit_btn.click(lambda s: decide(s, "q"), [state], outputs)
    continue_btn.click(continue_session, [state], outputs)
    exit_btn.click(restart, None, outputs)

if __name__ == "__main__":
    # Render asigna el puerto en la variable de entorno PORT; en local se usa 7860.
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
