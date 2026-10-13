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
    if not HF_TOKEN:
        raise RuntimeError("Falta la variable de entorno HF_TOKEN en la configuración del servicio.")
    path = hf_hub_download(DATASET_REPO, "simulation.csv", repo_type="dataset", token=HF_TOKEN)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, 1):
        r["orden_llegada"] = i
    return rows


LOT = load_lot()

# --- Sincronización con el repositorio de datos -------------------------------
scheduler = CommitScheduler(
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
    if local.exists():
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

# --- Recursos visuales --------------------------------------------------------
def _logo_data_uri():
    """Logo institucional (versión para fondo oscuro) incrustado como data URI."""
    try:
        raw = (Path(__file__).parent / "assets" / "images" / "UnisalleDarkLogo.png").read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    except Exception:
        return ""


LOGO_URI = _logo_data_uri()

GUIDE_HTML = """
<div class="guide">
  <div class="guide-title">CLASIFIQUE CADA CORREO SEGÚN SU NIVEL DE URGENCIA</div>
  <div class="guide-row"><span class="bar alta"></span><div><b>ALTA · tecla A</b><br>
    Riesgo a personas o infraestructura, corte total, emergencia que requiere atención inmediata.</div></div>
  <div class="guide-row"><span class="bar media"></span><div><b>MEDIA · tecla M</b><br>
    Afectación parcial del servicio o inconformidad operativa que debe atenderse en el corto plazo.</div></div>
  <div class="guide-row"><span class="bar baja"></span><div><b>BAJA · tecla B</b><br>
    Consulta, trámite administrativo o solicitud sin afectación del servicio.</div></div>
  <div class="guide-note">⏱ El tiempo se cronometra desde que aparece cada correo hasta su decisión.
    Lea cada correo completo antes de decidir; no hay respuestas correctas o incorrectas para usted.</div>
</div>
"""

LEGEND_HTML = """
<div class="legend">
  <div><span class="bar alta"></span><b>Alta:</b> riesgo a personas o infraestructura, corte total, emergencia.</div>
  <div><span class="bar media"></span><b>Media:</b> afectación parcial o inconformidad operativa a corto plazo.</div>
  <div><span class="bar baja"></span><b>Baja:</b> consulta, trámite o solicitud sin afectación del servicio.</div>
</div>
"""

HEADER_HTML = f"""
<div class="header">
  {'<img class="logo" src="' + LOGO_URI + '" alt="Universidad de La Salle">' if LOGO_URI else ''}
  <div class="program">{PROGRAM.upper()}</div>
  <div class="faculty">{FACULTY}</div>
  <div class="author">{AUTHOR}</div>
  <div class="title">Triaje manual de correos</div>
  <div class="subtitle">Clasificación humana por nivel de urgencia · comparativa de tiempos</div>
</div>
"""

FOOTER_HTML = ("<div class='footer'>Universidad de La Salle · Andrés Giovanny Rubiano Muñoz · 2026 © "
               "Todos los derechos reservados</div>")

STEPS = ["Consentimiento e identificación", "Triaje de correos", "Resumen de la sesión"]


def steps_html(current):
    """Indicador de pasos (1 a 3) con el paso actual resaltado."""
    items = []
    for i, label in enumerate(STEPS, 1):
        cls = "done" if i < current else ("current" if i == current else "")
        items.append(f"<div class='step {cls}'><span class='num'>{i}</span><span>{label}</span></div>")
    return "<div class='steps'>" + "".join(items) + "</div>"


def progress_html(pos, total):
    """Texto y barra de progreso del triaje."""
    pct = int(pos / total * 100) if total else 0
    return (f"<div class='progress'><div class='progress-text'><span>Correo <b>{pos}</b> de <b>{total}</b> "
            f"pendientes</span><span class='progress-pct'>{pct} %</span></div>"
            f"<div class='progress-track'><div class='progress-bar' style='width:{pct}%'></div></div></div>")


CSS = """
:root { --panel:#012E55; --accent:#FFC20E; --card:#F4F8FC; --ink:#1A2E42; --muted:#5A7A96; --border:#C8D8E8; }
/* Fondo blanco, ancho completo sin desbordes y esquinas cuadradas en toda la interfaz */
body, .gradio-container, .gradio-container .main { background:#FFFFFF !important; color:var(--ink); }
.gradio-container { max-width:100% !important; width:100% !important; margin:0 !important;
                    padding:0 24px 24px !important; overflow-x:hidden; box-sizing:border-box; }
.gradio-container *, .gradio-container *::before, .gradio-container *::after { border-radius:0 !important; }
footer { display:none !important; }

.header { background:var(--panel); color:#F0F4F8; padding:20px 24px 0; text-align:center; }
.header .logo { display:block; margin:0 auto 10px; height:96px; max-width:100%; object-fit:contain; }
.header .program { font-weight:700; font-size:13px; letter-spacing:.06em; color:#F0F4F8; }
.header .faculty, .header .author { font-size:12px; color:#8AABCA; }
.header .title { font-size:26px; font-weight:800; margin-top:12px; color:#FFFFFF; }
.header .subtitle { font-size:13px; color:#8AABCA; padding-bottom:14px; border-bottom:4px solid var(--accent); }

.steps { display:flex; gap:0; margin:16px 0 8px; border:1px solid var(--border); }
.step { flex:1; display:flex; align-items:center; gap:10px; padding:10px 14px; font-size:13px; color:var(--muted);
        background:#FFFFFF; border-right:1px solid var(--border); }
.step:last-child { border-right:none; }
.step .num { display:inline-flex; width:24px; height:24px; align-items:center; justify-content:center;
             border:2px solid var(--border); font-weight:700; font-size:12px; flex:none; }
.step.current { background:var(--card); color:var(--panel); font-weight:700; }
.step.current .num { background:var(--panel); border-color:var(--panel); color:var(--accent); }
.step.done { color:var(--panel); }
.step.done .num { background:var(--accent); border-color:var(--accent); color:var(--panel); }

.consent-box { max-height:300px; overflow-y:auto; border:1px solid var(--border); padding:4px 18px; background:#FFFFFF; }
.consent-box * { color:var(--ink) !important; }
.form-card { border:1px solid var(--border); background:var(--card); padding:12px 16px; }
.section-title { font-size:12px; font-weight:700; letter-spacing:.06em; color:var(--muted); margin:14px 0 6px; }

.guide, .guide *, .email-card, .email-card *, .legend, .legend *, .summary-card, .summary-card * { color:var(--ink) !important; }
.guide { background:#FFFFFF; border:1px solid var(--border); padding:16px 18px; }
.guide-title { font-weight:700; font-size:12px; letter-spacing:.05em; margin-bottom:10px; }
.guide-row { display:flex; gap:12px; margin:8px 0; font-size:14px; }
.guide-row .bar, .legend .bar { width:6px; flex:none; display:inline-block; }
.legend .bar { width:10px; height:10px; margin-right:8px; vertical-align:middle; }
.bar.alta { background:#DC3545; } .bar.media { background:#FFC20E; } .bar.baja { background:#1A8A4A; }
.guide-note { color:var(--muted) !important; font-size:12px; margin-top:12px; }
.legend { display:flex; flex-wrap:wrap; gap:6px 24px; font-size:13px; margin:6px 0 4px; }

.progress { margin:14px 0 10px; }
.progress-text { display:flex; justify-content:space-between; font-size:15px; color:var(--ink); }
.progress-pct { color:var(--muted); font-weight:700; }
.progress-track { height:8px; background:#D8E4EE; margin-top:6px; }
.progress-bar { height:100%; background:var(--accent); }

.email-card { background:var(--card); border:1px solid var(--border); padding:18px 20px; overflow-wrap:anywhere; }
.email-card .label { font-size:11px; font-weight:700; letter-spacing:.06em; color:var(--muted) !important; }
.email-card .subject { font-size:20px; font-weight:700; margin:4px 0 14px; }
.email-card .body { white-space:pre-wrap; font-size:16px; line-height:1.55; border-top:1px solid var(--border); padding-top:12px; }

.summary-card { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin:12px 0; }
.stat { border:1px solid var(--border); border-top:4px solid var(--accent); background:var(--card); padding:14px 16px; }
.stat .value { font-size:28px; font-weight:800; color:var(--panel) !important; }
.stat .label { font-size:11px; font-weight:700; letter-spacing:.06em; color:var(--muted) !important; }
.summary-note { border:1px solid var(--border); padding:12px 16px; margin-top:8px; background:#FFFFFF; color:var(--ink); }

.footer { text-align:center; color:var(--muted); font-size:12px; margin-top:18px; }
#btn-alta, #btn-media, #btn-baja { min-height:56px; font-size:16px; font-weight:700; border:none !important; }
#btn-alta  { background:#DC3545 !important; color:#FFFFFF !important; }
#btn-media { background:#FFC20E !important; color:#012E55 !important; }
#btn-baja  { background:#1A8A4A !important; color:#FFFFFF !important; }
#btn-start { background:var(--accent) !important; color:var(--panel) !important; font-weight:700; min-height:50px; font-size:16px; border:none !important; }
#btn-continue { background:var(--panel) !important; color:#FFFFFF !important; font-weight:700; border:none !important; }
"""

# Al cargar: fuerza el tema claro (la interfaz es blanca aunque el navegador esté en modo oscuro)
# y activa los atajos de teclado A/M/B/S, que pulsan los botones (solo si no se está escribiendo).
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
    const map = {a:'btn-alta', m:'btn-media', b:'btn-baja', s:'btn-saltar'};
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
    html = (f"<div class='email-card'><div class='label'>ASUNTO</div>"
            f"<div class='subject'>{escape(subject)}</div>"
            f"<div class='label'>MENSAJE</div><div class='body'>{escape(body)}</div></div>")
    return html, progress_html(pos, total)


def summary_html(state):
    """Resumen de la sesión (equivale a la pantalla final de la versión de escritorio)."""
    done = len(load_done(state["slug"]))
    total = len(LOT)
    stats = [("Clasificados en esta sesión", str(state["classified"])),
             ("Progreso acumulado", f"{done} / {total}")]
    if state["classified"]:
        stats.append(("Tiempo total de la sesión", f"{state['total_seconds']:.1f} s"))
        stats.append(("Promedio por correo", f"{state['total_seconds'] / state['classified']:.1f} s"))
    cards = "".join(f"<div class='stat'><div class='value'>{v}</div><div class='label'>{k.upper()}</div></div>"
                    for k, v in stats)
    if done >= total:
        note = "✅ <b>Triaje completo. Muchas gracias por su participación.</b>"
    else:
        note = (f"Quedan <b>{total - done}</b> correos. Puede continuar ahora o volver más tarde con el mismo "
                f"identificador: la sesión se reanuda donde quedó.")
    return (f"<div class='summary-card'>{cards}</div>"
            f"<div class='summary-note'>Anotador: <b>{escape(state['display'])}</b> · archivo "
            f"<code>manual/{state['slug']}.csv</code>, sincronizado automáticamente con el repositorio de datos "
            f"en menos de un minuto.<br><br>{note}</div>")


def show(screen, state, email="", progress="", summary=""):
    """Arma la tupla de salidas: estado, visibilidad de las tres pantallas, indicador de pasos y contenidos."""
    return (state,
            gr.update(visible=(screen == 1)), gr.update(visible=(screen == 2)), gr.update(visible=(screen == 3)),
            steps_html(screen), email, progress, summary)


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
        return show(3, state, summary=summary_html(state))
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
        return show(3, fresh, summary=summary_html(fresh))
    html, progress = render_email(fresh)
    return show(2, fresh, email=html, progress=progress)


def restart():
    """Vuelve a la pantalla de bienvenida para otro anotador."""
    return show(1, None)


# --- Interfaz -----------------------------------------------------------------
THEME = gr.themes.Soft(primary_hue="blue", radius_size=gr.themes.sizes.radius_none)

with gr.Blocks(title="Triaje manual de correos", css=CSS, js=SHORTCUTS_JS, theme=THEME) as demo:
    state = gr.State(None)
    gr.HTML(HEADER_HTML)
    steps = gr.HTML(steps_html(1))

    # Pantalla 1 · consentimiento, identificador y guía
    with gr.Column(visible=True) as welcome:
        gr.HTML("<div class='section-title'>PASO 1 · LEA EL CONSENTIMIENTO INFORMADO</div>")
        gr.Markdown(CONSENT_MD, elem_classes=["consent-box"])
        with gr.Column(elem_classes=["form-card"]):
            consent = gr.Checkbox(label="He leído y acepto el consentimiento informado y participo de forma libre y voluntaria.")
            name = gr.Textbox(label="Nombre o identificador del anotador",
                              placeholder="p. ej. Andrés Rubiano (úselo igual si vuelve a entrar: la sesión se reanuda)")
        gr.HTML("<div class='section-title'>PASO 2 · CÓMO CLASIFICAR</div>")
        gr.HTML(GUIDE_HTML)
        start_btn = gr.Button("Comenzar triaje →", elem_id="btn-start")

    # Pantalla 2 · triaje
    with gr.Column(visible=False) as triage:
        progress = gr.HTML()
        email_html = gr.HTML()
        gr.HTML(LEGEND_HTML)
        with gr.Row():
            alta_btn = gr.Button("Alta  (A)", elem_id="btn-alta")
            media_btn = gr.Button("Media  (M)", elem_id="btn-media")
            baja_btn = gr.Button("Baja  (B)", elem_id="btn-baja")
        with gr.Row():
            skip_btn = gr.Button("Saltar este correo  (S)", elem_id="btn-saltar", variant="secondary")
            quit_btn = gr.Button("Guardar y salir", variant="secondary")

    # Pantalla 3 · resumen
    with gr.Column(visible=False) as summary:
        summary_text = gr.HTML()
        with gr.Row():
            continue_btn = gr.Button("↻ Continuar el triaje", elem_id="btn-continue")
            restart_btn = gr.Button("Otro anotador", variant="secondary")

    gr.HTML(FOOTER_HTML)

    outputs = [state, welcome, triage, summary, steps, email_html, progress, summary_text]
    start_btn.click(start_session, [name, consent], outputs)
    alta_btn.click(lambda s: decide(s, "a"), [state], outputs)
    media_btn.click(lambda s: decide(s, "m"), [state], outputs)
    baja_btn.click(lambda s: decide(s, "b"), [state], outputs)
    skip_btn.click(lambda s: decide(s, "s"), [state], outputs)
    quit_btn.click(lambda s: decide(s, "q"), [state], outputs)
    continue_btn.click(continue_session, [state], outputs)
    restart_btn.click(restart, None, outputs)

if __name__ == "__main__":
    # Render asigna el puerto en la variable de entorno PORT; en local se usa 7860.
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
