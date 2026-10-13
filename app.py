"""
=============================================================================
Plataforma web de triaje manual cronometrado (Hugging Face Space · Gradio)
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

Variables de entorno del Space:
  HF_TOKEN      secreto · token de Hugging Face con permiso de escritura.
  DATASET_REPO  variable · p. ej. "RubianoAndy/thesis-simulation".

Salida: manual/<anotador>.csv  (columnas: id, orden_llegada, etiqueta_humana, segundos)
=============================================================================
"""
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
        raise RuntimeError("Falta el secreto HF_TOKEN en la configuración del Space.")
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

GUIDE_HTML = """
<div class="guide">
  <div class="guide-title">CLASIFIQUE CADA CORREO SEGÚN SU NIVEL DE URGENCIA</div>
  <div class="guide-row"><span class="bar alta"></span><div><b>ALTA [A]</b><br>
    Riesgo a personas o infraestructura, corte total, emergencia que requiere atención inmediata.</div></div>
  <div class="guide-row"><span class="bar media"></span><div><b>MEDIA [M]</b><br>
    Afectación parcial del servicio o inconformidad operativa que debe atenderse en el corto plazo.</div></div>
  <div class="guide-row"><span class="bar baja"></span><div><b>BAJA [B]</b><br>
    Consulta, trámite administrativo o solicitud sin afectación del servicio.</div></div>
  <div class="guide-note">⏱ El tiempo se cronometra desde que aparece cada correo hasta su decisión.
    Puede usar las teclas <b>A</b>, <b>M</b>, <b>B</b> y <b>S</b> (saltar).</div>
</div>
"""

HEADER_HTML = f"""
<div class="header">
  <div class="program">{PROGRAM.upper()}</div>
  <div class="faculty">{FACULTY}</div>
  <div class="author">{AUTHOR}</div>
  <div class="title">Triaje manual de correos</div>
  <div class="subtitle">Clasificación humana por nivel de urgencia · comparativa de tiempos</div>
</div>
"""

FOOTER_HTML = ("<div class='footer'>Universidad de La Salle · Andrés Giovanny Rubiano Muñoz · 2026 © "
               "Todos los derechos reservados</div>")

CSS = """
:root { --panel:#012E55; --accent:#FFC20E; --card:#F4F8FC; --ink:#1A2E42; --muted:#5A7A96; --border:#C8D8E8; }
.header { background: var(--panel); color:#F0F4F8; padding:22px 24px 0; border-radius:12px 12px 0 0; text-align:center; }
.header .program { font-weight:700; font-size:13px; letter-spacing:.06em; }
.header .faculty, .header .author { font-size:12px; color:#8AABCA; }
.header .title { font-size:26px; font-weight:800; margin-top:14px; }
.header .subtitle { font-size:13px; color:#8AABCA; padding-bottom:14px; border-bottom:4px solid var(--accent); }
/* Las tarjetas son claras en ambos temas: se fija el color del texto para que el tema oscuro de Gradio no lo aclare. */
.guide, .guide *, .email-card, .email-card * { color: var(--ink) !important; }
.guide { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px 18px; }
.guide-title { font-weight:700; font-size:12px; letter-spacing:.05em; margin-bottom:10px; }
.guide-row { display:flex; gap:12px; margin:8px 0; font-size:14px; }
.guide-row .bar { width:6px; border-radius:3px; flex:none; }
.bar.alta { background:#DC3545; } .bar.media { background:#FFC20E; } .bar.baja { background:#1A8A4A; }
.guide-note { color:var(--muted) !important; font-size:12px; margin-top:12px; }
.email-card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:18px 20px; }
.email-card .label { font-size:11px; font-weight:700; letter-spacing:.06em; color:var(--muted) !important; }
.email-card .subject { font-size:19px; font-weight:700; margin:4px 0 14px; }
.email-card .body { white-space:pre-wrap; font-size:15px; line-height:1.5; border-top:1px solid var(--border); padding-top:12px; }
.progress { font-weight:700; font-size:15px; }   /* hereda el color del tema (claro u oscuro) */
.footer { text-align:center; color:var(--muted); font-size:12px; margin-top:18px; }
#btn-alta  { background:#DC3545 !important; color:#fff !important; font-weight:700; }
#btn-media { background:#FFC20E !important; color:#012E55 !important; font-weight:700; }
#btn-baja  { background:#1A8A4A !important; color:#fff !important; font-weight:700; }
"""

# Atajos de teclado: A/M/B/S pulsan los botones correspondientes (solo si no se está escribiendo).
SHORTCUTS_JS = """
() => {
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
    return html, f"<div class='progress'>Correo {pos} de {total} pendientes</div>"


def summary_md(state):
    """Resumen de la sesión (equivale a la pantalla final de la versión de escritorio)."""
    done = len(load_done(state["slug"]))
    lines = [f"### Sesión de **{state['display']}**",
             f"- Clasificados en esta sesión: **{state['classified']}**",
             f"- Progreso acumulado: **{done} / {len(LOT)}**"]
    if state["classified"]:
        lines.append(f"- Tiempo total de la sesión: **{state['total_seconds']:.1f} s** "
                     f"(promedio **{state['total_seconds'] / state['classified']:.1f} s** por correo)")
    lines.append(f"- Archivo: `manual/{state['slug']}.csv` (se sincroniza automáticamente con el repositorio "
                 f"de datos en menos de un minuto).")
    if done >= len(LOT):
        lines.append("\n✅ **Triaje completo. Muchas gracias por su participación.**")
    else:
        lines.append(f"\nQuedan **{len(LOT) - done}** correos. Puede continuar ahora o volver más tarde "
                     f"con el mismo identificador.")
    return "\n".join(lines)


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
        return (state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                "", "", summary_md(state))
    html, progress = render_email(state)
    return (state, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
            html, progress, "")


def decide(state, choice):
    """Procesa la decisión (a/m/b/s/q) con la misma semántica que la versión de escritorio."""
    if not state or not state["pending"]:
        return state, gr.update(), gr.update(), gr.update(), "", "", ""
    seconds = time.perf_counter() - state["start"]
    if choice == "q":
        return (state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                "", "", summary_md(state))
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
        return (state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                "", "", summary_md(state))
    html, progress = render_email(state)
    return state, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), html, progress, ""


def continue_session(state):
    """Recarga los pendientes y reanuda el triaje."""
    fresh = new_session(state["display"], state["slug"])
    if not fresh["pending"]:
        return (fresh, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                "", "", summary_md(fresh))
    html, progress = render_email(fresh)
    return fresh, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), html, progress, ""


def restart():
    """Vuelve a la pantalla de bienvenida para otro anotador."""
    return (None, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            "", "", "")


# --- Interfaz -----------------------------------------------------------------
with gr.Blocks(title="Triaje manual de correos", css=CSS, js=SHORTCUTS_JS,
               theme=gr.themes.Soft(primary_hue="blue")) as demo:
    state = gr.State(None)
    gr.HTML(HEADER_HTML)

    # Pantalla 1 · bienvenida, consentimiento e identificador
    with gr.Column(visible=True) as welcome:
        gr.Markdown(CONSENT_MD)
        consent = gr.Checkbox(label="He leído y acepto el consentimiento informado y participo de forma libre y voluntaria.")
        name = gr.Textbox(label="Nombre o identificador del anotador", placeholder="p. ej. Andrés Rubiano")
        gr.HTML(GUIDE_HTML)
        start_btn = gr.Button("Comenzar triaje →", variant="primary")

    # Pantalla 2 · triaje
    with gr.Column(visible=False) as triage:
        progress = gr.HTML()
        email_html = gr.HTML()
        with gr.Row():
            alta_btn = gr.Button("🔴 Alta (A)", elem_id="btn-alta")
            media_btn = gr.Button("🟡 Media (M)", elem_id="btn-media")
            baja_btn = gr.Button("🟢 Baja (B)", elem_id="btn-baja")
        with gr.Row():
            skip_btn = gr.Button("⏭ Saltar (S)", elem_id="btn-saltar", variant="secondary")
            quit_btn = gr.Button("💾 Guardar y salir", variant="secondary")

    # Pantalla 3 · resumen
    with gr.Column(visible=False) as summary:
        summary_text = gr.Markdown()
        with gr.Row():
            continue_btn = gr.Button("↻ Continuar", variant="primary")
            restart_btn = gr.Button("Otro anotador", variant="secondary")

    gr.HTML(FOOTER_HTML)

    outputs = [state, welcome, triage, summary, email_html, progress, summary_text]
    start_btn.click(start_session, [name, consent], outputs)
    alta_btn.click(lambda s: decide(s, "a"), [state], outputs)
    media_btn.click(lambda s: decide(s, "m"), [state], outputs)
    baja_btn.click(lambda s: decide(s, "b"), [state], outputs)
    skip_btn.click(lambda s: decide(s, "s"), [state], outputs)
    quit_btn.click(lambda s: decide(s, "q"), [state], outputs)
    continue_btn.click(continue_session, [state], outputs)
    restart_btn.click(restart, None, outputs)

if __name__ == "__main__":
    demo.queue().launch()
