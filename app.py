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

# --- Recursos visuales (misma identidad que la app de escritorio) ------------
def _logo_data_uri():
    """Logo institucional (versión para fondo oscuro) incrustado como data URI."""
    try:
        raw = (Path(__file__).parent / "assets" / "images" / "UnisalleDarkLogo.png").read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    except Exception:
        return ""


LOGO_URI = _logo_data_uri()


def header_html(title, subtitle):
    """Cabecera azul de borde a borde y compacta: logo a la izquierda, bloque académico y título al centro."""
    logo = f'<img class="logo" src="{LOGO_URI}" alt="Universidad de La Salle">' if LOGO_URI else ""
    return f"""
<div class="header">
  <div class="header-inner">
    <div class="header-logo">{logo}</div>
    <div class="header-text">
      <div class="program">{PROGRAM.upper()}</div>
      <div class="faculty">{FACULTY} &nbsp;·&nbsp; {AUTHOR}</div>
      <div class="title">{escape(title)}</div>
      <div class="subtitle">{escape(subtitle)}</div>
    </div>
    <div class="header-logo"></div>
  </div>
</div>"""


GUIDE_HTML = """
<div class="guide">
  <div class="guide-title">CLASIFIQUE CADA CORREO SEGÚN SU NIVEL DE URGENCIA</div>
  <div class="guide-row"><span class="bar alta"></span><div><div class="lvl">ALTA &nbsp;[A]</div>
    Riesgo a personas o infraestructura, corte total, emergencia que requiere atención inmediata.</div></div>
  <div class="guide-row"><span class="bar media"></span><div><div class="lvl">MEDIA &nbsp;[M]</div>
    Afectación parcial del servicio o inconformidad operativa que debe atenderse en el corto plazo.</div></div>
  <div class="guide-row"><span class="bar baja"></span><div><div class="lvl">BAJA &nbsp;[B]</div>
    Consulta, trámite administrativo o solicitud sin afectación del servicio.</div></div>
</div>
<div class="hint">⏱&nbsp; El tiempo se cronometra desde que aparece cada correo hasta su decisión.</div>
"""

FOOTER_HTML = ("<div class='footer'>Universidad de La Salle &nbsp;·&nbsp; Andrés Giovanny Rubiano Muñoz &nbsp;·&nbsp; "
               "2026 © &nbsp;Todos los derechos reservados</div>")


def progress_html(pos, total):
    """«Correo n de N» y barra de progreso dorada, como en la versión de escritorio."""
    pct = pos / total * 100 if total else 0
    return (f"<div class='progress'><div class='progress-text'>Correo {pos} de {total}</div>"
            f"<div class='progress-track'><div class='progress-bar' style='width:{pct:.1f}%'></div></div></div>")


CSS = """
:root { --panel:#012E55; --accent:#FFC20E; --card:#F4F8FC; --ink:#1A2E42; --muted:#5A7A96; --border:#C8D8E8;
        --track:#D8E4EE; --neutral:#EEF2F7; --neutral-text:#5A7A96; --alta:#DC3545; --media:#FFC20E; --baja:#1A8A4A; }
body, .gradio-container, .gradio-container .main { background:#FFFFFF !important; color:var(--ink);
       font-family:"Segoe UI", system-ui, -apple-system, sans-serif; }
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

/* Cabecera de borde a borde, compacta (logo a un lado para ahorrar alto) */
.header { background:var(--panel); color:#F0F4F8; padding:14px 28px 0; }
.header-inner { display:flex; align-items:center; gap:16px; }
.header-logo { flex:0 0 190px; }
.header-logo .logo { height:66px; max-width:100%; object-fit:contain; display:block; }
.header-text { flex:1; text-align:center; }
.header .program { font-weight:700; font-size:15px; color:#FFFFFF; }
.header .faculty { font-size:12px; color:#8AABCA; margin-top:2px; }
.header .title { font-size:24px; font-weight:700; color:var(--accent); margin-top:6px; }
.header .subtitle { font-size:12px; color:#8AABCA; margin-top:2px; padding-bottom:10px; }
.header::after { content:""; display:block; height:3px; background:var(--accent); margin:0 -28px; }
@media (max-width: 760px) { .header-logo { display:none; } }

/* Columna central compacta (bienvenida y resumen) */
.narrow { max-width:640px; margin:0 auto; padding:20px 20px 6px; }
.welcome { max-width:1280px; margin:0 auto; padding:12px 28px 4px; }
.wide { padding:14px 36px 6px; }
.section-label { font-size:12px; font-weight:700; color:var(--panel); letter-spacing:.02em; margin:4px 0 6px; }
.hint { color:var(--muted); font-size:12px; margin:6px 0 8px; }
.ok { color:var(--baja); font-size:12px; margin:6px 0 8px; }

.consent-box { height:calc(100vh - 330px); max-height:calc(100vh - 330px); min-height:200px; overflow-y:auto; border:1px solid var(--border); background:#FFFFFF; padding:2px 16px; }
.consent-box * { color:var(--ink) !important; font-size:13px; }
.consent-box h3 { font-size:14px !important; }

/* Guía de niveles */
.guide, .guide * { color:var(--ink) !important; }
.guide { background:var(--card); border:1px solid var(--border); padding:14px 18px 6px; margin-top:4px; }
.guide-title { font-weight:700; font-size:12px; margin-bottom:12px; }
.guide-row { display:flex; gap:12px; margin:0 0 9px; font-size:12px; color:var(--muted) !important; }
.guide-row .bar { width:6px; flex:none; }
.guide-row .lvl { font-weight:700; font-size:14px; color:var(--ink) !important; margin-bottom:2px; }
.bar.alta { background:var(--alta); } .bar.media { background:var(--media); } .bar.baja { background:var(--baja); }

/* Triaje */
.progress { margin:2px 0 10px; }
.progress-text { font-weight:700; font-size:15px; color:var(--panel); }
.progress-track { height:6px; background:var(--track); margin-top:8px; }
.progress-bar { height:100%; background:var(--accent); }
.email-card, .email-card * { color:var(--ink) !important; }
.email-card { background:var(--card); border:1px solid var(--border); padding:18px 22px; overflow-wrap:anywhere; }
.email-card .label { font-size:11px; font-weight:700; color:var(--muted) !important; }
.email-card .subject { font-size:20px; font-weight:700; margin:6px 0 14px; }
.email-card .body { white-space:pre-wrap; font-size:15px; line-height:1.55; border-top:1px solid var(--border);
                    padding-top:14px; height:calc(100vh - 460px); min-height:160px; overflow-y:auto; }

/* Resumen */
.stats { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:8px 0 18px; }
.stat { border:1px solid var(--border); border-top:4px solid var(--accent); background:var(--card); padding:12px 18px; }
.stat .value { font-size:26px; font-weight:700; color:var(--panel) !important; }
.stat .label { font-size:11px; font-weight:700; color:var(--muted) !important; margin-top:4px; }
.file-card { border:1px solid var(--border); background:var(--card); padding:14px 18px; }
.file-card .label { font-size:11px; font-weight:700; color:var(--muted) !important; }
.file-card .path { font-size:12px; color:var(--ink) !important; margin-top:6px; }
.complete { color:var(--baja); font-weight:700; font-size:15px; margin:4px 0 14px; }
.footer { text-align:center; color:var(--muted); font-size:12px; padding:10px 0 10px; }

/* Botones (planos, como en la app de escritorio) */
#btn-start { background:var(--accent) !important; color:var(--panel) !important; font-weight:700; font-size:15px;
             border:none !important; min-height:44px; max-width:240px; margin:6px auto 0; display:block; }
#btn-alta, #btn-media, #btn-baja { min-height:52px; font-size:15px; font-weight:700; border:none !important; }
#btn-alta  { background:var(--alta) !important; color:#FFFFFF !important; }
#btn-media { background:var(--media) !important; color:var(--panel) !important; }
#btn-baja  { background:var(--baja) !important; color:#FFFFFF !important; }
#btn-skip  { background:var(--neutral) !important; color:var(--neutral-text) !important; font-weight:700; border:none !important;
             max-width:170px; }
#btn-quit, #btn-exit { background:var(--panel) !important; color:#F0F4F8 !important; font-weight:700; border:none !important; }
#btn-quit { max-width:220px; margin-left:auto; }
#btn-exit { max-width:120px; margin-left:auto; }
#btn-continue { background:var(--accent) !important; color:var(--panel) !important; font-weight:700; border:none !important;
                max-width:260px; }
.actions-row { justify-content:space-between; margin-top:16px !important; }
.consent-box .prose, .consent-box > div { max-height:none; overflow:visible; }
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
    html = (f"<div class='email-card'><div class='label'>ASUNTO</div>"
            f"<div class='subject'>{escape(subject)}</div>"
            f"<div class='label'>MENSAJE</div><div class='body'>{escape(body)}</div></div>")
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
    cards = "".join(f"<div class='stat'><div class='value'>{v}</div><div class='label'>{k.upper()}</div></div>"
                    for k, v in stats)
    head = (f"<div class='complete'>✓&nbsp; El anotador «{escape(state['display'])}» ya clasificó los {total} correos.</div>"
            if already_complete or done >= total else "")
    return (f"{head}<div class='stats'>{cards}</div>"
            f"<div class='file-card'><div class='label'>ARCHIVO DEL ANOTADOR</div>"
            f"<div class='path'>manual/{state['slug']}.csv &nbsp;·&nbsp; se sincroniza con el repositorio de datos "
            f"en menos de un minuto</div></div>")


def show(screen, state, email="", progress="", summary=""):
    """Arma la tupla de salidas: estado, cabecera, visibilidad de las pantallas y contenidos."""
    if screen == 1:
        header = header_html("Triaje manual de correos",
                             "Clasificación humana por nivel de urgencia · comparativa de tiempos")
        continue_lbl = gr.update()
    elif screen == 2:
        header = header_html("Triaje en curso",
                             f"Anotador: {state['display']}  ·  clasificación a ciegas en orden de llegada")
        continue_lbl = gr.update()
    else:
        header = header_html("Sesión finalizada", f"Anotador: {state['display']}")
        remaining = len(LOT) - len(load_done(state["slug"]))
        continue_lbl = gr.update(visible=remaining > 0, value=f"↻  Continuar ({remaining} pendientes)")
    return (state, header,
            gr.update(visible=(screen == 1)), gr.update(visible=(screen == 2)), gr.update(visible=(screen == 3)),
            email, progress, summary, continue_lbl)


def preview_name(raw):
    """Retroalimentación en vivo del identificador: cómo se guardará el archivo."""
    slug = slugify(format_display_name(raw or ""))
    if slug:
        return f"<div class='ok'>✔&nbsp; Se guardará como: &nbsp;{slug}.csv</div>"
    return "<div class='hint'>Escribe el identificador del anotador para habilitar el triaje.</div>"


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

with gr.Blocks(title="Triaje manual de correos", css=CSS, js=SHORTCUTS_JS, theme=THEME) as demo:
    state = gr.State(None)
    header = gr.HTML(header_html("Triaje manual de correos",
                                 "Clasificación humana por nivel de urgencia · comparativa de tiempos"))

    # Pantalla 1 · bienvenida: consentimiento (izquierda), identificador y guía (derecha).
    # Dos columnas anchas para que toda la pantalla quepa en el alto de un monitor sin desplazamiento.
    with gr.Column(visible=True, elem_classes=["welcome"]) as welcome:
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=360):
                gr.HTML("<div class='section-label'>CONSENTIMIENTO INFORMADO</div>")
                gr.Markdown(CONSENT_MD, elem_classes=["consent-box"])
                consent = gr.Checkbox(label="He leído y acepto el consentimiento informado y participo de forma libre y voluntaria.")
            with gr.Column(scale=1, min_width=360):
                gr.HTML("<div class='section-label'>IDENTIFICADOR DEL ANOTADOR</div>")
                name = gr.Textbox(show_label=False, placeholder="Nombre o identificador (p. ej. Andrés Rubiano)")
                name_hint = gr.HTML(preview_name(""))
                gr.HTML(GUIDE_HTML)
        start_btn = gr.Button("Comenzar triaje  →", elem_id="btn-start")

    # Pantalla 2 · triaje (ancho completo)
    with gr.Column(visible=False, elem_classes=["wide"]) as triage:
        progress = gr.HTML()
        email_html = gr.HTML()
        with gr.Row():
            alta_btn = gr.Button("🔴  Alta  (A)", elem_id="btn-alta")
            media_btn = gr.Button("🟡  Media  (M)", elem_id="btn-media")
            baja_btn = gr.Button("🟢  Baja  (B)", elem_id="btn-baja")
        with gr.Row(elem_classes=["actions-row"]):
            skip_btn = gr.Button("⏭  Saltar  (S)", elem_id="btn-skip")
            quit_btn = gr.Button("💾  Guardar y salir  (Q)", elem_id="btn-quit")

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
