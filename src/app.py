import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ───────────────────────── Configuración de rutas ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PARQUET_2019 = os.path.join(DATA_DIR, "ecp_2019.parquet")
PARQUET_2023 = os.path.join(DATA_DIR, "ecp_2023.parquet")

st.set_page_config(
    page_title="ECP Colombia 2019 vs 2023",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────── Utilidades ───────────────────────────────────
@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

@st.cache_data
def porcentaje_ponderado(df: pd.DataFrame, var: str, valor_si: int = 1) -> float:
    """Devuelve el % ponderado de registros donde var == valor_si (0 si no existe)."""
    if var not in df.columns:
        return 0.0
    yes_weight = df.loc[df[var] == valor_si, "WEIGHT"].sum()
    total_weight = df["WEIGHT"].sum()
    return 100 * yes_weight / total_weight if total_weight else 0.0

@st.cache_data
def weighted_mean(df: pd.DataFrame, var: str) -> float:
    """
    Devuelve la media ponderada de df[var] usando df['WEIGHT'],
    considerando sólo valores entre 1 y 5.
    """
    if var not in df.columns:
        return 0.0
    # Mascara de valores válidos 1–5
    mask = df[var].between(1, 5)
    # Suma de pesos sólo para los válidos
    w = df.loc[mask, "WEIGHT"].sum()
    if w == 0:
        return 0.0
    # Media ponderada restringida
    return (df.loc[mask, var] * df.loc[mask, "WEIGHT"]).sum() / w


def plot_importance_bar(df: pd.DataFrame, var_map: dict[str,str], y_range: tuple[float,float]=(1, 5)):
    """
    Dibuja un bar chart de la media ponderada de cada var en var_map.
    • var_map: {codigo:str -> etiqueta:str}
    """
    data = {"Elección": [], "Importancia": []}
    for var, label in var_map.items():
        if var in df.columns:
            data["Elección"].append(label)
            data["Importancia"].append(weighted_mean(df, var))
    imp_df = pd.DataFrame(data)
    fig = px.bar(
        imp_df,
        x="Elección",
        y="Importancia",
        text=imp_df["Importancia"].apply(lambda x: f"{x:.2f}"),
        labels={"Importancia": "Importancia media (1–5)"},
        template="plotly_white",
        height=400,
    )
    fig.update_traces(marker_line_color="black", marker_line_width=1)
    fig.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        margin=dict(t=40, b=40),
    )
    fig.update_yaxes(range=y_range, title="Escala 1–5")
    return fig

def engineer_ideology(df: pd.DataFrame) -> pd.DataFrame:
    """Crea solo la variable ideología_group derivada de P5328."""
    df = df.copy()

    # Agrupar auto-ubicación 1–10 en tres bloques
    if "P5328" in df.columns:
        df["ideology_group"] = pd.cut(
            df["P5328"],
            bins=[0, 3.5, 6.5, 10],
            labels=["Izquierda", "Centro", "Derecha"],
        )

        # Guardar también la media individual (por si se necesita)
        mask = df["P5328"].between(1, 10)
        w    = df.loc[mask, "WEIGHT"]
        df["ideology_score"] = df["P5328"].where(mask)

    return df

# ─────────────────────────── Carga de datos ────────────────────────────────
df2019 = load_data(PARQUET_2019)
df2023 = load_data(PARQUET_2023)

df2019_ideo = engineer_ideology(df2019)
df2023_ideo = engineer_ideology(df2023)

# ───────────────────────── Navegación lateral ─────────────────────────
# Estilos para la sidebar: fondo suave, botones full-width con hover
st.sidebar.markdown(
    """
    <style>
      [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
      /* Limita el ancho máximo del contenedor principal */
      .block-container {
        max-width: 1000px;  /* ajusta este valor a tu gusto */
        padding-left: 1rem;
        padding-right: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Define las opciones con íconos
nav_items = [
    ("🏠 Introducción",         "Introducción"),
    ("🔍 Datos faltantes",       "Datos faltantes"),
    ("🗳️ Participación política", "Participación política"),
    ("🎯 Relevancia electoral",   "Relevancia electoral"),
    ("🧭 Ubicación ideológica",   "Ubicación ideológica"),
]

if "view" not in st.session_state:
    st.session_state.view = "Introducción"

# 2) Reemplaza tu bloque de sidebar por esto:
st.sidebar.title("Encuesta de Cultura Política")
for label, key in nav_items:
    if st.sidebar.button(label):
        st.session_state.view = key

# añadir un separador y la versión al fondo de la sidebar
st.sidebar.markdown("---")
st.sidebar.caption("Versión 1.0.0")  # o el número que corresponda

view = st.session_state.view

# ───────────────── Definición de etiquetas y variables ─────────────────────
global_participacion = {1: "Sí votó", 2: "No votó", 99: "NS/NR"}
part_title_map = {"P6933": "¿Votó en las presidenciales de 2018/2022?"}
# Razones de no voto
to_no_vote = {
    "P5336S1": "< 18 años",
    "P5336S2": "No cédula",
    "P5336S6": "Políticos son corruptos",
    "P5336S7": "Los partidos o movimientos políticos no representan a los ciudadanos",
    "P5336S8": "Los candidatos prometen y no cumplen",
    "P5336S10": "Falta de credibilidad en el proceso electoral",
    "P5336S11": "Desinterés",
    "P5336S13": "Inseguridad",
    "P5336S14": "Falta de puestos de votación",
    "P5336S15": "Dificultad de acceso",
    "P5336S17": "Costos de transporte en que se incurre para registrarse o para votar",
    "P5336S19": "Desinformación de como votar (falta de pedagogía electoral)",
    "P5336S12": "Otra",
}

group_to_codes = {
    "Animadversión política": [
        "P5336S6",  # Políticos son corruptos
        "P5336S7",  # Los partidos no representan
        "P5336S8",  # Candidatos prometen y no cumplen
        "P5336S10", # Falta de credibilidad en el proceso
        "P5336S19",  # Desinformación de cómo votar
        "P5336S11" # Desinterés
    ],
    "Dificultad logística": [
        "P5336S14", # Falta de puestos de votación
        "P5336S15", # Dificultad de acceso
        "P5336S17",  # Costos de transporte
        "P5336S13"  # Inseguridad
    ],
    "Menor ó Cedula": [
        "P5336S1",  # < 18 años
        "P5336S2"  # No cédula
    ],
    "Otra": [
        "P5336S12"  # Otra
    ]
}

# Razones de voto
to_vote = {
    "P5337S1": "Apoyo al candidato",
    "P5337S2": "Programa de gobierno",
    "P5337S3": "Ideología política",
    "P5337S4": "Responsabilidad ciudadana",
    "P5337S5": "Influencia familiar/social",
}
# Dificultades al votar
to_difficulties = {
    "P5338S1": "Mesa lejana",
    "P5338S2": "Colas largas",
    "P5338S3": "Falla de cédula",
    "P5338S4": "Horario inconveniente",
    "P5338S5": "Barreras físicas",
}
# Transparencia conteo de votos
# Incisos de P5339S1 a P5339S3 según diccionario DANE
# Reemplaza etiquetas por el texto descriptivo de cada inciso
# Transparencia conteo de votos (tres niveles territoriales)
to_transparency = {
    "P5339S1": "Transparencia a nivel municipal",
    "P5339S2": "Transparencia a nivel departamental",
    "P5339S3": "Transparencia a nivel nacional",
}

to_no_identify = {
    "P5324S2": "Falta de credibilidad en los partidos",         
    "P5324S3": "Desinterés",                                                                   
    "P5324S4": "La política se puede hacer por otras vías",              
    "P5324S6": "Promesas incumplidas",                                                          
    "P5324S7": "Escándalos de corrupción",                                                     
    "P5324S8": "Persiguen intereses diferentes al bienestar de la comunidad",                 
    "P5324S5": "Otra razón",                                                                    
}

# ─────────────────────── Funciones de visualización de visualización ────────────────────────

def plot_weighted_bar(df, var, labels, y_range=(0, 100)):
    weights = df.groupby(var)["WEIGHT"].sum()
    total = weights.sum()
    perc_df = (
        (weights / total * 100)
        .reset_index()
        .rename(columns={"WEIGHT": "percent"})
    )
    perc_df[var] = perc_df[var].map(labels)
    fig = px.bar(
        perc_df,
        x=var,
        y="percent",
        text=perc_df["percent"].apply(lambda x: f"{x:.1f}%"),
        labels={var: "Categoría", "percent": "% (ponderado)"},
        template="plotly_white",
        height=350,
    )
    fig.update_traces(marker_line_color="black", marker_line_width=1)
    fig.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        margin=dict(t=25, b=25),
    )
    fig.update_yaxes(range=y_range)
    return fig

def plot_grouped_reasons(df, group_map, y_range=(0, 40)):
    total = df["WEIGHT"].sum()
    data = {"Categoría": [], "percent": []}
    for cat, codes in group_map.items():
        w = sum(
            df.loc[df[code] == 1, "WEIGHT"].sum()
            for code in codes
            if code in df.columns
        )
        pct = 100 * w / total if total else 0
        data["Categoría"].append(cat)
        data["percent"].append(pct)

    perc_df = pd.DataFrame(data)
    fig = px.bar(
        perc_df,
        x="Categoría",
        y="percent",
        text=perc_df["percent"].apply(lambda x: f"{x:.1f}%"),
        labels={"percent": "% (ponderado)"},
        template="plotly_white",
        height=350,
    )
    fig.update_traces(marker_line_color="black", marker_line_width=1)
    fig.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        margin=dict(t=25, b=25),
    )
    fig.update_yaxes(range=y_range)
    return fig

def plot_reasons(df, reason_vars, y_range=(0, 40)):
    data = {"Razón": [], "percent": []}
    for var, label in reason_vars.items():
        if var in df.columns:
            data["Razón"].append(label)
            data["percent"].append(porcentaje_ponderado(df, var))
    perc_df = pd.DataFrame(data)
    fig = px.bar(
        perc_df,
        x="Razón",
        y="percent",
        text=perc_df["percent"].apply(lambda x: f"{x:.1f}%"),
        labels={"percent": "% (ponderado)"},
        template="plotly_white",
        height=350,
    )
    fig.update_traces(marker_line_color="black", marker_line_width=1)
    fig.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        margin=dict(t=25, b=25),
    )
    fig.update_yaxes(range=y_range)
    return fig

# ──────────────────────────── Render de vistas ──────────────────────────────
if view == "Introducción":
    # ——— Título grande con emoji —————————
    st.markdown("<h2 style='text-align:center;'>📊 Introducción</h2>", unsafe_allow_html=True)
    st.markdown("___")

    # ——— Descripción breve ——————————————————
    st.markdown(
        """
        **Bienvenido al comparativo interactivo de la Encuesta de Cultura Política (ECP) 2019 vs 2023**  
        Descubre en tiempo real cómo han cambiado:
        - ✅ La **participación electoral**  
        - 💡 Las **motivaciones y barreras** al voto  
        - 🎯 La **relevancia** de cada elección  
        - ⚖️ La **ubicación ideológica** y **transparencia** política
        """
    )

    # ——— Métricas clave en tres columnas ——————
    df19_cnt = len(df2019)
    df23_cnt = len(df2023)
    vars_cnt = len(df2019.columns)

    delta = df23_cnt - df19_cnt
    c1, c2, c3 = st.columns(3)
    c1.metric("Encuestados 2019", f"{df19_cnt:,}")
    c2.metric(
        "Encuestados 2023",
        f"{df23_cnt:,}",
        delta=f"{delta:+,} nuevos",
        delta_color="normal"
    )
    c3.metric("Variables analizadas", vars_cnt)

    st.markdown("___")

    # ——— Mini-guía de navegación —————————————
    st.markdown(
        """
        📌 **Cómo usar esta app**  
        1. Selecciona una sección en la barra lateral.  
        2. Expande los acordeones para ver detalles.  
        3. Compara 2019 vs 2023 de un vistazo.  
        4. Haz zoom o pasa el cursor sobre los gráficos para más información.  
        """
    )


elif view == "Participación política":
    # ——— Título y descripción ——————————————————————————————————
    st.markdown("<h2>🏛️ Participación política</h2>", unsafe_allow_html=True)
    st.markdown("**Compara cómo han cambiado las tasas de voto y sus motivaciones de 2019 a 2023**")
    st.markdown("---")

    # ——— Métricas clave ——————————————————————————————————————
    sí19 = porcentaje_ponderado(df2019, "P6933")
    sí23 = porcentaje_ponderado(df2023, "P6933")
    m1, m2 = st.columns(2)
    m1.metric("ECP 2019 — % que votó", f"{sí19:.1f}%")
    m2.metric("ECP 2023 — % que votó", f"{sí23:.1f}%")
    st.markdown("")

    # ——— Pestañas para secciones —————————————————————————————————
    tabs = st.tabs([
        "📈 General",
        "🚫 No votar",
        "✅ Votar",
        "⏳ Dificultades",
        "🔍 Transparencia"
    ])

    # Tab 0: Participación general
    with tabs[0]:
        st.subheader(part_title_map["P6933"])
        c1, c2 = st.columns(2)
        y_max = max(sí19, sí23)
        y_range = (0, round(y_max + 5, -1))
        c1.subheader("ECP 2019")
        c1.plotly_chart(
            plot_weighted_bar(df2019, "P6933", global_participacion, y_range),
            use_container_width=True
        )
        c2.subheader("ECP 2023")
        c2.plotly_chart(
            plot_weighted_bar(df2023, "P6933", global_participacion, y_range),
            use_container_width=True
        )

    # Tab 1: Razones agrupadas para NO votar
    with tabs[1]:
        st.subheader("Razones agrupadas para NO votar")
        # calcula escala común
        vals = [
            porcentaje_ponderado(d, code)
            for d in (df2019, df2023)
            for code in sum(group_to_codes.values(), [])
        ]
        y_range = (0, round(max(vals + [0]) + 5, -1))
        c1, c2 = st.columns(2)
        c1.subheader("ECP 2019")
        c1.plotly_chart(
            plot_grouped_reasons(df2019, group_to_codes, y_range),
            use_container_width=True
        )
        c2.subheader("ECP 2023")
        c2.plotly_chart(
            plot_grouped_reasons(df2023, group_to_codes, y_range),
            use_container_width=True
        )

    # Tab 2: Razones para votar
    with tabs[2]:
        st.subheader("Razones para votar en presidenciales")
        vals = [
            porcentaje_ponderado(d, var)
            for d in (df2019, df2023)
            for var in to_vote
        ]
        y_range = (0, round(max(vals + [0]) + 5, -1))
        c1, c2 = st.columns(2)
        c1.subheader("ECP 2019")
        c1.plotly_chart(
            plot_reasons(df2019, to_vote, y_range),
            use_container_width=True
        )
        c2.subheader("ECP 2023")
        c2.plotly_chart(
            plot_reasons(df2023, to_vote, y_range),
            use_container_width=True
        )

    # Tab 3: Dificultades al momento de votar
    with tabs[3]:
        st.subheader("Dificultades al momento de votar")
        vals = [
            porcentaje_ponderado(d, var)
            for d in (df2019, df2023)
            for var in to_difficulties
        ]
        y_range = (0, round(max(vals + [0]) + 5, -1))
        c1, c2 = st.columns(2)
        c1.subheader("ECP 2019")
        c1.plotly_chart(
            plot_reasons(df2019, to_difficulties, y_range),
            use_container_width=True
        )
        c2.subheader("ECP 2023")
        c2.plotly_chart(
            plot_reasons(df2023, to_difficulties, y_range),
            use_container_width=True
        )

    # Tab 4: Transparencia del conteo de votos
    with tabs[4]:
        st.subheader("Transparencia del conteo de votos")
        vals = [
            porcentaje_ponderado(d, var)
            for d in (df2019, df2023)
            for var in to_transparency
        ]
        y_range = (0, round(max(vals + [0]) + 5, -1))
        c1, c2 = st.columns(2)
        c1.subheader("ECP 2019")
        c1.plotly_chart(
            plot_reasons(df2019, to_transparency, y_range),
            use_container_width=True
        )
        c2.subheader("ECP 2023")
        c2.plotly_chart(
            plot_reasons(df2023, to_transparency, y_range),
            use_container_width=True
        )

elif view == "Relevancia electoral":
    # ——— Encabezado ————————————————————————————————————————————
    st.markdown("<h2>🎯 Relevancia electoral</h2>", unsafe_allow_html=True)
    st.markdown("Datos de importancia disponibles **solo para la ECP 2019**.")
    st.markdown("___")

    # Etiquetas abreviadas
    importance_vars = {
        "P5321S1": "JAC",           "P5321S2": "Gob.",
        "P5321S3": "Concejo M/D",   "P5321S4": "Alcaldía M.",
        "P5321S5": "Asamblea Dpto.","P5321S6": "Concejo Bog.",
        "P5321S7": "Alcaldía Bog.", "P5321S8": "Concejo Dist.",
        "P5321S9": "Asamblea Dist."
    }

    # ═════════ Tabs: Media · Distribución · Radar · Partidismo ═════════
    tab_media, tab_likert, tab_radar, tab_part = st.tabs(
        ["🔢 Media", "📊 Distribución", "🕸️ Perfil radar", "🗳️ Partidismo"]
    )

    # ── Tab 1 · Media ponderada ───────────────────────────────────────—
    with tab_media:
        st.subheader("Media ponderada de importancia (1 – 5)")
        st.plotly_chart(
            plot_importance_bar(df2019, importance_vars, y_range=(1, 5)),
            use_container_width=True
        )

    # ── Tab 2 · Likert apilado ─────────────────────────────────────────
    with tab_likert:
        st.subheader("Distribución completa por nivel 1 – 5")

        tot = df2019["WEIGHT"].sum()
        rec = []
        for var, lbl in importance_vars.items():
            for lvl in range(1, 6):
                w   = df2019.loc[df2019[var] == lvl, "WEIGHT"].sum()
                pct = 100 * w / tot if tot else 0
                rec.append({"Elección": lbl, "Nivel": str(lvl), "Pct": pct})
        likert_df = pd.DataFrame(rec)

        fig_likert = px.bar(
            likert_df, y="Elección", x="Pct", color="Nivel",
            orientation="h", text=likert_df["Pct"].apply(lambda x: f"{x:.1f}%"),
            labels={"Pct": "% ponderado"}, template="plotly_white", height=500
        )
        fig_likert.update_traces(textposition="inside", textfont_size=9, marker_line_width=0)
        fig_likert.update_layout(
            barmode="stack", margin=dict(l=200, t=30, b=30), legend_title="Nivel"
        )
        st.plotly_chart(fig_likert, use_container_width=True)

    # ── Tab 3 · Radar ─────────────────────────────────────────────────
    with tab_radar:
        st.subheader("Perfil de importancia (Radar)")
        radar_df = pd.DataFrame({
            "Elección": list(importance_vars.values()),
            "Valor": [weighted_mean(df2019, v) for v in importance_vars]
        })
        fig_radar = px.line_polar(
            radar_df, r="Valor", theta="Elección",
            line_close=True, template="plotly_white", height=500
        )
        fig_radar.update_traces(fill="toself")
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[1, 5], tickvals=[1, 2, 3, 4, 5])),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tab 4 · Partidismo ────────────────────────────────────────────
    with tab_part:
        st.subheader("Identificación partidista y razones")

        # ── Identificación (barras) sin y_range fijo y con autorange automático ──
        labels_id = {1: "Sí", 2: "No", 99: "NS/NR"}

        c1, c2 = st.columns(2)

        # ECP 2019
        c1.subheader("ECP 2019")
        fig2019_id = plot_weighted_bar(df2019, "P5323", labels_id, y_range=None)
        # Arranca en 0 y se autorangea al máximo de los datos
        fig2019_id.update_yaxes(rangemode="tozero", autorange=True)
        c1.plotly_chart(fig2019_id, use_container_width=True)

        # ECP 2023
        c2.subheader("ECP 2023")
        fig2023_id = plot_weighted_bar(df2023, "P5323", labels_id, y_range=None)
        fig2023_id.update_yaxes(rangemode="tozero", autorange=True)
        c2.plotly_chart(fig2023_id, use_container_width=True)

        st.markdown("---")

        # ─ Razones para no identificarse (select + donuts) ─
        razon_sel = st.selectbox(
            "Selecciona una razón:",
            list(to_no_identify.values()),
            key="sel_no_identify"
        )
        inv = {v: k for k, v in to_no_identify.items()}
        var = inv[razon_sel]

        yes19 = porcentaje_ponderado(df2019, var)
        yes23 = porcentaje_ponderado(df2023, var)

        pies = []
        for pct, label in [(yes19, "ECP 2019"), (yes23, "ECP 2023")]:
            pies.append(
                px.pie(
                    names=["Sí", "No"], values=[pct, 100 - pct],
                    hole=0.4, title=f"{razon_sel} — {label}",
                    template="plotly_white", height=330
                ).update_traces(
                    textinfo="percent+label",
                    marker=dict(line=dict(color="black", width=1))
                )
            )

        p1, p2 = st.columns(2)
        p1.plotly_chart(pies[0], use_container_width=True)
        p2.plotly_chart(pies[1], use_container_width=True)

elif view == "Ubicación ideológica":
    # ——— Encabezado atractivo ——————————————————————————
    st.markdown("<h2>🧭 Ubicación ideológica (1 = izq. - 10 = der.)</h2>", unsafe_allow_html=True)
    st.markdown("___")

    # ——— Utilitario: media ponderada 1–10 —————————————————————
    def w_mean_1_10(df, var="P5328"):
        mask = df[var].between(1, 10)
        w    = df.loc[mask, "WEIGHT"]
        return (df.loc[mask, var] * w).sum() / w.sum() if w.sum() else None

    avg19 = w_mean_1_10(df2019)
    avg23 = w_mean_1_10(df2023)

    # ══════════ Tabs: Promedio • Distribución • Grupos ═════════════════
    tab_gauge, tab_dist, tab_group = st.tabs([
        "📍 Promedio ponderado",
        "📊 Distribución completa",
        "🏷️ Grupos ideológicos"
    ])

    # ── Tab 1: Gauges ——————————————————————————————————————
    with tab_gauge:
        fig = go.Figure()
        # 2019
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=avg19,
            title={"text":"ECP 2019","font":{"size":14}},
            domain={"x":[0,0.45],"y":[0,1]},
            gauge=dict(
                axis=dict(range=[1,10], dtick=1),
                bar=dict(color="#1f77b4"),
                steps=[
                    {"range":[1,3],"color":"#e5f1ff"},
                    {"range":[3,7],"color":"#c8e0ff"},
                    {"range":[7,10],"color":"#acd0ff"},
                ],
            )
        ))
        # 2023
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=avg23,
            title={"text":"ECP 2023","font":{"size":14}},
            domain={"x":[0.55,1],"y":[0,1]},
            gauge=dict(
                axis=dict(range=[1,10], dtick=1),
                bar=dict(color="#ff5733"),
                steps=[
                    {"range":[1,3],"color":"#ffeae5"},
                    {"range":[3,7],"color":"#ffd2c7"},
                    {"range":[7,10],"color":"#ffb4a3"},
                ],
            )
        ))
        fig.update_layout(
            grid={"rows":1,"columns":2,"pattern":"independent"},
            template="plotly_white", height=340,
            margin=dict(t=40,b=0,l=20,r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Distribución completa ————————————————————————
    with tab_dist:
        st.subheader("Distribución ponderada de la escala 1–10")
        records = []
        for year, df in [("2019", df2019), ("2023", df2023)]:
            total_w = df["WEIGHT"].sum()
            for score in range(1, 11):
                w   = df.loc[df["P5328"] == score, "WEIGHT"].sum()
                pct = 100 * w / total_w if total_w else 0
                records.append({"Año": year, "Puntuación": str(score), "Pct": pct})
        dist_df = pd.DataFrame(records)

        fig_dist = px.bar(
            dist_df, y="Año", x="Pct", color="Puntuación",
            orientation="h", text=dist_df["Pct"].apply(lambda x: f"{x:.1f}%"),
            template="plotly_white", height=380,
            labels={"Pct":"% ponderado"}
        )
        fig_dist.update_traces(textposition="inside", textfont_size=9, marker_line_width=0)
        fig_dist.update_layout(
            barmode="stack", uniformtext_mode="hide",
            margin=dict(l=140,t=30,b=30), legend_title="Puntuación"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Tab 3: Grupos ideológicos ——————————————————————————
    with tab_group:
        st.subheader("Proporción por grupo ideológico")
        # Define grupos
        bins = [0, 3.5, 6.5, 10]
        labels = ["Izquierda", "Centro", "Derecha"]
        records = []
        for year, df in [("2019", df2019), ("2023", df2023)]:
            grp = pd.cut(df["P5328"], bins=bins, labels=labels)
            pct = (
                df.assign(ideology_group=grp)
                  .groupby("ideology_group")["WEIGHT"].sum()
                  .div(df["WEIGHT"].sum()) * 100
            ).reset_index(name="Pct")
            pct["Año"] = year
            records.append(pct)
        group_df = pd.concat(records)

        fig_group = px.bar(
            group_df, x="ideology_group", y="Pct", color="Año",
            barmode="group", text="Pct",
            labels={"ideology_group":"Grupo","Pct":"% ponderado"},
            template="plotly_white", height=400
        )
        fig_group.update_traces(texttemplate="%{text:.1f}%", marker_line_width=1)
        fig_group.update_layout(margin=dict(t=30,b=30))
        st.plotly_chart(fig_group, use_container_width=True)

elif view == "Datos faltantes":
    st.header("🔍 Análisis de datos faltantes")
    st.markdown(
        "Explora el porcentaje de valores faltantes por variable "
        "en cada ronda de la ECP."
    )

    # Calcula porcentaje de faltantes por variable
    miss19 = df2019.isna().mean().mul(100).reset_index()
    miss19.columns = ["Variable", "% faltante 2019"]
    miss23 = df2023.isna().mean().mul(100).reset_index()
    miss23.columns = ["Variable", "% faltante 2023"]

    # Combina ambos años
    miss_df = pd.merge(
        miss19, miss23, on="Variable", how="outer"
    ).fillna(0).sort_values("% faltante 2019", ascending=False)

    # Muestra tabla interactiva
    st.dataframe(
        miss_df.style.format({"% faltante 2019": "{:.1f}%", "% faltante 2023": "{:.1f}%"}),
        use_container_width=True
    )

    # Gráfico de barras de las top 15 variables con más faltantes en 2019
    top_n = 15
    top_df = miss_df.head(top_n)
    fig_bar = px.bar(
        top_df,
        x="Variable",
        y=["% faltante 2019", "% faltante 2023"],
        barmode="group",
        text_auto=".1f",
        labels={"value": "% faltante", "variable": "Año"},
        template="plotly_white",
        height=450,
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        margin=dict(b=200, t=30)
    )
    st.plotly_chart(fig_bar, use_container_width=True)