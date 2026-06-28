import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import cv2
from pathlib import Path

# =====================================================
# AI LUX — Streamlit app
# =====================================================

st.set_page_config(
    page_title="AI LUX",
    page_icon="💡",
    layout="wide",
)

# -----------------------------
# PATHS
# -----------------------------
ASSETS_DIR = Path("assets")
LOGO_AI_LUX = ASSETS_DIR / "logo_ai_lux.png"
LOGO_UCLOUVAIN = ASSETS_DIR / "logo_uclouvain_white.png"
LOGO_INNOVIRIS = ASSETS_DIR / "logo_innoviris_white.png"
LOGO_LAB = ASSETS_DIR / "logo_lab_white.png"

# -----------------------------
# MODEL + PARAMETERS
# -----------------------------

@st.cache_resource
def load_model():
    return pickle.load(open("LuxModel.sav", "rb"))

regr = load_model()

THRESHOLD = 175
BLUR = 25
KSIZE = 21

VARS = [
    "orig_thr", "blur_thr", "blur_wa", "blur_ta", "blur_ba",
    "blur_ha", "op_thr", "dist_thr", "op_counts"
]

YELLOW = "#F1CF00"
WHITE = "#FFFFFF"
GREY = "#8A8A8A"
DARK = "#050505"
CARD = "#090909"
LINE = "#303030"

# -----------------------------
# IMAGE PROCESSING
# -----------------------------

def mask_ratios(mask):
    mask_ratio = mask.mean() / 255
    mask_vals = [i.sum() / 255 for i in mask]

    mask_wr = np.sum(mask, axis=0)
    mask_wr[mask_wr > 1] = 1
    mask_wr = np.sum(mask_wr)
    mask_wa = (mask_wr * 360) / mask.shape[1]

    mask_tr = 0
    for i in mask:
        if i.sum() / 255 < max(mask_vals) / 7:
            mask_tr += 1
        else:
            break
    mask_ta = (mask_tr * 180) / mask.shape[0]

    mask_br = 0
    for i in np.flipud(mask):
        if i.sum() / 255 < max(mask_vals) / 7:
            mask_br += 1
        else:
            break
    mask_ba = (mask_br * 180) / mask.shape[0]

    mask_hr = mask.shape[0] - (mask_tr + mask_br)
    mask_ha = (mask_hr * 180) / mask.shape[0]

    return [mask_ratio, mask_wa, mask_ta, mask_ba, mask_ha]


def data_from_image(image_file):
    datos = []

    img = Image.open(image_file).convert("RGB")
    img_np = np.array(img)

    thres = np.copy(img_np)
    thres[thres > THRESHOLD] = 255
    thres[thres < 255] = 0

    orig_thr = round((thres.mean() / 255), 4)
    datos.append(orig_thr)

    image_blur = cv2.medianBlur(thres, BLUR)
    image_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)

    blur_thr, blur_wa, blur_ta, blur_ba, blur_ha = mask_ratios(image_blur_gray)
    datos.extend([blur_thr, blur_wa, blur_ta, blur_ba, blur_ha])

    _, image_thresh = cv2.threshold(image_blur_gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((KSIZE, KSIZE), np.uint8)
    opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)

    op_thr = 1 - (opening.mean() / 255)
    datos.append(op_thr)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, dist_thr = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    dist_thr = np.uint8(dist_thr)
    dist_thr = 1 - (dist_thr.mean() / 255)
    datos.append(dist_thr)

    edge = cv2.Canny(opening, 120, 210)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    op_counts = len(contours)
    datos.append(op_counts)

    return datos


def classify(prediction, area_type):
    if area_type == "urban":
        if prediction < 3:
            return "UNDERLIT"
        if prediction < 15:
            return "PROPERLY LIT"
        return "OVERLIT"

    if area_type == "green":
        if prediction < 2:
            return "UNDERLIT"
        if prediction < 10:
            return "PROPERLY LIT"
        return "OVERLIT"

    return "UNKNOWN"


def class_color(label):
    if label == "UNDERLIT":
        return GREY
    if label == "PROPERLY LIT":
        return WHITE
    if label == "OVERLIT":
        return YELLOW
    return WHITE

# -----------------------------
# CSS
# -----------------------------

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {DARK};
        color: {WHITE};
        font-family: Helvetica, Arial, sans-serif;
    }}

    header {{ visibility: hidden; }}

    .block-container {{
        padding-top: 2.1rem;
        padding-bottom: 2.2rem;
        max-width: 1500px;
    }}

    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: {WHITE};
        font-family: Helvetica, Arial, sans-serif;
    }}

    .top-rule {{
        border-bottom: 2px solid {YELLOW};
        margin: 0.85rem 0 2rem 0;
    }}

    .nav {{
        color: {YELLOW};
        font-size: 15px;
        font-weight: 900;
        letter-spacing: 0.7px;
        text-transform: uppercase;
        border-bottom: 2px solid {YELLOW};
        padding-bottom: 9px;
        text-align: center;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }}

    .opensource {{
        color: {WHITE};
        font-size: 14px;
        opacity: 0.78;
        text-align: right;
        padding-top: 6px;
    }}

    .card {{
        border: 1px solid {LINE};
        border-radius: 12px;
        padding: 24px;
        background: {CARD};
        margin-bottom: 18px;
    }}

    .upload-card {{
        border: 1px dashed #565656;
        border-radius: 12px;
        padding: 30px 24px;
        text-align: center;
        background: #030303;
        margin-bottom: 15px;
    }}

    .upload-title {{
        font-size: 18px;
        font-weight: 900;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: {WHITE};
    }}

    .upload-subtitle {{
        font-size: 14px;
        opacity: 0.72;
        margin-top: 8px;
    }}

    .section-title {{
        font-size: 18px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 18px;
        color: {WHITE};
    }}

    .yellow {{ color: {YELLOW} !important; }}
    .grey {{ color: {GREY} !important; }}

    .big-number {{
        font-size: 92px;
        font-weight: 900;
        color: {YELLOW};
        line-height: 0.95;
        letter-spacing: -2px;
        margin-top: 18px;
        margin-bottom: 8px;
    }}

    .unit {{
        font-size: 42px;
        color: {WHITE};
        font-weight: 800;
    }}

    .small-label {{
        font-size: 13px;
        color: {WHITE};
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
    }}

    .image-frame {{
        border: 2px solid {WHITE};
        border-radius: 8px;
        padding: 0;
        overflow: hidden;
        margin-bottom: 12px;
    }}

    .meta-row {{
        display: flex;
        justify-content: space-between;
        gap: 20px;
        color: {WHITE};
        opacity: 0.78;
        font-size: 14px;
        margin-bottom: 18px;
    }}

    .data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}

    .data-table th {{
        color: {WHITE};
        text-align: left;
        padding: 10px 10px;
        border-bottom: 1px solid {LINE};
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 12px;
    }}

    .data-table td {{
        padding: 9px 10px;
        border-bottom: 1px solid #202020;
        color: {WHITE};
    }}

    .data-table .value {{
        color: {YELLOW};
        font-weight: 800;
    }}

    .classification-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0;
        text-align: center;
    }}

    .class-box {{ padding: 12px 18px 8px 18px; }}
    .class-box:first-child {{ border-right: 1px solid {LINE}; }}

    .class-label {{
        font-size: 24px;
        font-weight: 900;
        text-transform: uppercase;
        line-height: 1.05;
        margin-top: 18px;
    }}

    .class-note {{
        font-size: 12px;
        opacity: 0.58;
        margin-top: 10px;
    }}

    .bar-wrap {{
        margin-top: 38px;
        margin-bottom: 22px;
    }}

    .scale {{
        display: flex;
        justify-content: space-between;
        font-size: 15px;
        color: {WHITE};
        margin-bottom: 10px;
        opacity: 0.9;
    }}

    .bar {{
        height: 8px;
        border-radius: 8px;
        background: linear-gradient(
            to right,
            #7A7A7A 0%,
            #7A7A7A 25%,
            {WHITE} 25%,
            {WHITE} 60%,
            {YELLOW} 60%,
            {YELLOW} 100%
        );
        position: relative;
    }}

    .marker {{
        position: absolute;
        top: -8px;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: {DARK};
        border: 5px solid {YELLOW};
    }}

    .legend-row {{
        display: flex;
        justify-content: space-between;
        margin-top: 18px;
        font-weight: 900;
        text-transform: uppercase;
        font-size: 13px;
        letter-spacing: 0.5px;
    }}

    .interpretation p {{
        font-size: 15px;
        line-height: 1.5;
        opacity: 0.9;
    }}

    .footer {{
        border-top: 1px solid {LINE};
        margin-top: 28px;
        padding-top: 24px;
        padding-bottom: 8px;
    }}

    .fallback-logo {{
        font-size: 42px;
        line-height: 0.9;
        font-weight: 900;
        color: {YELLOW};
        letter-spacing: -1px;
    }}

    .fallback-subtitle {{
        font-size: 14px;
        color: {WHITE};
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-top: 7px;
    }}

    .stFileUploader label {{ color: {WHITE} !important; }}

    .stFileUploader section {{
        background-color: #050505 !important;
        border: 1px dashed #565656 !important;
        border-radius: 12px !important;
    }}

    .stFileUploader button {{
        background-color: {YELLOW} !important;
        color: #000000 !important;
        font-weight: 900 !important;
        border-radius: 6px !important;
        border: 0 !important;
    }}

    .stExpander {{
        border: 1px solid {LINE} !important;
        border-radius: 8px !important;
        background-color: #050505 !important;
    }}

    @media (max-width: 900px) {{
        .big-number {{ font-size: 62px; }}
        .unit {{ font-size: 30px; }}
        .classification-grid {{ grid-template-columns: 1fr; }}
        .class-box:first-child {{ border-right: none; border-bottom: 1px solid {LINE}; }}
        .meta-row {{ flex-direction: column; gap: 4px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# HEADER
# -----------------------------

h1, h2, h3 = st.columns([2.3, 1, 1])

with h1:
    if LOGO_AI_LUX.exists():
        st.image(str(LOGO_AI_LUX), width=340)
    else:
        st.markdown(
            """
            <div class="fallback-logo">AI LUX</div>
            <div class="fallback-subtitle">AI-powered nighttime illuminance estimation</div>
            """,
            unsafe_allow_html=True,
        )

with h2:
    st.markdown('<div class="nav">PREDICT</div>', unsafe_allow_html=True)

with h3:
    st.markdown('<div class="opensource">Open-source AI tool</div>', unsafe_allow_html=True)

st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)

# -----------------------------
# APP LAYOUT
# -----------------------------

left, right = st.columns([1.55, 1])

with left:
    st.markdown(
        """
        <div class="upload-card">
            <div class="upload-title"><span class="yellow">360°</span> Nighttime Image</div>
            <div class="upload-subtitle">Drag & drop a panoramic image or click to browse</div>
            <div class="upload-subtitle" style="font-size:12px; opacity:0.55;">JPG · PNG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

if uploaded_file is not None:
    predicted_vals = data_from_image(uploaded_file)
    prediction = float(regr.predict([predicted_vals]))

    urban_class = classify(prediction, "urban")
    green_class = classify(prediction, "green")

    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    filename = uploaded_file.name

    with left:
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="meta-row">
                <div>{filename}</div>
                <div>Resolution: {width} × {height} px</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title"><span class="yellow">◎</span> Extracted Data</div>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame({
            "Parameter": VARS,
            "Value": [round(v, 5) if isinstance(v, float) else v for v in predicted_vals],
        })

        rows = ""
        for i in range(0, len(df), 2):
            p1 = df.iloc[i]["Parameter"]
            v1 = df.iloc[i]["Value"]

            if i + 1 < len(df):
                p2 = df.iloc[i + 1]["Parameter"]
                v2 = df.iloc[i + 1]["Value"]
            else:
                p2, v2 = "", ""

            rows += f"""
            <tr>
                <td>{p1}</td><td class="value">{v1}</td>
                <td>{p2}</td><td class="value">{v2}</td>
            </tr>
            """

        st.markdown(
            f"""
            <table class="data-table">
                <tr>
                    <th>Parameter</th><th>Value</th>
                    <th>Parameter</th><th>Value</th>
                </tr>
                {rows}
            </table>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Advanced details"):
            st.dataframe(df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        marker_position = min(max(prediction / 50 * 100, 0), 100)

        st.markdown(
            f"""
            <div class="card">
                <div class="section-title"><span class="yellow">☼</span> Estimated Illuminance</div>
                <div class="big-number">{prediction:.2f} <span class="unit">lx</span></div>
                <div class="small-label">Horizontal illuminance</div>

                <div class="bar-wrap">
                    <div class="scale">
                        <span>0 lx</span><span>5</span><span>10</span><span>20</span><span>&gt;50</span>
                    </div>
                    <div class="bar">
                        <div class="marker" style="left: calc({marker_position}% - 11px);"></div>
                    </div>
                    <div class="legend-row">
                        <span style="color:{GREY};">Underlit</span>
                        <span style="color:{WHITE};">Properly lit</span>
                        <span style="color:{YELLOW};">Overlit</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="section-title"><span class="yellow">◇</span> Classification</div>
                <div class="classification-grid">
                    <div class="class-box">
                        <div class="small-label">Urban Area · P1</div>
                        <div class="class-label" style="color:{class_color(urban_class)};">{urban_class}</div>
                        <div class="class-note">Non-green public space thresholds</div>
                    </div>
                    <div class="class-box">
                        <div class="small-label">Green Area · P2</div>
                        <div class="class-label" style="color:{class_color(green_class)};">{green_class}</div>
                        <div class="class-note">Biodiversity-sensitive thresholds</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card interpretation">
                <div class="section-title"><span class="yellow">▣</span> Interpretation</div>
                <p>
                The estimated horizontal illuminance is
                <span class="yellow"><b>{prediction:.2f} lx</b></span>.
                According to AI LUX, this location is classified as
                <b style="color:{class_color(urban_class)};">{urban_class}</b>
                for urban public spaces and
                <b style="color:{class_color(green_class)};">{green_class}</b>
                for green areas.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    with right:
        st.markdown(
            """
            <div class="card">
                <div class="section-title"><span class="yellow">☼</span> Estimated Illuminance</div>
                <div class="big-number">-- <span class="unit">lx</span></div>
                <div class="small-label">Upload a 360° nighttime image to start.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
                <div class="section-title"><span class="yellow">◇</span> Classification</div>
                <div class="classification-grid">
                    <div class="class-box">
                        <div class="small-label">Urban Area · P1</div>
                        <div class="class-label grey">--</div>
                    </div>
                    <div class="class-box">
                        <div class="small-label">Green Area · P2</div>
                        <div class="class-label grey">--</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------
# FOOTER
# -----------------------------

st.markdown('<div class="footer">', unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns([1.5, 1, 1, 0.8])

with f1:
    if LOGO_AI_LUX.exists():
        st.image(str(LOGO_AI_LUX), width=220)
    else:
        st.markdown('<div class="fallback-logo">AI LUX</div>', unsafe_allow_html=True)

with f2:
    if LOGO_UCLOUVAIN.exists():
        st.image(str(LOGO_UCLOUVAIN), width=165)
    else:
        st.markdown('<div class="small-label">UCLouvain</div>', unsafe_allow_html=True)

with f3:
    if LOGO_INNOVIRIS.exists():
        st.image(str(LOGO_INNOVIRIS), width=165)
    else:
        st.markdown('<div class="small-label">innoviris.brussels</div>', unsafe_allow_html=True)

with f4:
    if LOGO_LAB.exists():
        st.image(str(LOGO_LAB), width=120)
    else:
        st.markdown('<div class="small-label">LAB Research</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
