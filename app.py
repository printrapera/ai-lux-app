import base64
from pathlib import Path
import pickle

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# =====================================================
# AI LUX — Streamlit single-page app
# =====================================================

st.set_page_config(
    page_title="AI LUX",
    page_icon="💡",
    layout="wide",
)

# -----------------------------
# Paths
# -----------------------------

ASSETS_DIR = Path("assets")

def find_asset(filename: str) -> Path | None:
    """Find assets either inside /assets or at repository root."""
    for path in [ASSETS_DIR / filename, Path(filename)]:
        if path.exists():
            return path
    return None

LOGO_AI_LUX = find_asset("logo_ai_lux.png")
LOGO_UCLOUVAIN = find_asset("logo_uclouvain_white.png")
LOGO_INNOVIRIS = find_asset("logo_innoviris_white.png")
LOGO_LAB = find_asset("logo_lab_white.png")


# -----------------------------
# Model + parameters
# -----------------------------

@st.cache_resource
def load_model():
    with open("LuxModel.sav", "rb") as f:
        return pickle.load(f)

regr = load_model()

THRESHOLD = 175
BLUR = 25
KSIZE = 21

VARS = [
    "orig_thr",
    "blur_thr",
    "blur_wa",
    "blur_ta",
    "blur_ba",
    "blur_ha",
    "op_thr",
    "dist_thr",
    "op_counts",
]

YELLOW = "#F1CF00"
WHITE = "#FFFFFF"
GREY = "#8A8A8A"
DARK = "#050505"
CARD = "#090909"
LINE = "#303030"


# -----------------------------
# Helpers
# -----------------------------

def img_to_base64(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def image_tag(path: Path | None, css_class: str, alt: str = "") -> str:
    encoded = img_to_base64(path)
    if encoded is None:
        return ""
    suffix = path.suffix.lower().replace(".", "")
    mime = "jpeg" if suffix in ["jpg", "jpeg"] else "png"
    return f'<img class="{css_class}" src="data:image/{mime};base64,{encoded}" alt="{alt}"/>'


# -----------------------------
# Image processing
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
    data = []

    img = Image.open(image_file).convert("RGB")
    img_np = np.array(img)

    thres = np.copy(img_np)
    thres[thres > THRESHOLD] = 255
    thres[thres < 255] = 0

    orig_thr = round((thres.mean() / 255), 4)
    data.append(orig_thr)

    image_blur = cv2.medianBlur(thres, BLUR)
    image_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)

    blur_thr, blur_wa, blur_ta, blur_ba, blur_ha = mask_ratios(image_blur_gray)
    data.extend([blur_thr, blur_wa, blur_ta, blur_ba, blur_ha])

    _, image_thresh = cv2.threshold(
        image_blur_gray,
        240,
        255,
        cv2.THRESH_BINARY_INV,
    )

    kernel = np.ones((KSIZE, KSIZE), np.uint8)
    opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)

    op_thr = 1 - (opening.mean() / 255)
    data.append(op_thr)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, dist_thr = cv2.threshold(
        dist_transform,
        0.3 * dist_transform.max(),
        255,
        0,
    )
    dist_thr = np.uint8(dist_thr)
    dist_thr = 1 - (dist_thr.mean() / 255)
    data.append(dist_thr)

    edge = cv2.Canny(opening, 120, 210)
    contours, _ = cv2.findContours(
        edge,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    op_counts = len(contours)
    data.append(op_counts)

    return data


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
    :root {{
        --yellow: {YELLOW};
        --white: {WHITE};
        --grey: {GREY};
        --dark: {DARK};
        --card: {CARD};
        --line: {LINE};
    }}

    .stApp {{
        background-color: var(--dark);
        color: var(--white);
        font-family: Helvetica, Arial, sans-serif;
    }}

    header {{
        visibility: hidden;
    }}

    .block-container {{
        padding: 1.15rem 1.6rem 1.2rem 1.6rem;
        max-width: 1500px;
    }}

    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: var(--white);
        font-family: Helvetica, Arial, sans-serif;
    }}

    .app-shell {{
        width: 100%;
    }}

    .topbar {{
        display: grid;
        grid-template-columns: minmax(230px, 390px) 1fr auto;
        align-items: start;
        column-gap: 28px;
        margin-bottom: 0.55rem;
    }}

    .brand-logo {{
        display: block;
        width: clamp(210px, 25vw, 340px);
        height: auto;
    }}

    .opensource {{
        color: var(--white);
        font-size: clamp(10px, 0.95vw, 14px);
        opacity: 0.85;
        text-align: right;
        padding-top: 8px;
        white-space: nowrap;
    }}

    .top-rule {{
        border-bottom: 2px solid var(--yellow);
        margin: 0.65rem 0 1.25rem 0;
    }}

    .main-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1.58fr) minmax(310px, 1fr);
        gap: 24px;
        align-items: start;
    }}

    .card {{
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: clamp(15px, 1.45vw, 24px);
        background: var(--card);
        margin-bottom: 16px;
    }}

    .upload-card {{
        border: 1px dashed #565656;
        border-radius: 12px;
        padding: clamp(20px, 2.4vw, 34px) 24px;
        text-align: center;
        background: #030303;
        margin-bottom: 13px;
    }}

    .upload-icon {{
        font-size: clamp(32px, 3.2vw, 48px);
        color: var(--yellow);
        line-height: 1;
        margin-bottom: 12px;
    }}

    .upload-title {{
        font-size: clamp(13px, 1.4vw, 18px);
        font-weight: 900;
        letter-spacing: 0.7px;
        text-transform: uppercase;
        color: var(--white);
    }}

    .upload-subtitle {{
        font-size: clamp(11px, 1vw, 14px);
        opacity: 0.72;
        margin-top: 8px;
    }}

    .section-title {{
        font-size: clamp(13px, 1.2vw, 18px);
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 15px;
        color: var(--white);
    }}

    .yellow {{
        color: var(--yellow) !important;
    }}

    .grey {{
        color: var(--grey) !important;
    }}

    .lux-number {{
        font-size: clamp(54px, 6.8vw, 96px);
        font-weight: 900;
        color: var(--yellow);
        line-height: 0.95;
        letter-spacing: -2px;
        margin-top: 8px;
        margin-bottom: 8px;
    }}

    .unit {{
        font-size: clamp(28px, 3vw, 44px);
        color: var(--white);
        font-weight: 800;
    }}

    .small-label {{
        font-size: clamp(10px, 0.95vw, 13px);
        color: var(--white);
        opacity: 0.78;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
    }}

    .image-frame {{
        border: 2px solid var(--white);
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 10px;
        background: #000;
    }}

    .image-frame img {{
        width: 100%;
        max-height: 310px;
        object-fit: cover;
        display: block;
    }}

    .meta-row {{
        display: flex;
        justify-content: space-between;
        gap: 20px;
        color: var(--white);
        opacity: 0.78;
        font-size: clamp(10px, 0.9vw, 13px);
        margin-bottom: 12px;
    }}

    .data-card {{
        padding: clamp(14px, 1.2vw, 20px);
    }}

    .data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: clamp(10px, 0.88vw, 13px);
    }}

    .data-table th {{
        color: var(--white);
        text-align: left;
        padding: 8px 10px;
        border-bottom: 1px solid var(--line);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: clamp(9px, 0.75vw, 11px);
    }}

    .data-table td {{
        padding: 7px 10px;
        border-bottom: 1px solid #202020;
        color: var(--white);
    }}

    .data-table .value {{
        color: var(--yellow);
        font-weight: 800;
    }}

    .lux-scale-wrap {{
        margin-top: clamp(22px, 2.6vw, 38px);
        margin-bottom: 18px;
    }}

    .lux-scale-labels {{
        display: flex;
        justify-content: space-between;
        font-size: clamp(11px, 1vw, 15px);
        color: var(--white);
        margin-bottom: 10px;
        opacity: 0.9;
    }}

    .lux-scale-track {{
        height: 8px;
        border-radius: 8px;
        background: linear-gradient(
            to right,
            #7A7A7A 0%,
            #7A7A7A 25%,
            var(--white) 25%,
            var(--white) 60%,
            var(--yellow) 60%,
            var(--yellow) 100%
        );
        position: relative;
    }}

    .lux-scale-marker {{
        position: absolute;
        top: -8px;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: var(--dark);
        border: 5px solid var(--yellow);
    }}

    .lux-scale-legend {{
        display: flex;
        justify-content: space-between;
        margin-top: 18px;
        font-weight: 900;
        text-transform: uppercase;
        font-size: clamp(10px, 0.9vw, 13px);
        letter-spacing: 0.5px;
    }}

    .classification-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0;
        text-align: center;
    }}

    .class-box {{
        padding: 12px 18px 8px 18px;
    }}

    .class-box:first-child {{
        border-right: 1px solid var(--line);
    }}

    .class-label {{
        font-size: clamp(20px, 2vw, 28px);
        font-weight: 900;
        text-transform: uppercase;
        line-height: 1.05;
        margin-top: 16px;
        color: var(--white);
    }}

    .class-note {{
        font-size: clamp(9px, 0.8vw, 12px);
        opacity: 0.58;
        margin-top: 10px;
    }}

    .interpretation p {{
        font-size: clamp(12px, 1vw, 15px);
        line-height: 1.48;
        opacity: 0.92;
        margin-bottom: 0;
    }}

    .footer {{
        border-top: 1px solid var(--line);
        margin-top: 18px;
        padding-top: 22px;
        padding-bottom: 4px;
    }}

    .footer-logos {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: clamp(52px, 8vw, 110px);
        width: 100%;
    }}

    .footer-logo {{
        max-height: 54px;
        max-width: 210px;
        object-fit: contain;
        display: block;
    }}

    .footer-logo.lab {{
        max-height: 64px;
    }}

    .fallback-logo {{
        font-size: 42px;
        line-height: 0.9;
        font-weight: 900;
        color: var(--yellow);
        letter-spacing: -1px;
    }}

    /* Streamlit file uploader */
    .stFileUploader label {{
        color: var(--white) !important;
    }}

    .stFileUploader section {{
        background-color: #050505 !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        min-height: 50px !important;
    }}

    .stFileUploader button {{
        background-color: var(--yellow) !important;
        color: #000000 !important;
        font-weight: 900 !important;
        border-radius: 6px !important;
        border: 0 !important;
    }}

    .stExpander {{
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        background-color: #050505 !important;
    }}

    /* Prevent white blocks from Streamlit components */
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        background: #050505 !important;
    }}

    @media (max-width: 1050px) {{
        .block-container {{
            padding: 1rem 1rem 1.1rem 1rem;
        }}

        .main-grid {{
            grid-template-columns: 1fr;
        }}

        .image-frame img {{
            max-height: none;
            object-fit: contain;
        }}

        .footer-logos {{
            gap: 34px;
            flex-wrap: wrap;
        }}

        .footer-logo {{
            max-height: 42px;
            max-width: 165px;
        }}
    }}

    @media (max-width: 700px) {{
        .topbar {{
            grid-template-columns: 1fr;
            row-gap: 10px;
        }}

        .opensource {{
            text-align: left;
        }}

        .classification-grid {{
            grid-template-columns: 1fr;
        }}

        .class-box:first-child {{
            border-right: none;
            border-bottom: 1px solid var(--line);
        }}

        .meta-row {{
            flex-direction: column;
            gap: 4px;
        }}

        .footer-logos {{
            flex-direction: column;
            gap: 18px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------

st.markdown('<div class="app-shell">', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="topbar">
        <div>
            {image_tag(LOGO_AI_LUX, "brand-logo", "AI LUX")}
        </div>
        <div></div>
        <div class="opensource">Open-source AI tool</div>
    </div>
    <div class="top-rule"></div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Main app
# -----------------------------

st.markdown('<div class="main-grid">', unsafe_allow_html=True)

left_col, right_col = st.columns([1.58, 1], gap="large")

with left_col:
    st.markdown(
        """
        <div class="upload-card">
            <div class="upload-icon">☁</div>
            <div class="upload-title">Drag & drop a <span class="yellow">360°</span> nighttime image here</div>
            <div class="upload-subtitle">or click to browse</div>
            <div class="upload-subtitle" style="font-size:12px; opacity:0.55;">JPG · PNG · Max 200MB</div>
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

    with left_col:
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

        st.markdown('<div class="card data-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title"><span class="yellow">◎</span> Extracted Data</div>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame({
            "Parameter": VARS,
            "Value": [
                round(v, 5) if isinstance(v, float) else v
                for v in predicted_vals
            ],
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

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        marker_position = min(max(prediction / 50 * 100, 0), 100)
        urban_color = class_color(urban_class)
        green_color = class_color(green_class)

        st.markdown(
            f"""
            <div class="card">
                <div class="section-title"><span class="yellow">☼</span> Estimated Illuminance</div>
                <div class="lux-number">{prediction:.2f} <span class="unit">lx</span></div>
                <div class="small-label">Horizontal illuminance</div>

                <div class="lux-scale-wrap">
                    <div class="lux-scale-labels">
                        <span>0</span><span>5</span><span>10</span><span>20</span><span>50+</span>
                    </div>
                    <div class="lux-scale-track">
                        <div class="lux-scale-marker" style="left: calc({marker_position}% - 11px);"></div>
                    </div>
                    <div class="lux-scale-legend">
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
                        <div class="class-label" style="color:{urban_color};">{urban_class}</div>
                        <div class="class-note">Non-green public space thresholds</div>
                    </div>
                    <div class="class-box">
                        <div class="small-label">Green Area · P2</div>
                        <div class="class-label" style="color:{green_color};">{green_class}</div>
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
                    <b style="color:{urban_color};">{urban_class}</b>
                    for urban public spaces and
                    <b style="color:{green_color};">{green_class}</b>
                    for green areas.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    with right_col:
        st.markdown(
            """
            <div class="card">
                <div class="section-title"><span class="yellow">☼</span> Estimated Illuminance</div>
                <div class="lux-number">-- <span class="unit">lx</span></div>
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

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Footer
# -----------------------------

st.markdown(
    f"""
    <div class="footer">
        <div class="footer-logos">
            {image_tag(LOGO_UCLOUVAIN, "footer-logo", "UCLouvain")}
            {image_tag(LOGO_INNOVIRIS, "footer-logo", "innoviris.brussels")}
            {image_tag(LOGO_LAB, "footer-logo lab", "LAB Research")}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
