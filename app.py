import base64
import html
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# =====================================================
# AI LUX — Streamlit one-page app
# =====================================================

st.set_page_config(
    page_title="AI LUX",
    page_icon="💡",
    layout="wide",
)


# =====================================================
# PATHS
# =====================================================

def find_asset(filename: str) -> Path:
    """Find assets whether they are stored in /assets or in the repository root."""
    candidates = [
        Path("assets") / filename,
        Path(filename),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


LOGO_AI_LUX = find_asset("logo_ai_lux.png")
LOGO_UCLOUVAIN = find_asset("logo_uclouvain_white.png")
LOGO_INNOVIRIS = find_asset("logo_innoviris_white.png")
LOGO_LAB = find_asset("logo_lab_white.png")


def image_to_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower().replace(".", "")
    mime = "jpeg" if suffix in ["jpg", "jpeg"] else "png"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{mime};base64,{encoded}"


LOGO_AI_LUX_B64 = image_to_base64(LOGO_AI_LUX)
LOGO_UCLOUVAIN_B64 = image_to_base64(LOGO_UCLOUVAIN)
LOGO_INNOVIRIS_B64 = image_to_base64(LOGO_INNOVIRIS)
LOGO_LAB_B64 = image_to_base64(LOGO_LAB)


# =====================================================
# MODEL + PARAMETERS
# =====================================================

@st.cache_resource
def load_model():
    with open("LuxModel.sav", "rb") as model_file:
        return pickle.load(model_file)


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


# =====================================================
# IMAGE PROCESSING
# =====================================================

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

    _, image_thresh = cv2.threshold(image_blur_gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((KSIZE, KSIZE), np.uint8)
    opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)

    op_thr = 1 - (opening.mean() / 255)
    data.append(op_thr)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, dist_thr = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    dist_thr = np.uint8(dist_thr)
    dist_thr = 1 - (dist_thr.mean() / 255)
    data.append(dist_thr)

    edge = cv2.Canny(opening, 120, 210)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def safe_round(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), 5)
    return value


# =====================================================
# CSS
# =====================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {DARK};
        color: {WHITE};
        font-family: Helvetica, Arial, sans-serif;
    }}

    header {{ visibility: hidden; height: 0rem; }}

    .block-container {{
        padding-top: 1.1rem;
        padding-bottom: 1.1rem;
        max-width: 1180px;
    }}

    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: {WHITE};
        font-family: Helvetica, Arial, sans-serif;
    }}

    .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 24px;
    }}

    .brand-logo {{
        width: 320px;
        max-width: 42vw;
        height: auto;
        display: block;
    }}

    .fallback-logo {{
        font-size: 42px;
        line-height: 0.9;
        font-weight: 900;
        color: {YELLOW};
        letter-spacing: -1px;
    }}

    .fallback-subtitle {{
        font-size: 13px;
        color: {WHITE};
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.4px;
        margin-top: 6px;
    }}

    .opensource {{
        color: {WHITE};
        font-size: 13px;
        opacity: 0.80;
        text-align: right;
        padding-top: 7px;
        white-space: nowrap;
    }}

    .top-rule {{
        border-bottom: 2px solid {YELLOW};
        margin: 0.75rem 0 1.25rem 0;
    }}

    .main-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(320px, 1fr);
        gap: 18px;
        align-items: start;
    }}

    .card {{
        border: 1px solid {LINE};
        border-radius: 11px;
        padding: 18px 20px;
        background: {CARD};
        margin-bottom: 14px;
    }}

    .upload-card {{
        border: 1px dashed #565656;
        border-radius: 11px;
        padding: 24px 20px;
        text-align: center;
        background: #030303;
        margin-bottom: 11px;
    }}

    .upload-icon {{
        font-size: 32px;
        color: {YELLOW};
        margin-bottom: 7px;
    }}

    .upload-title {{
        font-size: 16px;
        font-weight: 900;
        letter-spacing: 0.7px;
        text-transform: uppercase;
        color: {WHITE};
    }}

    .upload-subtitle {{
        font-size: 13px;
        opacity: 0.74;
        margin-top: 6px;
    }}

    .section-title {{
        font-size: 15px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 13px;
        color: {WHITE};
    }}

    .yellow {{ color: {YELLOW} !important; }}
    .grey {{ color: {GREY} !important; }}

    .big-number {{
        font-size: clamp(48px, 6.2vw, 82px);
        font-weight: 900;
        color: {YELLOW};
        line-height: 0.92;
        letter-spacing: -2px;
        margin-top: 13px;
        margin-bottom: 7px;
    }}

    .unit {{
        font-size: clamp(28px, 3.2vw, 42px);
        color: {WHITE};
        font-weight: 800;
    }}

    .small-label {{
        font-size: 11.5px;
        color: {WHITE};
        opacity: 0.76;
        text-transform: uppercase;
        letter-spacing: 0.75px;
        font-weight: 800;
    }}

    .image-frame {{
        border: 2px solid {WHITE};
        border-radius: 7px;
        padding: 0;
        overflow: hidden;
        margin-bottom: 8px;
    }}

    .meta-row {{
        display: flex;
        justify-content: space-between;
        gap: 18px;
        color: {WHITE};
        opacity: 0.82;
        font-size: 12px;
        margin-bottom: 11px;
    }}

    .data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }}

    .data-table th {{
        color: {WHITE};
        text-align: left;
        padding: 7px 8px;
        border-bottom: 1px solid {LINE};
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.45px;
        font-size: 10px;
    }}

    .data-table td {{
        padding: 6px 8px;
        border-bottom: 1px solid #202020;
        color: {WHITE};
    }}

    .data-table .value {{
        color: {YELLOW};
        font-weight: 900;
    }}

    .classification-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0;
        text-align: center;
    }}

    .class-box {{
        padding: 7px 16px 6px 16px;
    }}

    .class-box:first-child {{
        border-right: 1px solid {LINE};
    }}

    .class-label {{
        font-size: clamp(18px, 2.0vw, 24px);
        font-weight: 900;
        text-transform: uppercase;
        line-height: 1.05;
        margin-top: 14px;
    }}

    .class-note {{
        font-size: 10.5px;
        opacity: 0.60;
        margin-top: 9px;
    }}

    .bar-wrap {{
        margin-top: 27px;
        margin-bottom: 17px;
    }}

    .scale {{
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: {WHITE};
        margin-bottom: 9px;
        opacity: 0.9;
    }}

    .bar {{
        height: 7px;
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
        width: 21px;
        height: 21px;
        border-radius: 50%;
        background: {DARK};
        border: 5px solid {YELLOW};
    }}

    .legend-row {{
        display: flex;
        justify-content: space-between;
        margin-top: 16px;
        font-weight: 900;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.45px;
    }}

    .interpretation p {{
        font-size: 13px;
        line-height: 1.48;
        opacity: 0.92;
        margin-bottom: 0;
    }}

    .footer {{
        border-top: 1px solid {LINE};
        margin-top: 16px;
        padding-top: 22px;
        padding-bottom: 4px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: clamp(38px, 8vw, 92px);
    }}

    .footer-logo {{
        height: 52px;
        max-width: 210px;
        object-fit: contain;
        display: block;
    }}

    .footer-logo.lab {{
        height: 62px;
    }}

    .stFileUploader label {{
        color: {WHITE} !important;
    }}

    .stFileUploader section {{
        background-color: #050505 !important;
        border: 1px solid {LINE} !important;
        border-radius: 9px !important;
        min-height: 44px !important;
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

    .uploadedFile {{
        background-color: #050505 !important;
        border: 1px solid {LINE} !important;
    }}

    @media (max-width: 1050px) {{
        .main-grid {{
            grid-template-columns: 1fr;
        }}
        .brand-logo {{
            width: 260px;
            max-width: 70vw;
        }}
        .footer {{
            gap: 38px;
        }}
        .footer-logo {{
            height: 44px;
        }}
        .footer-logo.lab {{
            height: 54px;
        }}
    }}

    @media (max-width: 650px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        .topbar {{
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }}
        .opensource {{
            text-align: left;
            padding-top: 0;
        }}
        .upload-card {{
            padding: 20px 14px;
        }}
        .classification-grid {{
            grid-template-columns: 1fr;
        }}
        .class-box:first-child {{
            border-right: none;
            border-bottom: 1px solid {LINE};
        }}
        .meta-row {{
            flex-direction: column;
            gap: 3px;
        }}
        .footer {{
            flex-direction: column;
            gap: 20px;
        }}
        .footer-logo {{
            height: 42px;
        }}
        .footer-logo.lab {{
            height: 52px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# HEADER
# =====================================================

if LOGO_AI_LUX_B64:
    logo_html = f'<img class="brand-logo" src="{LOGO_AI_LUX_B64}" alt="AI LUX">'
else:
    logo_html = """
    <div>
        <div class="fallback-logo">AI LUX</div>
        <div class="fallback-subtitle">AI-powered nighttime illuminance estimation</div>
    </div>
    """

st.markdown(
    f"""
    <div class="topbar">
        <div>{logo_html}</div>
        <div class="opensource">Open-source AI tool</div>
    </div>
    <div class="top-rule"></div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# APP
# =====================================================

left_col, right_col = st.columns([1.55, 1], gap="large")

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
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="meta-row">
                <div>{html.escape(filename)}</div>
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

        df = pd.DataFrame(
            {
                "Parameter": VARS,
                "Value": [safe_round(v) for v in predicted_vals],
            }
        )

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
                <td>{html.escape(str(p1))}</td><td class="value">{html.escape(str(v1))}</td>
                <td>{html.escape(str(p2))}</td><td class="value">{html.escape(str(v2))}</td>
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

        st.markdown(
            f"""
            <div class="card">
                <div class="section-title"><span class="yellow">☼</span> Estimated Illuminance</div>
                <div class="big-number">{prediction:.2f} <span class="unit">lx</span></div>
                <div class="small-label">Horizontal illuminance</div>

                <div class="bar-wrap">
                    <div class="scale">
                        <span>0</span><span>5</span><span>10</span><span>20</span><span>50+</span>
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
    with right_col:
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


# =====================================================
# FOOTER
# =====================================================

def footer_logo(src: str | None, alt: str, extra_class: str = "") -> str:
    if not src:
        return f'<div class="small-label">{html.escape(alt)}</div>'
    return f'<img class="footer-logo {extra_class}" src="{src}" alt="{html.escape(alt)}">'


st.markdown(
    f"""
    <div class="footer">
        {footer_logo(LOGO_UCLOUVAIN_B64, "UCLouvain")}
        {footer_logo(LOGO_INNOVIRIS_B64, "innoviris.brussels")}
        {footer_logo(LOGO_LAB_B64, "LAB Research", "lab")}
    </div>
    """,
    unsafe_allow_html=True,
)
