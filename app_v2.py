import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import cv2

# -----------------------------
# MODEL + PARAMETERS
# -----------------------------

regr = pickle.load(open("LuxModel.sav", "rb"))

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

    datos.extend([
        blur_thr,
        blur_wa,
        blur_ta,
        blur_ba,
        blur_ha
    ])

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
        elif prediction < 15:
            return "PROPERLY LIT"
        else:
            return "OVERLIT"

    if area_type == "green":
        if prediction < 2:
            return "UNDERLIT"
        elif prediction < 10:
            return "PROPERLY LIT"
        else:
            return "OVERLIT"


def class_color(label):
    if label == "UNDERLIT":
        return GREY
    if label == "PROPERLY LIT":
        return WHITE
    if label == "OVERLIT":
        return YELLOW
    return WHITE


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI LUX",
    page_icon="💡",
    layout="wide"
)


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

    header {{
        visibility: hidden;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}

    h1, h2, h3, h4, h5, h6, p, span, label {{
        color: {WHITE} !important;
    }}

    .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2.2rem;
    }}

    .logo {{
        font-size: 44px;
        line-height: 0.9;
        font-weight: 900;
        color: {YELLOW};
        letter-spacing: -1px;
    }}

    .subtitle {{
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: {WHITE};
        text-transform: uppercase;
        margin-left: 18px;
        padding-top: 5px;
    }}

    .brand {{
        display: flex;
        align-items: flex-start;
    }}

    .nav {{
        font-size: 16px;
        font-weight: 800;
        color: {YELLOW};
        border-bottom: 2px solid {YELLOW};
        padding-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .opensource {{
        color: {WHITE};
        font-size: 14px;
        opacity: 0.85;
    }}

    .card {{
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 24px;
        background: #050505;
        margin-bottom: 18px;
    }}

    .upload-card {{
        border: 1px dashed #555555;
        border-radius: 12px;
        padding: 34px 24px;
        text-align: center;
        background: #030303;
        margin-bottom: 18px;
    }}

    .section-title {{
        font-size: 18px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 18px;
        color: {WHITE};
    }}

    .yellow {{
        color: {YELLOW} !important;
    }}

    .big-number {{
        font-size: 92px;
        font-weight: 900;
        color: {YELLOW};
        line-height: 1;
        letter-spacing: -2px;
        margin-top: 16px;
        margin-bottom: 8px;
    }}

    .unit {{
        font-size: 42px;
        color: {WHITE};
        font-weight: 800;
    }}

    .small-label {{
        font-size: 14px;
        color: {WHITE};
        opacity: 0.78;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
    }}

    .image-frame {{
        border: 3px solid {WHITE};
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 12px;
    }}

    .meta-row {{
        display: flex;
        justify-content: space-between;
        color: {WHITE};
        opacity: 0.8;
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
        padding: 10px;
        border-bottom: 1px solid #303030;
        font-weight: 700;
    }}

    .data-table td {{
        padding: 9px 10px;
        border-bottom: 1px solid #202020;
        color: {WHITE};
    }}

    .data-table .value {{
        color: {YELLOW};
        font-weight: 700;
    }}

    .classification-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0;
        text-align: center;
    }}

    .class-box {{
        padding: 18px;
    }}

    .class-box:first-child {{
        border-right: 1px solid #333333;
    }}

    .check {{
        width: 72px;
        height: 72px;
        border: 1.5px solid {WHITE};
        border-radius: 50%;
        margin: 18px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 38px;
        font-weight: 900;
        color: {WHITE};
    }}

    .class-label {{
        font-size: 22px;
        font-weight: 900;
        text-transform: uppercase;
    }}

    .bar-wrap {{
        margin-top: 38px;
        margin-bottom: 28px;
    }}

    .scale {{
        display: flex;
        justify-content: space-between;
        font-size: 15px;
        color: {WHITE};
        margin-bottom: 10px;
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
        font-weight: 800;
        text-transform: uppercase;
    }}

    .footer {{
        border-top: 1px solid #333333;
        margin-top: 26px;
        padding-top: 24px;
        display: grid;
        grid-template-columns: 1.2fr 1fr 1fr 1fr 1fr;
        gap: 24px;
        align-items: center;
    }}

    .footer-logo {{
        font-size: 36px;
        font-weight: 900;
        color: {YELLOW};
    }}

    .footer-text {{
        color: {WHITE};
        opacity: 0.75;
        font-size: 14px;
    }}

    .institution {{
        color: {WHITE};
        font-size: 24px;
        font-weight: 900;
    }}

    .stFileUploader {{
        color: {WHITE};
    }}

    .stFileUploader label {{
        color: {WHITE} !important;
    }}

    .stFileUploader section {{
        background-color: #050505 !important;
        border: 1px dashed #555555 !important;
        border-radius: 12px !important;
    }}

    .stFileUploader button {{
        background-color: {YELLOW} !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border-radius: 6px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# HEADER
# -----------------------------

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="logo">AI LUX</div>
            <div class="subtitle">AI-powered nighttime<br>illuminance estimation</div>
        </div>
        <div class="nav">Predict</div>
        <div class="opensource">Open-source AI tool</div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# APP
# -----------------------------

left, right = st.columns([1.55, 1])

with left:
    st.markdown(
        """
        <div class="upload-card">
            <div style="font-size:46px;color:#F1CF00;">☁</div>
            <div style="font-weight:900;text-transform:uppercase;">
                Drag & drop a 360° nighttime image here
            </div>
            <div style="opacity:0.75;margin-top:6px;">or click to browse</div>
            <div style="opacity:0.6;margin-top:14px;font-size:13px;">JPG, PNG</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    predicted_vals = data_from_image(uploaded_file)
    prediction = float(regr.predict([predicted_vals]))

    urban_class = classify(prediction, "urban")
    green_class = classify(prediction, "green")

    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")

    with left:
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        width, height = image.size
        filename = uploaded_file.name

        st.markdown(
            f"""
            <div class="meta-row">
                <div>📄 {filename}</div>
                <div>Resolution: {width} × {height} px</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title"><span class="yellow">◎</span> Extracted Data</div>',
            unsafe_allow_html=True
        )

        df = pd.DataFrame({
            "Parameter": VARS,
            "Value": [round(v, 5) if isinstance(v, float) else v for v in predicted_vals]
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
            unsafe_allow_html=True
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
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="section-title"><span class="yellow">◇</span> Classification</div>
                <div class="classification-grid">
                    <div class="class-box">
                        <div class="small-label">Urban Area (P1)</div>
                        <div class="check">✓</div>
                        <div class="class-label" style="color:{class_color(urban_class)};">{urban_class}</div>
                    </div>
                    <div class="class-box">
                        <div class="small-label">Green Area (P2)</div>
                        <div class="check">✓</div>
                        <div class="class-label" style="color:{class_color(green_class)};">{green_class}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="section-title"><span class="yellow">▣</span> Interpretation</div>
                <p>
                This location is estimated to provide approximately
                <span class="yellow"><b>{prediction:.2f} lx</b></span>
                of horizontal illuminance.
                </p>
                <p>
                According to AI LUX, this lighting level is considered
                <b style="color:{class_color(urban_class)};">{urban_class}</b>
                for urban public spaces and
                <b style="color:{class_color(green_class)};">{green_class}</b>
                for green areas.
                </p>
            </div>
            """,
            unsafe_allow_html=True
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
            unsafe_allow_html=True
        )


# -----------------------------
# FOOTER
# -----------------------------

st.markdown(
    """
    <div class="footer">
        <div>
            <div class="footer-logo">AI LUX</div>
            <div class="footer-text">AI-powered nighttime<br>illuminance estimation</div>
        </div>
        <div class="institution">UCLouvain</div>
        <div class="institution">innoviris<br>.brussels</div>
        <div class="institution">LAB<br><span style="font-size:14px;font-weight:400;">RESEARCH</span></div>
        <div class="footer-text">Version 1.0 · 2025<br>Contact: elena.agudosierra@uclouvain.be</div>
    </div>
    """,
    unsafe_allow_html=True
)
