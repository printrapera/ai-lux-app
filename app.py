import base64
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# =====================================================
# AI LUX — one-page Streamlit app
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
    for p in [ASSETS_DIR / filename, Path(filename)]:
        if p.exists():
            return p
    return None


LOGO_AI_LUX = find_asset("logo_ai_lux.png")
LOGO_UCLOUVAIN = find_asset("logo_uclouvain_white.png")
LOGO_INNOVIRIS = find_asset("logo_innoviris_white.png")
LOGO_LAB = find_asset("logo_lab_white.png")


# -----------------------------
# Constants
# -----------------------------

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
# Model
# -----------------------------

@st.cache_resource
def load_model():
    with open("LuxModel.sav", "rb") as f:
        return pickle.load(f)


regr = load_model()


# -----------------------------
# Helpers
# -----------------------------

def img_to_base64(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def img_html(path: Path | None, cls: str, alt: str = "") -> str:
    if path is None or not path.exists():
        return ""
    ext = path.suffix.lower().replace(".", "")
    mime = "jpeg" if ext in ["jpg", "jpeg"] else "png"
    return f'<img class="{cls}" src="data:image/{mime};base64,{img_to_base64(path)}" alt="{alt}">'


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
    data.append(len(contours))

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
    return {
        "UNDERLIT": GREY,
        "PROPERLY LIT": WHITE,
        "OVERLIT": YELLOW,
    }.get(label, WHITE)

def lux_to_position(lux):
    """
    Convierte los lux a la posición del marcador sobre la barra.
    Escala visual:
        0 lx   -> 0%
        5 lx   -> 25%
        10 lx  -> 50%
        20 lx  -> 75%
        50 lx  -> 100%
    """

    points = [
        (0, 0),
        (5, 25),
        (10, 50),
        (20, 75),
        (50, 100),
    ]

    if lux <= 0:
        return 0

    if lux >= 50:
        return 100

    for (x1, p1), (x2, p2) in zip(points[:-1], points[1:]):
        if x1 <= lux <= x2:
            return p1 + (lux - x1) * (p2 - p1) / (x2 - x1)

    return 100


# -----------------------------
# CSS
# -----------------------------

st.markdown(f"""
<style>
:root {{
  --yellow:{YELLOW};
  --white:{WHITE};
  --grey:{GREY};
  --dark:{DARK};
  --card:{CARD};
  --line:{LINE};
}}

.stApp {{
  background: var(--dark);
  color: var(--white);
  font-family: Helvetica, Arial, sans-serif;
}}

header {{ visibility:hidden; }}

.block-container {{
  padding-top: 1.0rem;
  padding-bottom: 1.0rem;
  max-width: 1440px;
}}

h1,h2,h3,h4,h5,h6,p,span,label,div {{
  font-family: Helvetica, Arial, sans-serif;
  color: var(--white);
}}

.topbar {{
  display:grid;
  grid-template-columns: minmax(220px, 360px) 1fr auto;
  align-items:start;
  gap:24px;
}}

.brand-logo {{
  width: min(340px, 28vw);
  height:auto;
  display:block;
}}

.opensource {{
  font-size: clamp(10px, 0.95vw, 14px);
  opacity:0.88;
  text-align:right;
  padding-top:9px;
  white-space:nowrap;
}}

.top-rule {{
  border-bottom:2px solid var(--yellow);
  margin: 0.75rem 0 1.15rem 0;
}}

.main-grid {{
  display:grid;
  grid-template-columns: minmax(0, 1.55fr) minmax(330px, 1fr);
  gap:24px;
  align-items:start;
}}

.card {{
  border:1px solid var(--line);
  border-radius:12px;
  padding: clamp(14px, 1.35vw, 22px);
  background: var(--card);
  margin-bottom: 16px;
}}

.upload-card {{
  border:1px dashed #565656;
  border-radius:12px;
  padding: clamp(18px, 2.2vw, 32px) 22px;
  text-align:center;
  background:#030303;
  margin-bottom: 12px;
}}

.upload-icon {{
  font-size: clamp(30px, 3vw, 46px);
  color: var(--yellow);
  line-height:1;
  margin-bottom:10px;
}}

.upload-title {{
  font-size: clamp(12px, 1.25vw, 17px);
  font-weight:900;
  text-transform:uppercase;
  letter-spacing:0.6px;
}}

.upload-subtitle {{
  font-size: clamp(10px, 0.95vw, 14px);
  opacity:0.72;
  margin-top:7px;
}}

.section-title {{
  font-size: clamp(12px, 1.1vw, 17px);
  font-weight:900;
  text-transform:uppercase;
  letter-spacing:0.55px;
  margin-bottom:14px;
}}

.yellow {{ color: var(--yellow) !important; }}
.grey {{ color: var(--grey) !important; }}

.lux-number {{
  font-size: clamp(58px, 6.6vw, 96px);
  font-weight:900;
  color:var(--yellow);
  line-height:0.95;
  letter-spacing:-2px;
  margin: 8px 0;
}}

.unit {{
  font-size: clamp(28px, 3vw, 44px);
  color:var(--white);
  font-weight:800;
}}

.small-label {{
  font-size: clamp(10px, 0.85vw, 13px);
  opacity:0.76;
  text-transform:uppercase;
  letter-spacing:0.75px;
  font-weight:700;
}}

.image-frame {{
  border:2px solid var(--white);
  border-radius:8px;
  overflow:hidden;
  background:#000;
  margin-bottom:10px;
}}

.image-frame img {{
  display:block;
  width:100%;
  max-height: 305px;
  object-fit:cover;
}}

.meta-row {{
  display:flex;
  justify-content:space-between;
  gap:20px;
  opacity:0.78;
  font-size:clamp(10px, 0.85vw, 13px);
  margin-bottom:12px;
}}

.data-card {{ padding: clamp(13px, 1.1vw, 19px); }}

.data-table {{
  width:100%;
  border-collapse:collapse;
  font-size:clamp(9px, 0.85vw, 13px);
}}

.data-table th {{
  color:var(--white);
  text-align:left;
  padding:8px 10px;
  border-bottom:1px solid var(--line);
  font-weight:800;
  text-transform:uppercase;
  letter-spacing:0.5px;
  font-size:clamp(8px, 0.72vw, 11px);
}}

.data-table td {{
  padding:7px 10px;
  border-bottom:1px solid #202020;
}}

.data-table .value {{
  color:var(--yellow);
  font-weight:800;
}}

.lux-scale-wrap {{
  margin-top: clamp(20px, 2.4vw, 34px);
  margin-bottom: 18px;
}}

.lux-scale-labels {{
  display:flex;
  justify-content:space-between;
  font-size:clamp(11px, 0.95vw, 15px);
  margin-bottom:10px;
  opacity:0.9;
}}

.lux-scale-track {{
  height:8px;
  border-radius:8px;
  background: linear-gradient(to right, #7A7A7A 0%, #7A7A7A 25%, var(--white) 25%, var(--white) 60%, var(--yellow) 60%, var(--yellow) 100%);
  position:relative;
}}

.lux-scale-marker {{
  position:absolute;
  top:-8px;
  width:22px;
  height:22px;
  border-radius:50%;
  background:var(--dark);
  border:5px solid var(--yellow);
}}

.lux-scale-legend {{
  display:flex;
  justify-content:space-between;
  margin-top:18px;
  font-weight:900;
  text-transform:uppercase;
  font-size:clamp(10px, 0.85vw, 13px);
  letter-spacing:0.5px;
}}

.classification-grid {{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:0;
  text-align:center;
}}

.class-box {{
  padding:12px 16px 8px 16px;
}}

.class-box:first-child {{
  border-right:1px solid var(--line);
}}

.class-label {{
  font-size:clamp(18px, 1.95vw, 27px);
  font-weight:900;
  text-transform:uppercase;
  line-height:1.05;
  margin-top:16px;
}}

.class-note {{
  font-size:clamp(9px, 0.75vw, 12px);
  opacity:0.58;
  margin-top:10px;
}}

.interpretation p {{
  font-size:clamp(11px, 0.95vw, 15px);
  line-height:1.48;
  opacity:0.92;
  margin-bottom:0;
}}

.footer {{
  border-top:1px solid var(--line);
  margin-top:18px;
  padding-top:22px;
  padding-bottom:4px;
}}

.footer-logos {{
  display:flex;
  justify-content:center;
  align-items:center;
  gap: clamp(48px, 8vw, 110px);
  width:100%;
}}

.footer-logo {{
    max-height:48px;
    max-width:180px;
    object-fit:contain;
    display:block;
}}

.footer-logo.lab {{
    max-height:60px;
}}

.footer-logo.innoviris {{
    max-height:95px;
    max-width:360px;
}}

.footer-logo.uclouvain {{
    max-height:42px;
    max-width:180px;
}}

.fallback-logo {{
  font-size:42px;
  line-height:0.9;
  font-weight:900;
  color:var(--yellow);
  letter-spacing:-1px;
}}

/* Streamlit uploader */
.stFileUploader label {{
  color:var(--white) !important;
}}

.stFileUploader section {{
  background-color:#050505 !important;
  border:1px solid var(--line) !important;
  border-radius:8px !important;
  min-height:50px !important;
}}

.stFileUploader button {{
  background-color:var(--yellow) !important;
  color:#000000 !important;
  font-weight:900 !important;
  border-radius:6px !important;
  border:0 !important;
}}

.stExpander {{
  border:1px solid var(--line) !important;
  border-radius:8px !important;
  background-color:#050505 !important;
}}

[data-testid="stDataFrame"] {{
  background:#050505 !important;
}}

@media (max-width: 1050px) {{
  .block-container {{ padding: 1rem; }}
  .main-grid {{ grid-template-columns:1fr; }}
  .image-frame img {{ max-height:none; object-fit:contain; }}
  .footer-logos {{ gap:34px; flex-wrap:wrap; }}
  .footer-logo {{ max-height:42px; max-width:165px; }}
}}

@media (max-width: 700px) {{
  .topbar {{ grid-template-columns:1fr; row-gap:10px; }}
  .opensource {{ text-align:left; }}
  .classification-grid {{ grid-template-columns:1fr; }}
  .class-box:first-child {{ border-right:none; border-bottom:1px solid var(--line); }}
  .meta-row {{ flex-direction:column; gap:4px; }}
  .footer-logos {{ flex-direction:column; gap:18px; }}
}}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    f'<div class="topbar"><div>{img_html(LOGO_AI_LUX, "brand-logo", "AI LUX")}</div><div></div><div class="opensource">Open-source AI tool</div></div><div class="top-rule"></div>',
    unsafe_allow_html=True,
)


# -----------------------------
# Layout
# -----------------------------

left_col, right_col = st.columns([1.55, 1], gap="large")

with left_col:
    st.markdown(
        '<div class="upload-card"><div class="upload-icon">☁</div><div class="upload-title">Drag & drop a <span class="yellow">360°</span> nighttime image here</div><div class="upload-subtitle">or click to browse</div><div class="upload-subtitle" style="font-size:12px; opacity:0.55;">JPG · PNG · Max 200MB</div></div>',
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
    urban_color = class_color(urban_class)
    green_color = class_color(green_class)

    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    filename = uploaded_file.name

    with left_col:
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="meta-row"><div>{filename}</div><div>Resolution: {width} × {height} px</div></div>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame({
            "Parameter": VARS,
            "Value": [round(v, 5) if isinstance(v, float) else v for v in predicted_vals],
        })

        rows = ""
        for i in range(0, len(df), 2):
            p1, v1 = df.iloc[i]["Parameter"], df.iloc[i]["Value"]
            if i + 1 < len(df):
                p2, v2 = df.iloc[i + 1]["Parameter"], df.iloc[i + 1]["Value"]
            else:
                p2, v2 = "", ""

            rows += f"<tr><td>{p1}</td><td class='value'>{v1}</td><td>{p2}</td><td class='value'>{v2}</td></tr>"

        st.markdown(
            f'<div class="card data-card"><div class="section-title"><span class="yellow">◎</span> Extracted Data</div><table class="data-table"><tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>{rows}</table></div>',
            unsafe_allow_html=True,
        )

        with st.expander("Advanced details"):
            st.dataframe(df, use_container_width=True)

    with right_col:
        marker_position = lux_to_position(prediction)

        st.markdown(
            f'<div class="card"><div class="section-title"><span class="yellow">☼</span> Estimated Illuminance</div><div class="lux-number">{prediction:.2f} <span class="unit">lx</span></div><div class="small-label">Horizontal illuminance</div><div class="lux-scale-wrap"><div class="lux-scale-labels"><span>0</span><span>5</span><span>10</span><span>20</span><span>50+</span></div><div class="lux-scale-track"><div class="lux-scale-marker" style="left:calc({marker_position}% - 11px);"></div></div><div class="lux-scale-legend"><span style="color:{GREY};">Underlit</span><span style="color:{WHITE};">Properly lit</span><span style="color:{YELLOW};">Overlit</span></div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="card"><div class="section-title"><span class="yellow">◇</span> Classification</div><div class="classification-grid"><div class="class-box"><div class="small-label">Urban Area · P1</div><div class="class-label" style="color:{urban_color};">{urban_class}</div><div class="class-note">Non-green public space thresholds</div></div><div class="class-box"><div class="small-label">Green Area · P2</div><div class="class-label" style="color:{green_color};">{green_class}</div><div class="class-note">Biodiversity-sensitive thresholds</div></div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="card interpretation"><div class="section-title"><span class="yellow">▣</span> Interpretation</div><p>The estimated horizontal illuminance is <span class="yellow"><b>{prediction:.2f} lx</b></span>. According to AI LUX, this location is classified as <b style="color:{urban_color};">{urban_class}</b> for urban public spaces and <b style="color:{green_color};">{green_class}</b> for green areas.</p></div>',
            unsafe_allow_html=True,
        )

else:
    with right_col:
        st.markdown(
            '<div class="card"><div class="section-title"><span class="yellow">☼</span> Estimated Illuminance</div><div class="lux-number">-- <span class="unit">lx</span></div><div class="small-label">Upload a 360° nighttime image to start.</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card"><div class="section-title"><span class="yellow">◇</span> Classification</div><div class="classification-grid"><div class="class-box"><div class="small-label">Urban Area · P1</div><div class="class-label grey">--</div></div><div class="class-box"><div class="small-label">Green Area · P2</div><div class="class-label grey">--</div></div></div></div>',
            unsafe_allow_html=True,
        )


# -----------------------------
# Footer
# -----------------------------

st.markdown(
    f'<div class="footer"><div class="footer-logos">{img_html(LOGO_UCLOUVAIN, "footer-logo uclouvain", "UCLouvain")}{img_html(LOGO_INNOVIRIS, "footer-logo innoviris", "innoviris.brussels")}{img_html(LOGO_LAB, "footer-logo lab", "LAB Research")}</div></div>',
    unsafe_allow_html=True,
)
