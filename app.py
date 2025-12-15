import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import uuid

# rembg(배경제거) 관련 코드는 모두 삭제했습니다.
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("설치 필요: pip install streamlit-drawable_canvas")
    st.stop()

st.set_page_config(page_title="Scanner App Lite", layout="wide")
st.title("📱 스캔 어플처럼 면 지정하기 (Lite)")
st.markdown("""
**사용 방법:**
1. 왼쪽 도구바에서 **'다각형(Polygon)'** 아이콘을 선택하세요.
2. 제품의 **정면 모서리 4개**를 순서대로 클릭하세요.
3. **첫 번째 찍은 점을 다시 클릭**하면 도형이 닫히면서 면이 칠해집니다! 🟩
""")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_warped_image(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

uploaded_file = st.sidebar.file_uploader("사진 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if 'last_file_name' not in st.session_state or st.session_state.last_file_name != uploaded_file.name:
        st.session_state.last_file_name = uploaded_file.name
        st.session_state.canvas_key = str(uuid.uuid4())

    image_pil = Image.open(uploaded_file).convert("RGB") 
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 캔버스 미리보기용 리사이즈
    canvas_width = 700
    w_percent = (canvas_width / float(image_pil.size[0]))
    canvas_height = int((float(image_pil.size[1]) * float(w_percent)))
    resized_preview = image_pil.resize((canvas_width, canvas_height))
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("1. 면 그리기 (Polygon)")
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.4)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=resized_preview,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="polygon",
            key=st.session_state.canvas_key,
        )
        st.caption("↺ 되돌리기는 왼쪽 하단 아이콘")

    with col2:
        st.subheader("2. 결과 확인")
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0 and objects[0]["type"] == "path":
                path_data = objects[0]["path"]
                points = []
                for item in path_data:
                    if len(item) == 3: 
                        x = item[1] / w_percent
                        y = item[2] / w_percent
                        points.append([x, y])
                
                if len(points) > 4: points = points[:4]

                if len(points) == 4:
                    pts = np.array(points)
                    warped_bgr = get_warped_image(img_bgr, pts)
                    
                    st.write("👇 **비율 조정**")
                    aspect_ratio = st.slider("가로 비율", 0.5, 2.0, 1.0, 0.05)
                    h, w = warped_bgr.shape[:2]
                    new_w = int(w * aspect_ratio)
                    final_bgr = cv2.resize(warped_bgr, (new_w, h), interpolation=cv2.INTER_LANCZOS4)
                    
                    # 샤픈 필터
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    final_bgr = cv2.filter2D(final_bgr, -1, kernel)
                    
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_pil = Image.fromarray(final_rgb)
                    
                    st.image(final_pil, caption="변환 결과", use_column_width=True)
                    
                    # 단순 다운로드 버튼 (배경 제거 X)
                    buf = io.BytesIO()
                    final_pil.save(buf, format="PNG")
                    st.download_button("이미지 다운로드", buf.getvalue(), "scan_result.png", "image/png")
                else:
                    st.warning("⚠️ 사각형을 닫아주세요.")
            else:
                st.info("👈 왼쪽 사진 위에 4점을 찍어주세요.")