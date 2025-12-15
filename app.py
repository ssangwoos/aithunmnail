import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 라이브러리 체크
try:
    from rembg import remove
except ImportError:
    pass
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("설치 필요: pip install streamlit-drawable_canvas")
    st.stop()

st.set_page_config(page_title="Scanner App Style", layout="wide")
st.title("📱 스캔 어플처럼 면 지정하기")
st.markdown("""
**사용 방법:**
1. 왼쪽 도구바에서 **'다각형(Polygon)'** 아이콘(별 모양이나 펜 모양)을 선택하세요.
2. 제품의 **정면 모서리 4개**를 순서대로 클릭하세요.
3. **첫 번째 찍은 점을 다시 클릭**하면 도형이 닫히면서 면이 칠해집니다! 🟩
""")

# === 1. 좌표 정렬 함수 ===
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

# === 2. 투시 변환 함수 ===
def get_warped_image(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped

# === 메인 화면 ===
uploaded_file = st.sidebar.file_uploader("사진 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 캔버스 리사이즈
    canvas_width = 700
    w_percent = (canvas_width / float(image_pil.size[0]))
    canvas_height = int((float(image_pil.size[1]) * float(w_percent)))
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("1. 면 그리기 (Polygon)")
        
        # [핵심 변경] drawing_mode를 'polygon'으로 설정
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.4)",  # 반투명 초록색 채우기
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=image_pil,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="polygon", # 다각형 모드!
            key="canvas",
        )
        st.caption("↺ 맘에 안 들면 왼쪽 하단 '되돌리기' 버튼을 누르세요.")

    with col2:
        st.subheader("2. 결과 확인")
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            
            # 다각형 데이터(path)가 있는지 확인
            if len(objects) > 0 and objects[0]["type"] == "path":
                # SVG 경로 데이터에서 좌표 추출하는 로직
                path_data = objects[0]["path"]
                points = []
                for item in path_data:
                    # ['M', x, y] 또는 ['L', x, y] 형태임 ('Z'는 닫기 명령)
                    if len(item) == 3: 
                        x = item[1] / w_percent
                        y = item[2] / w_percent
                        points.append([x, y])
                
                # 중복된 마지막 점(닫는 점) 제거 로직
                if len(points) > 4:
                    points = points[:4] # 앞의 4개만 사용

                if len(points) == 4:
                    pts = np.array(points)
                    warped_bgr = get_warped_image(img_bgr, pts)
                    
                    # 비율 보정 슬라이더
                    st.write("👇 **비율 조정 (뚱뚱함/홀쭉함)**")
                    aspect_ratio = st.slider("가로 비율", 0.5, 2.0, 1.0, 0.05)
                    
                    h, w = warped_bgr.shape[:2]
                    new_w = int(w * aspect_ratio)
                    final_bgr = cv2.resize(warped_bgr, (new_w, h), interpolation=cv2.INTER_LANCZOS4)
                    
                    # 샤픈 필터 자동 적용
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    final_bgr = cv2.filter2D(final_bgr, -1, kernel)
                    
                    # 출력
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_pil = Image.fromarray(final_rgb)
                    
                    st.image(final_pil, caption="변환 결과", use_column_width=True)
                    
                    # 배경 제거 버튼
                    if st.button("✂️ 배경 제거 및 다운로드"):
                        with st.spinner("마무리 작업 중..."):
                            try:
                                nobg = remove(final_pil)
                                buf = io.BytesIO()
                                nobg.save(buf, format="PNG")
                                st.download_button("PNG 다운로드", buf.getvalue(), "scan_result.png", "image/png")
                                st.success("완료!")
                            except:
                                st.error("배경 제거 실패")
                else:
                    st.warning("⚠️ 사각형이 아닙니다. 모서리 4개만 정확히 찍고 도형을 닫아주세요.")
            else:
                st.info("👈 왼쪽 사진 위에 마우스로 4점을 찍어 '초록색 면'을 만들어주세요.")