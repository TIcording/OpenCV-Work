import cv2
import numpy as np

def perspective_transform(image, pts):
    # 변환할 사각형의 크기
    width = 600
    height = 400

    # 변환될 사각형의 좌표
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    # 변환 행렬 계산
    M = cv2.getPerspectiveTransform(pts, dst)

    # 투시변환 적용
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

def mouse_callback(event, x, y, flags, param):
    # 마우스 콜백 함수: 사용자가 꼭지점을 클릭하여 사각형을 그림
    global pts, image_copy, selected_pt_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pt_idx = None
        for i, pt in enumerate(pts):
            if np.linalg.norm(np.array([x, y]) - np.array(pt)) < 10:  # 클릭한 좌표와 가까운 꼭지점 찾기
                selected_pt_idx = i
                break

    elif event == cv2.EVENT_LBUTTONUP:
        if selected_pt_idx is not None:
            # 선택한 꼭지점 좌표 수정
            pts[selected_pt_idx] = [x, y]
            selected_pt_idx = None

    # 이미지 업데이트
    image_copy = image.copy()
    for pt in pts:
        cv2.circle(image_copy, tuple(pt), 5, (255, 0, 0), -1)  # 빨간색 점으로 변경 (BGR 순서로 색상 지정)
    cv2.polylines(image_copy, [np.array(pts, dtype=np.int32)], True, (0, 0, 255), 2)  # 빨간색 선으로 변경
    cv2.imshow("Image", image_copy)

# 이미지 불러오기
image = cv2.imread("namecard.jpg")

# 이미지 복사하여 윈도우에 표시
image_copy = image.copy()
cv2.imshow("Image", image_copy)

pts = [[100, 50], [700, 50], [700, 800], [100, 800]]
selected_pt_idx = None

# 마우스 클릭 이벤트 콜백 설정
cv2.setMouseCallback("Image", mouse_callback)

print("Please adjust the corners of the rectangle.")
print("Press 's' key to perform perspective transformation.")

while True:
    key = cv2.waitKey(1)
    if key == ord('s'):  # 's' 키를 누르면 투시변환 수행
        warped = perspective_transform(image, np.array(pts, dtype="float32"))
        cv2.imshow("Warped Image", warped)
    elif key == 27:  # ESC 키를 누르면 종료
        break

cv2.destroyAllWindows()
