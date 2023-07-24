import cv2
import numpy as np

cap1 = cv2.VideoCapture('./dog.mp4')
cap2 = cv2.VideoCapture('./dog2.mp4')

frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
fps1 = cap1.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)

# 두 동영상의 크기를 동일하게 설정 (이 때, 가로와 세로의 크기가 같아야 함)
w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
w2 = round(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
h2 = round(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 크기가 다른 두 동영상을 동일한 크기로 resize
if w != w2 or h != h2:
    print("두 동영상의 크기가 다릅니다. 크기를 동일하게 맞춥니다.")
    cap2 = cv2.VideoCapture('./dog2.mp4')  # 동영상 다시 로드
    cap2 = cv2.resize(cap2, (w, h))  # 크기를 w x h로 변경

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('mix.avi', fourcc, fps1, (w, h))

# 1번 동영상 재생
for i in range(frame_cnt1):
    ret1, frame1 = cap1.read()

    if not ret1:
        break

    if i >= frame_cnt1 - int(fps1 * 2):
        # 2번 동영상을 좌측에서 시작하여 서서히 합성
        ret2, frame2 = cap2.read()
        if not ret2:
            break

        alpha = 1 - (frame_cnt1 - i) / (fps1 * 2)  # 2초 동안 서서히 나타날 수 있도록 alpha 조절
        frame_mixed = np.zeros((h, w, 3), dtype=np.uint8)
        frame_mixed[:, :int(w * alpha), :] = frame2[:, :int(w * alpha), :]
        frame_mixed[:, int(w * alpha):, :] = frame1[:, int(w * alpha):, :]

        cv2.imshow('output', frame_mixed)
        out.write(frame_mixed)
    else:
        cv2.imshow('output', frame1)
        out.write(frame1)

    if cv2.waitKey(30) == 27:  # 지연 시간을 30 밀리초로 설정
        break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
