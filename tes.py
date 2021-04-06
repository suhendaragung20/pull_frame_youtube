import masked_face_detection
import cv2




frame = cv2.imread("6.jpg")

mf = masked_face_detection.masked_face_detection()

frame, output_info = mf.detect(frame, show_result=True, target_shape=(260, 260))

frame = mf.draw(frame, output_info)

cv2.imwrite("result.png", frame)