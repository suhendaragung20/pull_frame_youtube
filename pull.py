import pafy
import cv2
import time
import masked_face_detection


url = "http://youtube.com/watch?v=aBda6AhYGiA&ab_channel=JapanWalking%5BMakimakiWalk%5D"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

capture = cv2.VideoCapture(best.url)

skip = 4
count = 0

mf = masked_face_detection.masked_face_detection()

while True:

    key = cv2.waitKey(1) & 0xFF
    grabbed, frame = capture.read()

    print(type(frame))

    if frame is None:
        continue
    
    count += 1

    if count <= skip:
        continue
    else:
        count = 0
        pass

    print(frame.shape[:2])

    frame, output_info = mf.detect(frame, show_result=True, target_shape=(260, 260))

    frame = mf.draw(frame, output_info)

    cv2.imshow("frame", frame)

    if key == ord("q"):
        # do a bit of cleanup
        cv2.destroyAllWindows()
        break