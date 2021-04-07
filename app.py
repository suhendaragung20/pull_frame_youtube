from flask import Flask, render_template, Response
import cv2
import time
# import camera
app = Flask(__name__)

# cam = camera.camera('rtsp://admin:QPPZFE@192.168.100.57:554/H.264_stream')
# cam = cv2.VideoCapture('rtsp://admin:QPPZFE@192.168.100.57:554/H.264_stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)


import numpy as np
import pafy
import cv2
import time
import masked_face_detection


url = "http://youtube.com/watch?v=aBda6AhYGiA&ab_channel=JapanWalking%5BMakimakiWalk%5D"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

capture = cv2.VideoCapture(best.url)



mf = masked_face_detection.masked_face_detection()


def skipFrames(timegap, FPS, cap, CALIBRATION):
   latest = None
   while True :  
      for i in range(int(timegap*FPS/CALIBRATION)) :
        _,latest = cap.read()
        if(not _):
           time.sleep(0.5)#refreshing time
           break
      else:
        break
   return latest


def skipFrames(timegap, FPS, cap, CALIBRATION):
    latest = None
    ret = None
    while True :  
        for i in range(int(timegap*FPS/CALIBRATION)) :
            ret,latest = cap.read()
            if(not ret):
                time.sleep(0.5)#refreshing time
                break
        else:
            break
    return latest, ret


def gen_frames():  # generate frame by frame from camera
    # FPS = 60
    # CALIBRATION = 1.5
    # gap = 0.1
    frame = None
    skip = 4
    count = 0
    while True:
        # print("================================")

        grabbed, frame = capture.read()

        # print(type(frame))

        if frame is None:
            continue
        
        count += 1

        if count <= skip:
            continue
        else:
            count = 0
            pass

        if frame is None:
            continue

        '''
            crop_1   crop_2    crop_3    crop_4
            crop_5   crop_6    crop_7    crop_8
            crop_9   crop_10   crop_11   crop_12
        '''

        h, w = frame.shape[:2]

        x_1 = int(w*(1/4))
        x_2 = int(w*(2/4))
        x_3 = int(w*(3/4))

        y_1 = int(h*(1/3))
        y_2 = int(h*(2/3))

        crop_img_1 = frame[0:y_1, 0:x_1]
        crop_img_2 = frame[0:y_1, x_1+1:x_2]
        crop_img_3 = frame[0:y_1, x_2+1:x_3]
        crop_img_4 = frame[0:y_1, x_3+1:w]

        crop_img_5 = frame[y_1+1:y_2, 0:x_1]
        crop_img_6 = frame[y_1+1:y_2, x_1+1:x_2]
        crop_img_7 = frame[y_1+1:y_2, x_2+1:x_3]
        crop_img_8 = frame[y_1+1:y_2, x_3+1:w]

        crop_img_9 = frame[y_2+1:h, 0:x_1]
        crop_img_10 = frame[y_2+1:h, x_1+1:x_2]
        crop_img_11 = frame[y_2+1:h, x_2+1:x_3]
        crop_img_12 = frame[y_2+1:h, x_3+1:w]

        # ===========================

        crop_img_1, output_info = mf.detect(crop_img_1, show_result=True, target_shape=(260, 260))
        crop_img_1 = mf.draw(crop_img_1, output_info)

        crop_img_2, output_info = mf.detect(crop_img_2, show_result=True, target_shape=(260, 260))
        crop_img_2 = mf.draw(crop_img_2, output_info)

        crop_img_3, output_info = mf.detect(crop_img_3, show_result=True, target_shape=(260, 260))
        crop_img_3 = mf.draw(crop_img_3, output_info)

        crop_img_4, output_info = mf.detect(crop_img_4, show_result=True, target_shape=(260, 260))
        crop_img_4 = mf.draw(crop_img_4, output_info)

        # ===========================

        crop_img_5, output_info = mf.detect(crop_img_5, show_result=True, target_shape=(260, 260))
        crop_img_5 = mf.draw(crop_img_5, output_info)

        crop_img_6, output_info = mf.detect(crop_img_6, show_result=True, target_shape=(260, 260))
        crop_img_6 = mf.draw(crop_img_6, output_info)

        crop_img_7, output_info = mf.detect(crop_img_7, show_result=True, target_shape=(260, 260))
        crop_img_7 = mf.draw(crop_img_7, output_info)

        crop_img_8, output_info = mf.detect(crop_img_8, show_result=True, target_shape=(260, 260))
        crop_img_8 = mf.draw(crop_img_8, output_info)

        # ===========================

        crop_img_9, output_info = mf.detect(crop_img_9, show_result=True, target_shape=(260, 260))
        crop_img_9 = mf.draw(crop_img_9, output_info)

        crop_img_10, output_info = mf.detect(crop_img_10, show_result=True, target_shape=(260, 260))
        crop_img_10 = mf.draw(crop_img_10, output_info)

        crop_img_11, output_info = mf.detect(crop_img_11, show_result=True, target_shape=(260, 260))
        crop_img_11 = mf.draw(crop_img_11, output_info)

        crop_img_12, output_info = mf.detect(crop_img_12, show_result=True, target_shape=(260, 260))
        crop_img_12 = mf.draw(crop_img_12, output_info)

        '''
            combine_1    combine_2    combine_3    combine_4
        '''

        # vertical ======================
        combine_1 = np.concatenate((crop_img_1, crop_img_5), axis=0)
        combine_1 = np.concatenate((combine_1, crop_img_9), axis=0)

        combine_2 = np.concatenate((crop_img_2, crop_img_6), axis=0)
        combine_2 = np.concatenate((combine_2, crop_img_10), axis=0)

        combine_3 = np.concatenate((crop_img_3, crop_img_7), axis=0)
        combine_3 = np.concatenate((combine_3, crop_img_11), axis=0)

        combine_4 = np.concatenate((crop_img_4, crop_img_8), axis=0)
        combine_4 = np.concatenate((combine_4, crop_img_12), axis=0)
        # ===========================

        # horizontal ======================
        frame_result = np.concatenate((combine_1, combine_2), axis=1)
        frame_result = np.concatenate((frame_result, combine_3), axis=1)
        frame_result = np.concatenate((frame_result, combine_4), axis=1)
        # ===========================


        tic = time.time()
        ret, buffer = cv2.imencode('.jpg', frame_result)
        frame_result = buffer.tobytes()

        # gap = time.time()-s
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_result + b'\r\n')  # concat frame one by one and show result

def gen_frames_bird():
    frame = None
    while True:
        frame = bb.get()

        if frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result





@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_2')
def video_feed_2():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='localhost', port=5001, debug=True)
