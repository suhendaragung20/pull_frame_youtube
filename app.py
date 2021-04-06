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
        print("================================")

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

        frame, output_info = mf.detect(frame, show_result=True, target_shape=(260, 260))

        frame = mf.draw(frame, output_info)


        if frame is None:
            continue
        tic = time.time()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # gap = time.time()-s
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

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
    return Response(gen_frames_bird(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='192.168.100.119', port=5001, debug=True)
