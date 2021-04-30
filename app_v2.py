from flask import Flask, render_template, Response
import cv2
import time
# import camera
app = Flask(__name__)


import argparse
import sys

import numpy as np
import masked_face_detection
import fresh_frame



# parse the command line
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
	help="path to input image")
ap.add_argument("-v", "--level", type=int, default=1,
	help="level detail deteksi")
args = vars(ap.parse_args())

# input stream
source = 0
if args["source"] == "0":
    source = 0
elif args["source"] == "1":
    source = 1
else:
    source = args["source"]


vs = cv2.VideoCapture(source)
vs.set(cv2.CAP_PROP_FPS, 30)

level_fragment = args["level"]

# a way to watch the camera unthrottled
def callback(img):
    aa = 1

fresh = fresh_frame.FreshestFrame(vs)
fresh.callback = callback


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
    frame = None
    cnt = 0
    while True:
        print("================================")
        toc = time.time()
        tic = time.time()

        # GET FRESH ============
        t0 = time.perf_counter()
        cnt,frame = fresh.read(seqnumber=cnt+1)
        dt = time.perf_counter() - t0
        if dt > 0.010: # 10 milliseconds
            print("NOTICE: read() took {dt:.3f} secs".format(dt=dt))

        # let's pretend we need some time to process this frame
        print("processing {cnt}...".format(cnt=cnt), end=" ", flush=True)
        # ==================
        
        if level_fragment == 1:
            output_frame = detect_level_1(frame)
        elif level_fragment == 2:
            output_frame = detect_level_2(frame)
        
        print("detection time", time.time() - tic)
        print("resolution", output_frame.shape[:2])

        if frame is None:
            continue
        tic = time.time()
        ret, buffer = cv2.imencode('.jpg', output_frame)
        output_frame = buffer.tobytes()
        print("add to buffer", time.time() - tic)
        print("total time", time.time() - toc)
        # gap = time.time()-s
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')  # concat frame one by one and show result


def convert_box_level_2(output_info, idx_segmen, ratio, ori_size):
    new_info = []
    (H, W) = ori_size
    (ratio_h, ratio_w) = ratio
    for info in output_info:
        (idx_mask, conf, startX, startY, endX, endY) = info
        new_startX = int(startX * ratio_w)
        new_startY = int(startY * ratio_h)
        new_endX = int(endX * ratio_w)
        new_endY = int(endY * ratio_h)
        
        if idx_segmen == 2:
            new_startX = new_startX + int(W/2)
            new_endX = new_endX + int(W/2)
        elif idx_segmen == 3:
            new_startY = new_startY + int(H/2)
            new_endY = new_endY + int(H/2)
        elif idx_segmen == 4:
            new_startX = new_startX + int(W/2)
            new_endX = new_endX + int(W/2) 
            new_startY = new_startY + int(H/2)
            new_endY = new_endY + int(H/2)
    
        new_info.append((idx_mask, conf, new_startX, new_startY, new_endX, new_endY))
    
    return new_info
        
        
def detect_level_1(frame):
    ori_h, ori_w = frame.shape[:2]
    ori_frame = frame.copy()
    ratio_w = ori_w / 260
    ratio_h = ori_h / 260
    
    frame = cv2.resize(frame, (260, 260))
    
    crop_img_1, output_info = mf.detect(frame, show_result=True, target_shape=(260, 260))
    output_info = convert_box_level_2(output_info, 1, (ratio_h, ratio_w), (ori_h,ori_w))
    ori_frame = mf.draw(ori_frame, output_info)
    
    return ori_frame
    
        

def detect_level_2(frame):
    '''
        crop_1   crop_2
        crop_3   crop_4
    '''

    ori_h, ori_w = frame.shape[:2]
    ori_frame = frame.copy()
    
    ratio_w = ori_w / 520
    ratio_h = ori_h / 520
    
    frame = cv2.resize(frame, (520, 520))
    
    h, w = frame.shape[:2]

    x_center = int(w/2)
    y_center = int(h/2)

    crop_img_1 = frame[0:y_center, 0:x_center]
    crop_img_2 = frame[0:y_center, x_center+1:w]
    crop_img_3 = frame[y_center+1:h, 0:x_center]
    crop_img_4 = frame[y_center+1:h, x_center+1:w]
        
    crop_img_1, output_info = mf.detect(crop_img_1, show_result=True, target_shape=(260, 260))
    output_info = convert_box_level_2(output_info, 1, (ratio_h, ratio_w), (ori_h,ori_w))
    ori_frame = mf.draw(ori_frame, output_info)
    
    crop_img_2, output_info = mf.detect(crop_img_2, show_result=True, target_shape=(260, 260))
    output_info = convert_box_level_2(output_info, 2, (ratio_h, ratio_w), (ori_h,ori_w))
    ori_frame = mf.draw(ori_frame, output_info)

    crop_img_3, output_info = mf.detect(crop_img_3, show_result=True, target_shape=(260, 260))
    output_info = convert_box_level_2(output_info, 3, (ratio_h, ratio_w), (ori_h,ori_w))
    ori_frame = mf.draw(ori_frame, output_info)

    crop_img_4, output_info = mf.detect(crop_img_4, show_result=True, target_shape=(260, 260))
    output_info = convert_box_level_2(output_info, 4, (ratio_h, ratio_w), (ori_h,ori_w))
    ori_frame = mf.draw(ori_frame, output_info)  
        
    return ori_frame


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
