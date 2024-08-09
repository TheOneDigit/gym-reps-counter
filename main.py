import numpy as np 
import cv2
import mediapipe as mp
import time
import humanposemodule as hpm
from tqdm.notebook import tqdm



video_path = "/content/gym.mp4"


vid = cv2.VideoCapture(video_path)
previous_time = 0
current_time = 0
posedetect = hpm.poseDetector(detection_confident=True)
count = 0
direction = 0  # two directions 0 = up, 100 = down
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_video.avi', fourcc, 20.0, (1280, 720))
total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
pbar = tqdm(total=total_frames)


while True:
  ret, frame = vid.read()
  if not ret:
    print('break')
    break

  frame = cv2.resize(frame, (1280,720))
  frame = posedetect.findpose(frame, draw_landmark=False)
  lmlist = posedetect.findlocation(frame, draw_landmark=False)

  if len(lmlist) != 0: 
      print('True')

      angle = posedetect.findangle(frame, 12, 14, 16)
      per = np.interp(angle, (180, 340), (0,100))
      bar = np.interp(angle, (180, 340), (650,100))
      
      color = (0,255,255)
      
      # count number of time 
      if per == 100:
          color = (0,0,255)
          if direction == 0:
              count += 0.5
              direction = 1
      
      if per == 0:
          color = (0,0,255)
          if direction == 1:
              count += 0.5
              direction = 0
      
      # Bar
      cv2.rectangle(frame, (1100,100), (1175,650), color, 3)
      cv2.rectangle(frame, (1100,int(bar)), (1175,650), color, cv2.FILLED)
      cv2.putText(frame, str(int(per))+"%", (1100,75), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
      
      # Count
      cv2.rectangle(frame, (0,450), (250,720), (0,255,255), cv2.FILLED)
      cv2.putText(frame, str(int(count)), (60,600), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 8)
      
      
  current_time = time.time()
  fps = 1 / (current_time - previous_time)
  previous_time = current_time

  cv2.putText(frame, "frame rate: "+str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 5)
  out.write(frame)
  pbar.update(1)  

vid.release()
cv2.destroyAllWindows()
