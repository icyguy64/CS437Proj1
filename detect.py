# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils

import picar_4wd as fc
import numpy as np

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  angle = 0
  distance = 0
  polarity = 1
  distances = np.zeros((10,10))
  stop_sign = False
  turn_left = False
  turn_right = False
  turn_forward = False
  double = False
  once = False
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    
    
    counter += 1
    image = cv2.flip(image, 1)

    # Run object detection estimation using the model.
    detections = detector.detect(image)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detections)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
              font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      fc.stop()
      break

    #cv2.imshow('object_detector', image)
    detections = detector.detect(image) 
    cv2.imshow('object_detector', image)
    for detection in detections:
      category = detection.categories[0]
      class_name = category.label
      probability = round(category.score, 2)
      if class_name == 'stop sign':
        stop_sign = True
      else:
        stop_sign = False
        
    if stop_sign:
      fc.stop()   
    else:
      if angle == 5 or angle == -5:
        fc.stop() 
      elif angle == 45:
        if once:
          double = True
        once = True
        
      elif angle == 0 and once and double:
        once = False
        double = False
        if turn_left:
          turn_left = False
          fc.turn_left(2)
          distances[0:10,0:10] = 0
        elif turn_right:
          turn_right = False
          fc.turn_right(2)
          distances[0:10,0:10] = 0
        elif turn_forward:
          turn_forward = False
          fc.forward(1)
          distances[0:10,0:10] = 0
      
    # servo angle
    # get distance
    distance = fc.get_distance_at(-1*angle)
    
    #if distance < 30 and distance > 0:
      #print("Angle is %f" % angle)
      #print("Distance is %f" % distance)
    x_coord = int(np.floor(5 + distance*np.sin(np.radians(angle))))
    y_coord = int(np.floor(distance*np.cos(np.radians(angle)))) + 1
    #print("x coordinate is %i" % x_coord)
    #print("y coordinate is %i" % y_coord)
    if y_coord > 0 and y_coord < 10 and x_coord > 0 and x_coord < 10 and distance < 50 and distance > 0:
      #print("xy coordinate is %i,%i" % (x_coord, y_coord))
      distances[y_coord,x_coord] = 1
      if y_coord+1 < 10 and x_coord+1 < 10:
        distances[y_coord+1,x_coord+1] = 1
      if y_coord+1 < 10 and x_coord-1 >= 0:
        distances[y_coord+1,x_coord-1] = 1
      if y_coord-1 >= 0 and x_coord+1 < 10:
        distances[y_coord-1,x_coord+1] = 1
      if y_coord-1 >= 0 and x_coord-1 >= 0:
        distances[y_coord-1,x_coord-1] = 1
      if y_coord+1 < 10:
        distances[y_coord+1,x_coord] = 1
      if y_coord-1 >= 0:
        distances[y_coord-1,x_coord] = 1
      if x_coord+1 < 10:
        distances[y_coord,x_coord+1] = 1
      if x_coord-1 >= 0:
        distances[y_coord,x_coord-1] = 1
    #print(distances)
    if angle == 45:
      polarity = 0
      #distances[0:10,5:9] = 0
    if angle == 0 :
      print(distances)
      # a star search
      #print(sum(map(sum, distances[0:10,0:5])))
      left_portion = sum(map(sum, distances[0:10,0:5])) 
      right_portion = sum(map(sum, distances[0:10,5:10]))
      if left_portion == 0 and right_portion == 0:
        if stop_sign == False:
          turn_forward = True
          turn_left = False
          turn_right = False
          goal = (5,9)
      elif left_portion < right_portion:
        goal = (0,9)
        if stop_sign == False and right_portion > 5:
          turn_left = True
          turn_right = False
          turn_forward = False
      elif left_portion > right_portion:
        goal = (9,9)
        if stop_sign == False and left_portion > 5:
          turn_right = True
          turn_left = False
          turn_forward = False
      elif left_portion + right_portion > 70:
        goal = (5,0)
        fc.stop()
      else: 
        goal = (5,9)
      start = (5,0)
      #print(goal)
      # list to keep track of visted nodes and initialize queue
      queue = []
      visited = []
      current_cost = heuristic(start[0], start[1] ,goal[0], goal[1])
      invalid_cost = current_cost
      queue.append(start)
      visited.append(start)
      while queue:
        s = queue.pop(0)
       
        if s[0] == goal[0] and s[1] == goal[1]:
          #print("done")
          break
        
        if s[1]+1 < 10 and distances[s[0],s[1]+1] == 0:
          frw = (s[0],s[1]+1)
          cost_frw = heuristic(goal[0],goal[1],frw[0],frw[1])
        else:
          #frw = s
          cost_frw = 999999 
        if s[0]-1 >= 0 and distances[s[0]-1,s[1]] == 0:
          lft = (s[0]-1,s[1])
          cost_lft = heuristic(goal[0],goal[1],lft[0],lft[1])
        else: 
          #lft = s
          cost_lft = 9999999 
        if s[0]+1 < 10 and distances[s[0]+1,s[1]] == 0:
          rght = (s[0]+1,s[1])
          cost_rght = heuristic(goal[0],goal[1],rght[0],rght[1]) 
        else:
          #rght = s
          cost_rght = 9999999
        cost_list = [cost_frw,cost_lft,cost_rght]
        min_cost = min(cost_list)
        min_index = cost_list.index(min_cost)
        #print("min")
        #print(cost_list)
        #print(min_cost)
        #print(invalid_cost)
        #print(visited)
        if min_index == 0:
            next = frw
            #fc.forward(1)
        if min_index == 1:
            next = lft
            #fc.turn_left(20)
        if min_index == 2:
            next == rght
            #fc.turn_right(20)
        #if min_index == ValueError:
            #break
        if next not in visited:
          visited.append(next)
          queue.append(next)
          current_cost = min_cost
        #print("yes")
      print(visited)
      
      
        
      
      # clears the detection 
      #distances[0:10,0:10] = 0
    if angle == -45:
      polarity = 1
      #distances[0:9,0:3] = 0
    
    if polarity == 1:
      angle = angle + 5
    else:
      angle = angle - 5
    #scan_list = fc.scan_step(35)
    #if not scan_list:
     # continue
    #if scan_list[3:7] != [2,2,2,2]:
      #turn_right = True
    
    
  cap.release()
  cv2.destroyAllWindows()

def heuristic(x1,y1,x2,y2) -> float:
  return abs(x1-x2) + abs(y1-y2)

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
