import cv2
import mediapipe as mp
import numpy as np
# from debug_funs import draw_points

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

left_counter = 0 
left_stage = None

right_counter = 0 
right_stage = None



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# VIDEO FEED
cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = cv2.flip(image, 1)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # image = cv2.flip(image,1)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        try:
            landmarks = results.pose_landmarks.landmark

            # finding landmarks for left arm 
            left_shoulder = (np.multiply( (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y), (1280,720))).astype(np.int)

            left_elbow = (np.multiply( (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y), (1280,720))).astype(np.int)

            left_wrist = (np.multiply( (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y), (1280,720))).astype(np.int)

            # finding landmarks for right arm 
            right_shoulder = (np.multiply( (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y), (1280,720))).astype(np.int)

            right_elbow = (np.multiply( (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y), (1280,720))).astype(np.int)
            right_wrist = (np.multiply( (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y), (1280,720))).astype(np.int)


            # calculating angle 
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)

            # showing the angle on the image 
            cv2.putText(image, str(left_angle), left_elbow,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, str(right_angle), right_elbow,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # ##### DEBUG Funs 
            # image = draw_points(image, [right_shoulder, right_wrist, right_elbow], 
            #                     ["right_shoulder","right_wrist","right elbow"])

            # image = draw_points(image, [left_shoulder, left_wrist, left_elbow], 
            #                     ["left_shoulder","left wrist","left elbow"])


            # calculating reps for left arm 
            if left_angle > 160:
                left_stage = "down"
            if left_angle < 30 and left_stage =='down':
                left_stage="up"
                left_counter +=1
                print("Left reps - ",left_counter)

            # calculating reps for right arm 
            if right_angle > 160:
                right_stage = "down"
            if right_angle < 30 and right_stage =='down':
                right_stage="up"
                right_counter +=1
                print("RIGHT reps - ",right_counter)

            # Scoreboard 

            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            cv2.rectangle(image, (1050,0), (1280,73), (245,117,16), -1)
        
            # Scoreboard for RIGHT
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, left_stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            ## Scoreboard for RIGHT 
            cv2.putText(image, 'REPS', (1050,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_counter), 
                        (1050,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data for right arm 
            cv2.putText(image, 'STAGE', (1100,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, right_stage, 
                        (1100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        except: 
            pass 

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
