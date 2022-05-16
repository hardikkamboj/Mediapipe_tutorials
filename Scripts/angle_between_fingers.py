import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

count = 0

list_joints = [[6,7,8], [5,6,7]]

def draw_finger_angles(image, results, joint_list):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [1280, 720]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image
            
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, ) as hands: 
    while cap.isOpened():

        count += 1 
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        # print(results)
        
        # Rendering results
        # if results.multi_hand_landmarks:
        #     for num, hand in enumerate(results.multi_hand_landmarks):
        #         mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
        #                                 mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        #                                 mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
        #                                  )
        
        # Rendering results part 2
        if results.multi_hand_landmarks:
            for hand, hand_result in zip(results.multi_hand_landmarks,results.multi_handedness ):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )

                label = hand_result.classification[0].label
                score = hand_result.classification[0].score

                coords = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                    [1280,720]).astype(int))

                cv2.putText(image, label, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                image = draw_finger_angles(image, results, list_joints)
            
        cv2.imshow('Hand Tracking', image)

        # if count%5==0 and results.multi_hand_landmarks:
        #     print(results.multi_handedness)
        #     print(len(results.multi_hand_landmarks))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# print(results.multi_hand_landmarks)
print(results.multi_handedness)
cap.release()
cv2.destroyAllWindows()
