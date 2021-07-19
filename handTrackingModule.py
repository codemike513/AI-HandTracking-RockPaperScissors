import cv2
import mediapipe as mp
import time

class handDetector():
    # static mode, maximum hands, detection Confidence, track Confidence
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_in_frame in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_in_frame, self.mpHands.HAND_CONNECTIONS)
                    # self.mpDraw.draw_landmarks(img, hand_in_frame, self.mpHands.HAND_CONNECTIONS, self.mpDraw.DrawingSpec((255, 0, 0), 8, 3), self.mpDraw.DrawingSpec((0, 0, 255), 2, 4))

        return img  
    
    def findPosition(self, img, handNo = 0, draw = True):

        lmList = []

        if self.results.multi_hand_landmarks: 
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # height width and channels of image
                h ,w ,c = img.shape
                # Positions
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return lmList    

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while(cap.isOpened()):
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()