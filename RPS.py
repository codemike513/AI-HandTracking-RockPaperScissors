import cv2
import handTrackingModule as htm
import os
from random import choice


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    elif move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


wCam, hCam = 640, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "images"
imgList = os.listdir(folderPath)
# print(imgList)
overlayList = []
for imgPath in imgList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    # print(f'{folderPath}/{imgPath}')
    overlayList.append(image)

overlayList[0] = cv2.resize(overlayList[0], (200, 200))
overlayList[1] = cv2.resize(overlayList[1], (200, 200))
overlayList[2] = cv2.resize(overlayList[2], (200, 200))

detector = htm.handDetector()

tipIds = [4, 8, 12, 16, 20]

prev_move = None
user_move = ""

while True:
    success, img = cap.read()

    cv2.rectangle(img, (105, 0), (565, 50), (102, 255, 255), cv2.FILLED)
    cv2.putText(img, "ROCK PAPER SCISSORS", (115, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 69, 255), 4)
    cv2.line(img, (305, 55), (305, 380), (0, 0, 0), 2)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []

        # For Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # For Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if totalFingers == 5:
            user_move = "paper"
            cv2.rectangle(img, (15, 75), (280, 110),
                          (102, 255, 255), cv2.FILLED)
            cv2.putText(img, f'User Move: {user_move}',
                        (19, 100), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        elif totalFingers == 2:
            if lmList[tipIds[1]][2] < lmList[tipIds[1]-2][2] and lmList[tipIds[2]][2] < lmList[tipIds[2]-2][2]:
                user_move = "scissors"
                cv2.rectangle(img, (15, 75), (280, 110),
                              (102, 255, 255), cv2.FILLED)
                cv2.putText(img, f'User Move: {user_move}',
                            (19, 100), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                pass
        elif totalFingers == 0:
            user_move = "rock"
            cv2.rectangle(img, (15, 75), (280, 110),
                          (102, 255, 255), cv2.FILLED)
            cv2.putText(img, f'User Move: {user_move}',
                        (19, 100), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        if prev_move != user_move:
            computer_move = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move, computer_move)

        prev_move = user_move

        if computer_move == "paper":
            img[160:360, 365:565] = overlayList[0]
        elif computer_move == "scissors":
            img[160:360, 365:565] = overlayList[2]
        elif computer_move == "rock":
            img[160:360, 365:565] = overlayList[1]


        cv2.rectangle(img, (325, 75), (615, 110), (102, 255, 255), cv2.FILLED)
        cv2.putText(
            img, f'Computer Move: {computer_move}', (330, 100), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.rectangle(img, (180, 405), (540, 450), (102, 255, 255), cv2.FILLED)
        cv2.putText(img, f'Winner: {winner}',
                    (190, 440), font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Rock Paper Scissor", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
