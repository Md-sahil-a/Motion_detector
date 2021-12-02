import cv2, pandas
from datetime import datetime, time, timezone

first_frame = None
status_list = [None,None]
time_up =[None]      

df = pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 100:
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        reci = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,155,0), 3)
    status_list.append(status)

    status_list=status_list[-2:]


    if status_list[-1] == 1 and status_list[-2] == 0:
        time_up.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        time_up.append(datetime.now())


    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key=cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            time_up.append(datetime.now())
        break

print(status_list)
print(time_up)

for i in range(0,len(time_up), 2):
    df=df.append({"Start":time_up[i], "End":time_up[i+1]}, ignore_index=True)

df.to_csv("Times.csv")


video.release()
cv2.destroyAllWindows