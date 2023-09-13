import cv2
import time
from threading import Thread
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os

mail_sent = False
first_frame = None
status_list = [None, None]
times = []


def send_mail(timestamp, image_path):
    global mail_sent
    mail_sent = True
    img_data = open(image_path, "rb").read()
    image = MIMEImage(img_data, name=os.path.basename(image_path))

    message = MIMEText('Motion is detected.\tTimestamp: ' + timestamp, 'plain')
    msg = MIMEMultipart()
    msg['Subject'] = 'Motion detected!'
    msg.attach(message)
    msg.attach(image)
    smtpObj = smtplib.SMTP_SSL('', 465)
    smtpObj.login("", "")

    try:
        smtpObj.sendmail("", [""], msg.as_string())
        print("Successfully sent email")
    except smtplib.SMTPException as e:
        print("Error: unable to send email")
        print(e)
        mail_sent = False


print('Motion detector is starting. Please move out of the frame.')

for i in range(5, 0, -1):  # odbrojavanje
    print('Starting in', i)
    time.sleep(1)

video = cv2.VideoCapture(0)
in_frame_counter = 0

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (_, cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 5000:
            continue
        status = 1  # 1 ima pokreta, 0 nema pokreta

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    status_list.append(status)

    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 1:
        in_frame_counter += 1
        if in_frame_counter > 20:
            in_frame_counter = 0
            if not mail_sent:
                name = (time.strftime('%c') + '.png').replace('/', '-').replace(':', '-')
                cv2.imwrite(name, frame)
                timest = time.strftime('%c')
                Thread(target=send_mail, args=(timest, name)).start()

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(time.strftime('%c'))
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(time.strftime('%c'))
        mail_sent = False

    # cv2.imshow("Gray Frame", gray)
    # cv2.imshow("Delta Frame", delta_frame)
    # cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(time.strftime('%c'))
        break

for i in range(0, len(times), 2):
    with open('motion-log.txt', 'a') as f:
        f.write('Start: %s\tEnd: %s\n' % (times[i], times[i + 1]))

video.release()
cv2.destroyAllWindows()
