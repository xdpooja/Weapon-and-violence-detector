from tkinter import *
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
from firebase_admin import credentials, initialize_app, storage
cred = credentials.Certificate('guardianeye-741f2-firebase-adminsdk-95oi7-c935556371.json')
initialize_app(cred, {'storageBucket': 'guardianeye-741f2.appspot.com'})

model = YOLO('bestt.pt')
cap = cv2.VideoCapture(0)
width, height = 800, 600
detection_threshold = 0.3
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
app = Tk()
app.title("Code Clan")
app.iconbitmap('cctv.ico')
app.configure(background='white')
app.geometry('800x600')
label_widget = Label(app)
label_widget.grid(row = 0, column = 1, pady = 1)
running = False
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def start():
    global running
    running = True
def stop():
    global running
    running = False


def tracking():
    frame_count=0
    if running :
        _, frame = cap.read()
        tracking1 = model.track(frame, persist=True, classes=0)
        frame_ = tracking1[0].plot()
        opencv_image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        label_widget.photo_image = photo_image
        label_widget.configure(image=photo_image)
        label_widget.after(10, tracking)
        for result in tracking1:
            for r in result.boxes.data.tolist(): 
                x1, y1, x2, y2,id, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                id = int(id)
                print(r)
                for class_id in range(0,2):
                    if frame_count <5:
                        
                        person_filename = os.path.join(output_dir, f'person_{frame_count}.jpg')
                        frame_count +=1
                        cv2.imwrite(person_filename, frame_)
                        
                        break
                    break


button1 = Button(app, text="Start Tracking", command=lambda:[start(),tracking()])
button1.grid(row = 2, column = 1, pady = 2)

fileName = "output/person_0.jpg"
bucket = storage.bucket()
blob = bucket.blob(fileName)
blob.upload_from_filename(fileName)


print("your file url", blob.public_url)

app.mainloop()
