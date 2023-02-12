import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import Label, Button, Text
from PIL import Image, ImageTk
#from tkinter import *
# Load the pre-trained model
model = tf.keras.models.load_model('waste_segregation_model.h5')



# Initialize the video capture
cap = cv2.VideoCapture(0) # 0 indicates that we want to use the default camera

class WasteSegregationApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("Waste Segregation")

        self.label = Label(self, text="")
        self.label.grid(row=500,column=500)

        self.update_frame()

    def update_frame(self):
        ret, frame = cap.read()

        # Preprocess the frame
        frame = cv2.resize(frame, (150, 150)) # resize the frame to 150x150
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("uint8")

        # Use the pre-trained model to make a prediction
        preprocessed_frame = np.expand_dims(frame, axis=0)
        prediction = model.predict(preprocessed_frame)

        # Get the class with the highest predicted probability
        predicted_class = np.argmax(prediction)
        prediction_array=["Cardboard waste", "glass waste", "Metal waste","paper waste","plastic waste","trash waste"]

        # Display the prediction on the frame
        frame = cv2.resize(frame, (500, 500))
        cv2.putText(frame, f"Prediction: {prediction_array[predicted_class]}", (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        # Convert the OpenCV image to a PIL image
        image = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image)

        self.label.configure(image=image)
        self.label.image = image

        self.after(1, self.update_frame)

app = WasteSegregationApp()
app.geometry("900x900")
app['background']='#CDCDC8'
heading=tk.Label(app, text="Akalka", justify="center", relief="groove", font=("Times New Roman",44), borderwidth=10).grid(row=400,column=500)

app.mainloop()

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
