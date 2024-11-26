import os
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util
from test import test


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")
        self.main_window.title("Face Attendance System")

        # Buttons
        self.login_button_main_window = util.get_button(self.main_window, 'Login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'Logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'Register New User', 'gray', self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=750, y=400)

        # Webcam label
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        # Webcam Initialization
        self.add_webcam(self.webcam_label)

        # Directory and Log File Setup
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        self.log_path = './log.txt'

    def add_webcam(self, label):
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Unable to access the webcam.")
            return

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        # Continuously capture frames from the webcam
        ret, frame = self.cap.read()

        if not ret or frame is None:
            print("Error: Failed to capture frame.")
            return

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        # Schedule next frame update
        self._label.after(20, self.process_webcam)

    def login(self):
        # Perform anti-spoofing check
        label = test(
            image=self.most_recent_capture_arr,
            model_dir=r"E:\antispoofing-project\pythonProject\face-attendance-system\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
            device_id=0
        )

        if label == 1:
            # Recognize face
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Error', 'Unknown user. Please register or try again.')
            else:
                util.msg_box('Welcome Back!', f'Welcome, {name}!')
                with open(self.log_path, 'a') as f:
                    f.write(f'{name},{datetime.datetime.now()},in\n')

        else:
            util.msg_box('Warning', 'Spoofing attempt detected!')

    def logout(self):
        # Perform anti-spoofing check
        label = test(
            image=self.most_recent_capture_arr,
            model_dir=r"E:\antispoofing-project\pythonProject\face-attendance-system\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
            device_id=0
        )

        if label == 1:
            # Recognize face
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Error', 'Unknown user. Please register or try again.')
            else:
                util.msg_box('Goodbye!', f'Goodbye, {name}!')
                with open(self.log_path, 'a') as f:
                    f.write(f'{name},{datetime.datetime.now()},out\n')

        else:
            util.msg_box('Warning', 'Spoofing attempt detected!')

    def register_new_user(self):
        # Create a new window for registering a user
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")
        self.register_new_user_window.title("Register New User")

        # Accept Button
        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user
        )
        self.accept_button_register_new_user_window.place(x=750, y=300)

        # Try Again Button
        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try Again', 'red', self.try_again_register_new_user
        )
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        # Image Display Label
        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        # Entry Text for Username
        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Please, \ninput username:'
        )
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        # Add captured image to label
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        # Validate and register the new user
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()

        if not name:
            util.msg_box('Error', 'Please enter a username.')
            return

        face_encodings = face_recognition.face_encodings(self.register_new_user_capture)

        if not face_encodings:
            util.msg_box('Error', 'No face detected. Please try again.')
            return

        embeddings = face_encodings[0]

        file_path = os.path.join(self.db_dir, f'{name}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User registered successfully!')
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
