import cv2 as cv
import face_recognition as fr
import tkinter as tk
from PIL import Image, ImageTk
import customtkinter as ctk
import os


class Recognizer:
    def __init__(self):
        self.recognizer_cc = cv.CascadeClassifier("filters/haarcascade_frontalface_default.xml")
        self.__owners_images = []
        self.load_owner_image()

    def load_owner_image(self):
        try:
            for filename in os.listdir("owners"):
                path = os.path.join("owners", filename)
                image_to_recognition = fr.load_image_file(path)
                self.__owners_images.append(fr.face_encodings(image_to_recognition)[0])
        except Exception as e:
            print(f"Error loading owner's image: {e}")

    def compare(self, recognize, img):

        if not self.__owners_images:
            return 3  # немає дозволених лиць

        if len(recognize) == 0:
            return 2  # Не знайдено лиця

        unknown_faces = fr.face_encodings(img)
        result = []

        for owner_face in self.__owners_images:
            result.append(fr.compare_faces(unknown_faces, owner_face, tolerance=0.6))

        if any(any(owner_result) for owner_result in result):
            return 1  # Доступ дозволено
        else:
            return 0  # Доступ не дозволено


class GUI:
    def __init__(self):
        self.__win = ctk.CTk()
        self.__win.geometry("850x650+300+20")
        self.__win.title("Підтвердження особи")
        self.__win.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start message
        self.__message_label = ctk.CTkLabel(self.__win,
                                            text="Привіт! \n"
                                                 "Щоб потрапити у приміщення, відскануйте обличчя. \n"
                                                 "Натисніть 'Старт' для початку.",
                                            font=("Times new Roman", 28))
        self.__message_label.pack()

        self.__check_button = ctk.CTkButton(self.__win,
                                            text="Старт",
                                            command=self.start_video,
                                            font=("Times New Roman", 20))
        self.__check_button.pack(pady=10)

        # Camera
        self.__camera = cv.VideoCapture(0)

        # Frame from camera
        self.__frame = None

        # Result of verification
        self.__label_var = ctk.StringVar()
        self.__label_var.set("")
        self.__result_label = ctk.CTkLabel(self.__win,
                                           textvariable=self.__label_var,
                                           font=("Times new Roman", 24))
        self.__result_label.pack()

        self.__recogniser = Recognizer()

        # Video from camera
        self.__web_cam = tk.Label(self.__win, text="")

        # Face detector
        self.__recognize = None

    def start_video(self):
        self.__message_label.configure(text="Коли будете готові, натисніть 'Перевірити'")
        self.__web_cam.pack()
        self.show_cam()
        self.__check_button.configure(command=self.verify,
                                      text="Перевірити")

    def show_cam(self):
        _, self.__frame = self.__camera.read()
        self.__frame = cv.cvtColor(self.__frame, cv.COLOR_BGR2RGB)
        self.__recognize = self.__recogniser.recognizer_cc.detectMultiScale(self.__frame,
                                                                            scaleFactor=1.1,
                                                                            minNeighbors=5)
        for (x, y, w, h) in self.__recognize:
            cv.rectangle(self.__frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        photo = ImageTk.PhotoImage(image=Image.fromarray(self.__frame))
        self.__web_cam.photo = photo
        self.__web_cam.configure(image=photo)
        self.__win.after(10, self.show_cam)

    def verify(self):
        result = self.__recogniser.compare(self.__recognize, self.__frame)
        match result:
            case 1:
                self.__label_var.set("Доступ дозволено")
            case 0:
                self.__label_var.set("Доступ не дозволено")
            case 2:
                self.__label_var.set("Не знайдено лиця!")
            case 3:
                self.__label_var.set("У папці owners відсутні фото!")

    def on_closing(self):
        self.__camera.release()
        self.__win.destroy()

    def run(self):
        self.__win.mainloop()


# Start
gui = GUI()
gui.run()
