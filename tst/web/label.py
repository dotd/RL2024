import tkinter


class Label:

    def __init__(self):
        # GUI for saving models with Tkinter
        self.FONT = "Fixedsys 12 bold"  # GUI font
        self.save_command1 = 0
        self.save_command2 = 0
        self.save_command3 = 0
        self.load_command_1 = 0
        self.load_command_2 = 0
        self.load_command_3 = 0
        self.resume_command = 0
        self.stop_command = 0
        self.window = tkinter.Tk()
        self.window.lift()
        self.window.attributes("-topmost", True)
        self.window.title("DQN-Vision Manager")
        self.lbl = tkinter.Label(self.window, text="Manage training -->")
        self.lbl.grid(column=0, row=0)
        self.btn1 = tkinter.Button(self.window, text="Save 1", font=self.FONT, command=self.clicked1, bg="gray")
        self.btn1.grid(column=1, row=0)
        self.btn2 = tkinter.Button(self.window, text="Save 2", font=self.FONT, command=self.clicked2, bg="gray")
        self.btn2.grid(column=2, row=0)
        self.btn3 = tkinter.Button(self.window, text="Save Best", font=self.FONT, command=self.clicked3, bg="gray")
        self.btn3.grid(column=3, row=0)
        self.load_btn1 = tkinter.Button(self.window, text="Load 1", font=self.FONT, command=self.clicked_load1,
                                        bg="blue")
        self.load_btn1.grid(column=1, row=1)
        self.load_btn2 = tkinter.Button(self.window, text="Load 2", font=self.FONT, command=self.clicked_load2,
                                        bg="blue")
        self.load_btn2.grid(column=2, row=1)
        self.load_btn3 = tkinter.Button(self.window, text="Load Best", font=self.FONT, command=self.clicked_load3,
                                        bg="blue")
        self.load_btn3.grid(column=3, row=1)
        self.resume_btn = tkinter.Button(self.window, text="Resume Training", font=self.FONT,
                                         command=self.clicked_resume, bg="green")
        self.resume_btn.grid(column=1, row=2)
        self.stop_btn = tkinter.Button(self.window, text="Stop Training", font=self.FONT, command=self.clicked_stop,
                                       bg="red")
        self.stop_btn.grid(column=3, row=2)

    def clicked1(self):
        self.lbl.configure(text="Model saved in slot 1!")
        self.save_command1 = True

    def clicked2(self):
        self.lbl.configure(text="Model saved in slot 2!")
        self.save_command2 = True

    def clicked3(self):
        self.lbl.configure(text="Model saved in slot 3!")
        self.save_command3 = True

    def clicked_load1(self):
        self.lbl.configure(text="Model loaded from slot 1!")
        self.load_command_1 = True

    def clicked_load2(self):
        self.lbl.configure(text="Model loaded from slot 2!")
        self.load_command_2 = True

    def clicked_load3(self):
        self.lbl.configure(text="Model loaded from slot 3!")
        self.load_command_3 = True

    def clicked_resume(self):
        self.lbl.configure(text="Training resumed!")
        self.resume_command = True

    def clicked_stop(self):
        self.lbl.configure(text="Training stopped!")
        self.stop_command = True
