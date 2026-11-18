import customtkinter as ctk
from gui import ScannerApp

def main():
    ctk.set_appearance_mode("system")
    app = ScannerApp()
    app.mainloop()

if __name__ == "__main__":
    main()