import threading
import cv2
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from ultralytics import YOLO

# Set the window size for the Kivy app
Window.size = (800, 600)

class MyYOLOApp(App):
    def build(self):
        self.model = YOLO('./myset/best.pt')
        
        # Main layout for the app
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Title Label
        title = Label(
            text="Automated Data Labelling for Health Care using Deep Learning",
            font_size=24,
            size_hint=(1, 0.1),
            halign="center",
            valign="middle"
        )
        main_layout.add_widget(title)

        # Button to start webcam detection
        start_button = Button(
            text="START HEALTH CARE LABELLING FROM WEBCAM",
            size_hint=(1, 0.1),
            font_size=18,
            on_press=self.run_tracker_from_webcam
        )
        main_layout.add_widget(start_button)

        # Button to select a file for detection
        select_file_button = Button(
            text="SELECT HEALTH CARE LABELLING FROM FILE",
            size_hint=(1, 0.1),
            font_size=18,
            on_press=self.open_file_chooser
        )
        main_layout.add_widget(select_file_button)

        # Label for Author's name
        author_label = Label(
            text="Author: Ilo Chinonyelum",
            font_size=20,
            size_hint=(1, 0.1),
            halign="center"
        )
        main_layout.add_widget(author_label)

        return main_layout

    def open_file_chooser(self, instance):
        """Opens a file chooser dialog to select an image."""
        file_chooser = FileChooserIconView()
        popup_layout = BoxLayout(orientation='vertical')
        popup_layout.add_widget(file_chooser)

        # Button to confirm file selection
        select_button = Button(text="Select", size_hint=(1, 0.1))
        popup_layout.add_widget(select_button)

        # Create a popup with the file chooser
        popup = Popup(
            title='Select a File',
            content=popup_layout,
            size_hint=(0.9, 0.9)
        )

        select_button.bind(on_press=lambda x: self.run_file_detection(file_chooser.path, popup))
        popup.open()

    def run_file_detection(self, selected_file, popup):
        """Run detection on the selected file."""
        popup.dismiss()
        path = selected_file

        # Run YOLO detection on the selected file
        results = self.model([path], save=True, show=True, conf=0.55, iou=0.4)
        
        for result in results:
            print(result.path)  # Display results path

    def run_tracker_from_webcam(self, instance):
        """Start tracking from the webcam."""
        tracker_thread = threading.Thread(target=self.run_tracker_in_thread, args=(0, self.model), daemon=True)
        tracker_thread.start()

    def run_tracker_in_thread(self, filename, model):
        """
        Runs the YOLO model on a video or webcam stream.
        """
        video = cv2.VideoCapture(filename)  # Open the webcam or video file
        
        # Set width and height for the video feed
        width, height = 1024, 720
        video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        while True:
            ret, frame = video.read()  # Read video frames
            if not ret:
                break  # Exit if no frames are read

            # Run the YOLO model on the frame
            results = model.track(frame, persist=True, conf=0.43)
            res_plotted = results[0].plot()
            
            # Display the frame with bounding boxes
            cv2.imshow("Tracking from Webcam", res_plotted)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MyYOLOApp().run()
