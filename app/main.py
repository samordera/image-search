from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView

# Conditional import for Android-specific modules
try:
    from android.permissions import request_permissions, Permission
    android_imported = True
except ImportError:
    android_imported = False

class MyApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # Request permissions only if running on Android
        if android_imported:
            request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])

        self.filechooser = FileChooserIconView()
        self.filechooser.bind(on_selection=self.selected)
        self.add_widget(self.filechooser)

        self.image = Image()
        self.add_widget(self.image)

    def selected(self, filechooser, selection):
        if selection:
            self.image.source = selection[0]

class MainApp(App):
    def build(self):
        return MyApp()

if __name__ == '__main__':
    MainApp().run()
