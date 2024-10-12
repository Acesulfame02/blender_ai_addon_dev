import bpy
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ReloadAddonHandler(FileSystemEventHandler):
    def __init__(self, addon_name):
        self.addon_name = addon_name
    
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f"Detected change in {event.src_path}")
            bpy.ops.preferences.addon_disable(module=self.addon_name)
            bpy.ops.preferences.addon_enable(module=self.addon_name)
            print(f"Reloaded {self.addon_name} due to changes in {event.src_path}")

def register():
    addon_name = 'addon'  # Make sure this matches the symlink name
    addon_path = os.path.join(bpy.utils.script_path_user(), "addons", addon_name)
    print(f"Addon path: {addon_path}")
    
    if not os.path.exists(addon_path):
        print(f"Addon path does not exist: {addon_path}")
        return

    event_handler = ReloadAddonHandler(addon_name)
    observer = Observer()
    observer.schedule(event_handler, addon_path, recursive=True)
    observer.start()
    print("Observer started")

def unregister():
    observer.stop()
    observer.join()

if __name__ == "__main__":
    register()
