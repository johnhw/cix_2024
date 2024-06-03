from key_test import capture_keys
from tkanvas import TKanvas
import queue

from tkinter import mainloop
import tkinter as tk

import numpy as np
from multiprocessing import Queue, Process

import time
import matplotlib.pyplot as plt
from key_noise import Corrupter


def tkcolor(rgb):
    return "#" + "".join(("%02X" % (int(c * 255)) for c in rgb[:3]))



class KeyDisplay(object):
    def __init__(
        self,
        q,
        
    ):
        
        self.q = q  # keyboard input
        self.keys = np.zeros(128, dtype=np.float32)
        self.key_map = {}
        self.cursor = (0,0)
        
        self.active_scan_code = None 
        self.canvas = TKanvas(
            draw_fn=self.draw,
            tick_fn=self.tick,
            event_fn=self.mouse_event,
            w=2300,
            h=730,
        )
        self.canvas.title("Ctrl-ESC-ESC-ESC to quit")
        self.keyboard_image = tk.PhotoImage(file="../imgs/qwerty.png")
        
        self.canvas.canvas.create_image(0,0,anchor=tk.NW, image=self.keyboard_image)
        self.text = self.canvas.text(
            0,0,
            text="OK",
            fill="white",
            anchor="w",
            font=("Arial", 16),
        )

        self.cursor_point = self.canvas.circle(self.cursor[0], self.cursor[1], 10, fill="red")
        

    def update_keymap(self, x, y):
        if self.active_scan_code is not None:
            self.key_map[self.active_scan_code] = (x, y)
            self.cursor = (x, y)

    def mouse_event(self, src, etype, event):        
        if etype == "mousedown":
            self.update_keymap(event.x, event.y)
            
    def tick(self, dt):
        try:
            result = self.q.get(block=False)
            if result:
                arr_bytes, t, code, event, name  = result
                # update cursor display
                if code in self.key_map:                    
                    self.cursor = self.key_map[code]
                else:
                    self.cursor = (0,0)
                # set the current scan code
                self.active_scan_code = code
                self.keys[:] = np.frombuffer(arr_bytes, dtype=np.float32)
            else:
                self.canvas.quit(None)
                   
        except queue.Empty:
            # no updates, do nothing
            pass
        
    def draw(self, src):
        # draw the blank squares for the outputs
        
        self.canvas.canvas.moveto(self.cursor_point, self.cursor[0], self.cursor[1])        
        
        


def key_tk(*args, **kwargs):
    import keyboard

    current_state = keyboard.stash_state()
    q = Queue()
    keys = Process(target=capture_keys, args=(q,))
    keys.start()
    k = KeyDisplay(q, *args, **kwargs)
    keyboard.restore_state(current_state)
    time.sleep(0.5)
    keyboard.restore_state(current_state)
    return current_state


if __name__ == "__main__":
    import keyboard, atexit

    current_state = keyboard.stash_state()
    atexit.register(keyboard.restore_state, current_state)
    q = Queue()
    keys = Process(target=capture_keys, args=(q,))
    keys.start()
    k = KeyDisplay(q)
    mainloop()

