from key_test import capture_keys
from tkanvas import TKanvas
import queue
import shelve
from tkinter import mainloop

import numpy as np
from multiprocessing import Queue, Process

import time
import matplotlib.pyplot as plt


def tkcolor(rgb):
    return "#" + "".join(("%02X" % (int(c * 255)) for c in rgb[:3]))


class TKMatrix(object):
    def __init__(self, canvas, shape, size, origin=None, cmap=None):
        self.origin = origin or (size / 2, size / 2)
        self.cmap = cmap or plt.get_cmap("viridis")
        self.shape = shape
        self.size = size
        self.canvas = canvas
        self.create_rects()

    def create_rects(self):
        self.rects = []
        sz = self.size
        ox, oy = self.origin
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                rect = self.canvas.rectangle(
                    ox + j * sz,
                    oy + i * sz,
                    ox + (j + 1) * sz,
                    oy + (i + 1) * sz,
                    fill="blue",
                )
                self.rects.append(rect)

    def update(self, matrix):
        assert matrix.shape == self.shape
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                ix = i * self.shape[1] + j
                color = self.cmap(matrix[i, j])[:3]
                self.canvas.canvas.itemconfig(self.rects[ix], fill=tkcolor(color))

class KeyModel:
    def __init__(self, key_map, size, bw=0.02, noise=0.0001, alpha=0.8, intensity=2.0):
        self.key_map = key_map
        self.states = {}
        self.img_width = key_map["width"]
        self.img_height = key_map["height"]
        self.key_image = np.zeros((size, size)) 
        self.key_buffer = np.zeros((size, size))
        self.key_mesh = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        self.x_scale = 1.0 / max(self.img_width, self.img_height)
        self.y_scale = 1.0 / max(self.img_height, self.img_width) 
        self.x_offset = (1.0 - self.img_width * self.x_scale) / 2
        self.y_offset = (1.0 - self.img_height * self.y_scale) / 2
        self.noise = noise 
        self.alpha = alpha 
        self.bw = bw 
        self.intensity = intensity
        
    def kernel(self, ix, iy):
        # convert to (0,1) normalised coordinates
        x = (ix) * self.x_scale #+ self.x_offset
        y = (iy) * self.y_scale #+ self.y_offset        
        mx, my = self.key_mesh 
        # eval kernel
        return np.exp(-((((mx-x) ** 2 + (my-y) ** 2) / (2 * self.bw ** 2)))) * self.intensity

    def up(self, code):
        if code not in self.states:
            self.states[code] = "down"
        if code in self.key_map and self.states[code] == "down":
            self.key_image -= self.kernel(*self.key_map[code])
            self.states[code] = "up"

    def down(self, code):
        if code not in self.states:
            self.states[code] = "up"
        if code in self.key_map and self.states[code] == "up":
            self.key_image += self.kernel(*self.key_map[code])
            self.states[code] = "down"

    def tick(self):
        self.key_buffer = self.alpha * self.key_buffer + (1 - self.alpha) * self.key_image
        self.key_buffer += np.random.normal(0, self.noise, self.key_image.shape)
        self.key_buffer = np.clip(self.key_buffer, 0, 1)        


class KeyDisplay(object):
    def __init__(
        self,
        q,
        res=32,
        alpha=0.7,
        noise=0.02,     
        bw=0.025,   
        intensity=2.5,
        slc=(9,26),
    ):        
        
        self.q = q  # keyboard input
        self.key_map = shelve.open("keymap.db")    # connect to the keymap database
        self.model = KeyModel(self.key_map, res, alpha=alpha, noise=noise, bw=bw, intensity=intensity)
        self.block_size = 24
        self.slc = slc
        self.status = "OK"
                
        self.canvas = TKanvas(
            draw_fn=self.draw,
            tick_fn=self.tick,
            w=self.block_size * (slc[1]+1),
            h=self.block_size * (slc[0]+2),
        )

        self.canvas.title("Ctrl-ESC-ESC-ESC to quit")
        self.matrix_display = TKMatrix(self.canvas, slc, self.block_size)                
        self.text = self.canvas.text(
            self.block_size / 2,
            self.canvas.h - self.block_size / 2,
            text=str(self.status),
            fill="white",
            anchor="w",
            font=("Arial", 16),
        )


    def tick(self, dt):
        try:
            while not self.q.empty():
                result = self.q.get(block=False)
                if result:
                    arr_bytes, t, code, event, name  = result                
                    code = str(code)
                    if event == "down":
                        self.model.down(code)
                    else:
                        self.model.up(code)
                    
                else:
                    self.canvas.quit(None)
                   
        except queue.Empty:
            # no updates, do nothing
            pass
        
    def draw(self, src):
        # draw the blank squares for the outputs
        self.model.tick()
        self.matrix_display.update(self.model.key_buffer[:self.slc[0], :self.slc[1]])


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

