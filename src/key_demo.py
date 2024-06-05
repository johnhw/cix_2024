from key_test import capture_keys
from tkanvas import TKanvas
from key_model import KeyModel
import cmasher
import csv
import queue
import shelve
from tkinter import mainloop
import tkinter as tk
from pathlib import Path

import numpy as np
from multiprocessing import Queue, Process
import zmq
import time
import matplotlib.pyplot as plt
import json
from zmqutils import req_rep_loop

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

    def normalised_coordinates(self, x, y):
        # return coordinates in range [0,1] from an input
        # location in screen coordinates
        width = self.shape[1] * self.size
        height = self.shape[0] * self.size
        nx, ny = (x - self.origin[0]) / width, (y - self.origin[1]) / height
        
        return nx, ny 
    
    def unnormalised_coordinates(self, nx, ny):
        # return screen coordinates from normalised coordinates
        width = self.shape[1] * self.size
        height = self.shape[0] * self.size
        x, y = nx * width + self.origin[0], ny * height + self.origin[1]
        return x, y

    def update(self, matrix):
        assert matrix.shape == self.shape
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                ix = i * self.shape[1] + j
                color = self.cmap(matrix[i, j])[:3]
                self.canvas.canvas.itemconfig(self.rects[ix], fill=tkcolor(color))

class KeyRecorder:
    def __init__(self, fname, overwrite=False):
        if fname is not None:
            if overwrite:
                self.record_file = open(fname, "w")
            else:
                self.record_file = open(fname, "a")
            self.record_csv = csv.writer(self.record_file)
        else:
            self.record_file = None # dummy writer

    def write_row(self, inp, out):
        inp = np.array(inp).astype(np.float32).flatten().tolist()
        out = np.array(out).astype(np.float32).flatten().tolist()
        if self.record_file is not None:
            self.record_csv.writerow(inp + out)

    def close(self):
        if self.record_file is not None:
            self.record_file.close()



class KeyDisplay(object):
    def __init__(
        self,
        q,
        res=32,
        alpha=0.7,
        noise=0.02,     
        bw=0.03,   
        intensity=1.8,
        slc=(9,30),
        record_fname=None,
        overwrite=False,
        zmq_port=None,
    ):        
        
        self.q = q  # keyboard input
        self.local_dir = Path(__file__).parent
        self.key_map = shelve.open(self.local_dir / "keymap.db")    # connect to the keymap database
        self.model = KeyModel(self.key_map, res, alpha=alpha, noise=noise, bw=bw, intensity=intensity)
        self.block_size = 24
        self.slc = slc
        self.status = "OK"
        self.cursor_radius = 10
        self.seq = 0
        # for recording data to a file
        self.key_recorder = KeyRecorder(fname=record_fname, overwrite=overwrite)                

        # connect to zeromq to read/receive messages
        self.zmq_port = zmq_port
        if self.zmq_port is not None:
            self.recv_q = Queue()
            self.send_q = Queue()
            self.zmq_process = Process(target=req_rep_loop, args=(self.zmq_port, self.send_q, self.recv_q))
            # guarantee that the process is killed when the main process is killed
            self.zmq_process.daemon = True
            self.zmq_process.start()
                
        self.canvas = TKanvas(
            draw_fn=self.draw,
            tick_fn=self.tick,
            event_fn=self.event,
            w=self.block_size * (slc[1]+1),
            h=self.block_size * (slc[0]+2),
        )

        self.canvas.title("Ctrl-ESC-ESC-ESC to quit")
        # create the matrix display
        self.matrix_display = TKMatrix(self.canvas, slc, self.block_size, cmap=cmasher.cm.bubblegum)                
        self.text = self.canvas.text(
            self.block_size / 2,
            self.canvas.h - self.block_size / 2,
            text=str(self.status),
            fill="white",
            anchor="w",
            font=("Arial", 16),            
        )

        # click point (for data collection)
        self.cursor_point = self.canvas.circle(-100, -100, self.cursor_radius, fill="red")

        # target point (from a remote server)
        
        self.target_point = self.canvas.circle(-100, -100, 1, outline="red", width=1)


    def event(self, src, etype, event):        
        if etype == "mouseup":
            # on click, set the cursor display
            click = [event.x, event.y]
            self.canvas.canvas.moveto(self.cursor_point, click[0]-self.cursor_radius, click[1]-self.cursor_radius)
            key_slice = self.model.key_buffer[:self.slc[0], :self.slc[1]]
            nx, ny = self.matrix_display.normalised_coordinates(*click) # normalise screen coords to [0,1]
            
            self.key_recorder.write_row(key_slice, [nx, ny])
        
            
    def tick(self, dt):
        try:
            while not self.q.empty():
                # clear cursor
                self.canvas.canvas.moveto(self.cursor_point, -100,-100)
                # read results
                result = self.q.get(block=False)
                if result:
                    arr_bytes, t, code, event, name  = result                
                    code = str(code)
                    if event == "down":
                        self.model.down(code)
                    else:
                        self.model.up(code)
                    
                else:
                    self.key_recorder.close()
                    self.canvas.quit(None)
                    if self.zmq_port:
                        self.send_q.put({"quit":True})
        except queue.Empty:
            # no updates, do nothing
            pass
        
    def update_zmq(self):
        if self.zmq_port is None:
            return
        self.seq += 1
        
        # read/write from the zmq socket
        buffer = self.model.key_buffer[:self.slc[0], :self.slc[1]]
        try:
            self.send_q.put_nowait({"touch":buffer.tolist(), "seq":self.seq})
        except queue.Full:
            pass
        
        # respond to cursor updates    
        try_again = True
        while try_again:
            response = None 
            try:
                response = self.recv_q.get_nowait()
            except queue.Empty:
                try_again = False
            if response is not None:
                if "target" in response:
                    # convert to screen coordinates
                    x, y = self.matrix_display.unnormalised_coordinates(response["target"]["x"], response["target"]["y"])
                    rx, ry = self.matrix_display.unnormalised_coordinates(response["target"]["x"]+response["target"]["radius"], json_response["target"]["y"])
                    print(x, y, rx, ry)
                    radius = rx - x
                    self.canvas.canvas.coords(self.target_point, x-radius, y-radius, x+radius, y+radius)                                                           
                

    def draw(self, src):
        # draw the blank squares for the outputs
        self.model.tick()
        self.matrix_display.update(self.model.key_buffer[:self.slc[0], :self.slc[1]])   
        self.update_zmq()


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

import click 
@click.command()
@click.option("--file", default=None, help="Record filename")
@click.option("--overwrite", is_flag=True, help="Overwrite record file")
@click.option("--zmq_port", default=None, help="ZMQ port")
def start_key_display(file=None, overwrite=False, zmq_port=None):
    k = KeyDisplay(q, record_fname=file, overwrite=overwrite, zmq_port=zmq_port)
    mainloop()

if __name__ == "__main__":
    import keyboard, atexit

    current_state = keyboard.stash_state()
    atexit.register(keyboard.restore_state, current_state)
    q = Queue()
    keys = Process(target=capture_keys, args=(q,))
    keys.start()    
    start_key_display()
    

