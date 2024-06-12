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
from zmqutils import async_req_loop


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
            self.record_file = None  # dummy writer

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
        frame_drop=0.0,
        position_drift=0.0,
        intensity_std=0.0,
        intensity=1.8,
        slc=(9, 30),
        record_fname=None,
        timeseries_mode=False,
        overwrite=False,
        zmq_port=None,
        lag=0,
        bw_std=0.0,
    ):

        self.q = q  # keyboard input
        self.local_dir = Path(__file__).parent
        self.key_map = shelve.open(
            self.local_dir / "keymap.db"
        )  # connect to the keymap database
        self.model = KeyModel(
            self.key_map,
            res,
            alpha=alpha,
            noise=noise,
            bw=bw,
            intensity=intensity,
            frame_drop=frame_drop,
            position_drift=position_drift,
            intensity_std=intensity_std,
            lag=lag,
            slc=slc,
            bw_std=bw_std,
        )
        self.block_size = 32
        self.status = "OK"
        self.cursor_radius = 10
        self.seq = 0
        # for recording data to a file
        self.key_recorder = KeyRecorder(fname=record_fname, overwrite=overwrite)
        self.timeseries_mode = timeseries_mode
        # connect to zeromq to read/receive messages
        self.zmq_port = zmq_port
        if self.zmq_port is not None:
            self.recv_q, self.send_q = Queue(), Queue()
            self.zmq_process = Process(
                target=async_req_loop, args=(self.zmq_port, self.send_q, self.recv_q)
            )
            # guarantee that the process is killed when the main process is killed
            self.zmq_process.daemon = True
            self.zmq_process.start()

        self.canvas = TKanvas(
            draw_fn=self.draw,
            tick_fn=self.tick,
            event_fn=self.event,
            w=self.block_size * (slc[1] + 1),
            h=self.block_size * (slc[0] + 2),
        )

        self.canvas.title("Ctrl-ESC-ESC-ESC to quit")
        # create the matrix display
        self.matrix_display = TKMatrix(
            self.canvas, slc, self.block_size, cmap=cmasher.cm.bubblegum
        )
        self.text = self.canvas.text(
            self.block_size / 2,
            self.canvas.h - self.block_size / 2,
            text=str(self.status),
            fill="white",
            anchor="w",
            font=("Arial", 16),
        )

        # click point (for data collection)
        self.cursor_point = self.canvas.circle(
            -100, -100, self.cursor_radius, fill="red"
        )

        self.cursor_h_crosshair = self.canvas.line(
            0, 0, 0, self.canvas.h, fill="white", width=1)
        self.cursor_v_crosshair = self.canvas.line(
            0, 0, self.canvas.w, 0, fill="white", width=1)
        
        # target point (from a remote server)
        self.target_point = self.canvas.circle(-100, -100, 1, outline="white", width=2)

    def event(self, src, etype, event):
        if etype == "mouseup":
            # on click, set the cursor display
            click = [event.x, event.y]
            self.canvas.canvas.moveto(
                self.cursor_point,
                click[0] - self.cursor_radius,
                click[1] - self.cursor_radius,
            )
            key_slice = self.model.output
            nx, ny = self.matrix_display.normalised_coordinates(
                *click
            )  # normalise screen coords to [0,1]

            self.key_recorder.write_row(key_slice, [nx, ny])

    def tick(self, dt):
        try:
            while not self.q.empty():
                # clear cursor
                self.canvas.canvas.moveto(self.cursor_point, -100, -100)
                # read results
                result = self.q.get(block=False)
                # did we get a key press?
                if result:
                    arr_bytes, t, code, event, name = result
                    code = str(code)
                    if code=="1":
                        self.model.clear()
                    if event == "down":
                        self.model.down(code)
                    else:
                        self.model.up(code)

                else:
                    # no more key presses, close the record file
                    # and shut it all down
                    self.key_recorder.close()
                    self.canvas.quit(None)
                    if self.zmq_port:
                        self.send_q.put({"quit": True})
                        # kill it dead!
                        time.sleep(0.5)
                        self.zmq_process.terminate()
                        time.sleep(0.5)
                        self.zmq_process.kill()
        except queue.Empty:
            # no updates, do nothing
            pass

    def update_zmq(self):
        if self.zmq_port is None:
            return
        self.seq += 1

        # read/write from the zmq socket
        buffer = self.model.output
        try:
            self.send_q.put_nowait({"touch": buffer.tolist(), "seq": self.seq})
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
                    x, y = self.matrix_display.unnormalised_coordinates(
                        response["target"]["x"], response["target"]["y"]
                    )
                    rx, ry = self.matrix_display.unnormalised_coordinates(
                        response["target"]["x"] + response["target"]["radius"],
                        response["target"]["y"],
                    )
                    radius = rx - x
                    self.canvas.canvas.coords(
                        self.target_point,
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                    )
                    self.canvas.canvas.coords(
                        self.cursor_h_crosshair, 0, y, self.canvas.w, y)
                    self.canvas.canvas.coords(
                        self.cursor_v_crosshair, x, 0, x, self.canvas.h)
                    

    def draw(self, src):
        # draw the blank squares for the outputs
        if self.timeseries_mode and self.key_recorder is not None:
            key_slice = self.model.output
            # write out 0,0 in timeseries mode
            self.key_recorder.write_row(key_slice, [0, 0])

        self.model.tick()
        self.matrix_display.update(self.model.output)
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
@click.option("--file", default=None, help="Record filename, for creating training data or timeseries mode")
@click.option("--overwrite", is_flag=True, help="Overwrite record file")
@click.option("--zmq_port", default=None, help="ZMQ port, for live input")
@click.option("--noise", default=0.02, help="Noise level")
@click.option("--frame_drop", default=0.0, help="Frame drop fraction")
@click.option("--position_drift", default=0.0, help="Position drift fraction")
@click.option("--intensity_std", default=0.0, help="Intensity standard deviation")
@click.option("--bw", default=0.04, help="Bandwidth")
@click.option("--lag", default=0, help="Simulated lag (in frames)")
@click.option("--bw_std", default=0.0, help="Bandwidth standard deviation")
@click.option("--timeseries_mode", is_flag=True, help="Timeseries mode (record continuously, without any target positions)")
def start_key_display(
    file=None,
    timeseries_mode=False,
    overwrite=False,
    zmq_port=None,
    noise=0.02,
    frame_drop=0.0,
    position_drift=0.0,
    bw_std=0.0,
    bw=0.04,
    lag=0,
    intensity_std=0.0,
):
    k = KeyDisplay(
        q,
        record_fname=file,
        overwrite=overwrite,
        timeseries_mode=timeseries_mode,
        zmq_port=zmq_port,
        noise=noise,
        frame_drop=frame_drop,
        position_drift=position_drift,
        intensity_std=intensity_std,
        bw_std=bw_std,
        bw=bw,
        lag=lag,
    )
    mainloop()


if __name__ == "__main__":
    import keyboard, atexit

    current_state = keyboard.stash_state()
    atexit.register(keyboard.restore_state, current_state)
    q = Queue()
    keys = Process(target=capture_keys, args=(q,))
    keys.daemon = True
    keys.start()
    start_key_display()
