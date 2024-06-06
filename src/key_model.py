import numpy as np 
import csv 

class KeyModel:
    def __init__(self, key_map, size, bw=0.02, noise=0.0001, alpha=0.8, intensity=2.0,
                 frame_drop=0, lag=0, intensity_std=0.0, position_drift=0.0, slc=(9, 30), bw_std=0.0):
        self.key_map = key_map
        self.states = {}
        self.img_width = key_map["width"]
        self.img_height = key_map["height"]
        self.key_image = np.zeros((size, size)) 
        self.key_buffer = np.zeros((size, size))
        self.out_buffer = np.zeros((1+lag, size, size))    
        self.key_mesh = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        self.x_scale = 1.0 / max(self.img_width, self.img_height)
        self.y_scale = 1.0 / max(self.img_height, self.img_width) 
        self.x_offset = (1.0 - self.img_width * self.x_scale) / 2
        self.y_offset = (1.0 - self.img_height * self.y_scale) / 2
        self.noise = noise 
        self.alpha = alpha 
        self.bw = bw 
        self.bw_std = bw_std
        self.intensity = intensity
        self.frame_drop = frame_drop
        self.lag = lag
        self.intensity_std = intensity_std
        self.position_drift = position_drift
        self.slc = slc
        self.output = np.zeros([slc[0], slc[1]])
        self.keys_down = set()

        
    def kernel(self, x, y, bw, intensity):
        # convert to (0,1) normalised coordinates
       
        mx, my = self.key_mesh 
        # eval kernel
        return np.exp(-((((mx-x) ** 2 + (my-y) ** 2) / (2 * bw ** 2)))) * intensity

    def up(self, code):
        
        if code not in self.states:
            self.states[code] = "down"
        if code in self.key_map and self.states[code] == "down":
            if code in self.keys_down:
                self.keys_down.remove(code)
            self.states[code] = "up"

    def down(self, code):
        if code not in self.states:
            self.states[code] = "up"
        if code in self.key_map and self.states[code] == "up":
            self.keys_down.add(code)
            self.states[code] = "down"

    def clear(self):
        self.keys_down.clear()

    def tick(self):
        self.key_image[:] = 0.0
        for code in self.keys_down:
            noised_intensity = self.intensity + np.random.normal(0, 1) * self.intensity_std 
            noised_bw = self.bw + np.random.normal(0, 1) * self.bw_std
            ix,iy = self.key_map[code]
            x, y = ix * self.x_scale, iy * self.y_scale
            x, y = x + np.random.normal(0, self.position_drift), y + np.random.normal(0, self.position_drift)

            self.key_image += self.kernel(x, y, noised_bw, noised_intensity)
        self.key_buffer = self.alpha * self.key_buffer + (1 - self.alpha) * self.key_image
        self.key_buffer += np.random.normal(0, self.noise, self.key_image.shape)
        self.key_buffer = np.clip(self.key_buffer, 0, 1)      
        dropped_frame = np.random.rand() < self.frame_drop
        self.out_buffer = np.roll(self.out_buffer, 1, axis=0)
        self.out_buffer[0] = self.key_buffer  
        if dropped_frame:
            self.out_buffer[0] = 0.0
        self.output = self.out_buffer[-1, :self.slc[0], :self.slc[1]]
    

    def _simulate_press(self, buffer, x, y, size=0.02, intensity=1.0):
        # simulate a key press at x,y
        buffer[:] = self.kernel(x * self.x_scale, y * self.y_scale, bw=size, intensity=intensity)
        buffer += np.random.normal(0, self.noise, buffer.shape)
        buffer[:] = np.clip(buffer, 0, 1)      
        return buffer

    def simulate_press(self, x, y, size=0.02, intensity=1.0):
        self._simulate_press(self.key_image, x, y, size, intensity)
        return self.key_image[:self.slc[0], :self.slc[1]]
    
class KeySimulator:
    def __init__(self, size, slc, **kwargs):
        key_map = {"width":1.0, "height":slc[0]/slc[1]}
        self.km = KeyModel(key_map, size, slc=slc, **kwargs)
        self.size = size
        self.slc = slc

    def press(self, x, y, size=0.04, intensity=1.5, steps=5, slc=(9, 30)):
        x = self.slc[1] * x / self.size
        y = self.slc[0] * y / self.size
        img = self.km.simulate_press(x, y, size, intensity)
        return img
    
    def buffer_press(self, buffer, x, y, size=0.04, intensity=1.5):
        x = self.slc[1] * x / self.size
        y = self.slc[0] * y / self.size
        self.km._simulate_press(buffer, x, y, size, intensity)
        return buffer 


def multi_simulate(size, slc, xs, ys, sizes, intensities, **kwargs):
    sim = KeySimulator(size, slc, **kwargs)
    n = len(xs)
    buffers = np.zeros((n, size, size)) 
    for i in range(n):
        x, y = xs[i], ys[i]
        size = sizes[i]
        intensity = intensities[i] 
        sim.buffer_press(buffers[i], x, y, size, intensity)
    return buffers[:, :slc[0], :slc[1]]     

def batch_simulate(size, slc, n, out_file, size_sampler, intensity_sampler, x_sampler, y_sampler, **kwargs):
    sim = KeySimulator(size, slc, **kwargs)
    with open(out_file, "a") as f:
        writer = csv.writer(f)
        for i in range(n):
            x, y = x_sampler(), y_sampler()
            bw = size_sampler()
            intensity = intensity_sampler()
            img = sim.press(x, y, bw, intensity)
            csv_row = img.flatten().tolist() + [x,y]
            writer.writerow(csv_row)
    print(f"Wrote {n} samples to {out_file}")
    