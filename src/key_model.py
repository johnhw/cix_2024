import numpy as np 

class KeyModel:
    def __init__(self, key_map, size, bw=0.02, noise=0.0001, alpha=0.8, intensity=2.0,
                 frame_drop=0, lag=0, intensity_std=0.0, position_drift=0.0, slc=(9, 30)):
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
        self.intensity = intensity
        self.frame_drop = frame_drop
        self.lag = lag
        self.intensity_std = intensity_std
        self.position_drift = position_drift
        self.slc = slc
        self.output = np.zeros([slc[0], slc[1]])

        
    def kernel(self, ix, iy, bw, intensity):
        # convert to (0,1) normalised coordinates
        x = (ix) * self.x_scale #+ self.x_offset
        y = (iy) * self.y_scale #+ self.y_offset      
        x = x + np.random.normal(0, self.position_drift)
        y = y + np.random.normal(0, self.position_drift)
        noised_intensity = intensity + np.random.normal(0, 1) * self.intensity_std 
        mx, my = self.key_mesh 
        # eval kernel
        return np.exp(-((((mx-x) ** 2 + (my-y) ** 2) / (2 * bw ** 2)))) * noised_intensity

    def up(self, code):
        if code not in self.states:
            self.states[code] = "down"
        if code in self.key_map and self.states[code] == "down":
            self.key_image -= self.kernel(*self.key_map[code], self.bw, self.intensity)
            self.states[code] = "up"

    def down(self, code):
        if code not in self.states:
            self.states[code] = "up"
        if code in self.key_map and self.states[code] == "up":
            self.key_image += self.kernel(*self.key_map[code], self.bw, self.intensity)
            self.states[code] = "down"

    def tick(self):
        self.key_buffer = self.alpha * self.key_buffer + (1 - self.alpha) * self.key_image
        self.key_buffer += np.random.normal(0, self.noise, self.key_image.shape)
        self.key_buffer = np.clip(self.key_buffer, 0, 1)      
        dropped_frame = np.random.rand() < self.frame_drop
        self.out_buffer = np.roll(self.out_buffer, 1, axis=0)
        self.out_buffer[0] = self.key_buffer  
        if dropped_frame:
            self.out_buffer[0] = 0.0
        self.output = self.out_buffer[-1, :self.slc[0], :self.slc[1]]
            
    def simulate_press(self, x, y, size=0.02, intensity=1.0, steps=5, slc=(9, 30)):
        # simulate a key press at x,y
        self.key_image[:] = 0.0
        self.key_image += self.kernel(x, y, bw=size, intensity=intensity)
        for i in range(steps):
            self.tick()
        return self.key_buffer


class KeySimulator:
    def __init__(self, size, slc, **kwargs):
        key_map = {"width":1.0, "height":slc[0]/slc[1]}
        self.km = KeyModel(key_map, size, **kwargs)
        self.size = size
        self.slc = slc

    def press(self, x, y, size=0.04, intensity=1.5, steps=5, slc=(9, 30)):
        x = self.slc[1] * x / self.size
        y = self.slc[0] * y / self.size
        img = self.km.simulate_press(x, y, size, intensity, steps)
        
        return img[:slc[0], :slc[1]]

import csv 
def batch_simulate(size, slc, n, out_file, size_sampler, intensity_sampler, x_sampler, y_sampler, **kwargs):
    sim = KeySimulator(size, slc, **kwargs)
    with open(out_file, "a") as f:
        writer = csv.writer(f)
        for i in range(n):
            x, y = x_sampler(), y_sampler()
            size = size_sampler()
            intensity = intensity_sampler()
            img = sim.press(x, y, size, intensity)
            csv_row = img.flatten().tolist() + [x,y]
            writer.writerow(csv_row)
    print(f"Wrote {n} samples to {out_file}")
    