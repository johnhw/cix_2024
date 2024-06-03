import numpy as np 

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

