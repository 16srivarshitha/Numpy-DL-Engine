import numpy as np
from PIL import Image
from .features import get_raw, get_polynomial, get_fourier, normalize_features

class ImageDataset:
    def __init__(self, image_path, image_type='gray', method='raw', order=5, freq=10):
        self.image_path = image_path
        self.image_type = image_type.lower()
        self.method = method.lower()
        self.order = order
        self.freq = freq
        
        self._load_image()
        self._create_dataset()
    
    def _load_image(self):
        img = Image.open(self.image_path)
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.LANCZOS)
        
        if self.image_type == 'gray':
            img = img.convert('L')
            self.img_array = np.array(img) / 255.0
            self.img_array = self.img_array[:, :, np.newaxis]
            self.num_channels = 1
        elif self.image_type == 'rgb':
            img = img.convert('RGB')
            self.img_array = np.array(img) / 255.0
            self.num_channels = 3
        else:
            raise ValueError(f"Unknown image_type: {self.image_type}")
        
        self.height, self.width = self.img_array.shape[:2]
        print(f"Loaded {self.image_type} image: {self.width}x{self.height}")
    
    def _create_dataset(self):
        coords_list = []
        pixels_list = []
        
        for y in range(self.height):
            for x in range(self.width):
                x_norm = x / (self.width - 1)
                y_norm = y / (self.height - 1)
                coords_list.append([x_norm, y_norm])
                pixel = self.img_array[y, x]
                pixels_list.append(pixel)
        
        self.coords = np.array(coords_list, dtype=np.float32)
        self.pixels = np.array(pixels_list, dtype=np.float32)
        
        if self.method == 'raw':
            self.features = get_raw(self.coords)
        elif self.method == 'polynomial':
            self.features = get_polynomial(self.coords, order=self.order)
        elif self.method == 'fourier':
            self.features = get_fourier(self.coords, freq=self.freq)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.features = normalize_features(self.features, self.method)
        self.num_samples = len(self.features)
    
    def shuffle(self):
        indices = np.random.permutation(self.num_samples)
        self.features = self.features[indices]
        self.pixels = self.pixels[indices]
        self.coords = self.coords[indices]
    
    def get_batch(self, batch_size):
        for i in range(0, self.num_samples, batch_size):
            end_idx = min(i + batch_size, self.num_samples)
            yield self.features[i:end_idx], self.pixels[i:end_idx]
    
    def get_all_data(self):
        return self.features, self.pixels
    
    def reconstruct_image(self, predictions):
        if self.image_type == 'gray':
            img = predictions.reshape(self.height, self.width)
        else:
            img = predictions.reshape(self.height, self.width, self.num_channels)
        return np.clip(img, 0, 1)