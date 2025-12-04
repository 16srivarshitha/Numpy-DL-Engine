import os
import time
import numpy as np
from src.layers import Linear, ReLU, Sigmoid
from src.engine import Model
from src.data import ImageDataset
from utils.vis import save_reconstruction

def train(image_path, method='fourier', epochs=100, freq=10):
    print(f"\n--- Training with {method.upper()} features ---")
    
    # 1. Setup Data
    dataset = ImageDataset(image_path, image_type='gray', method=method, freq=freq)
    input_dim = dataset.features.shape[1]
    
    # 2. Define Architecture 
    layers = [
        Linear(input_dim, 512, ReLU()),      
        Linear(512, 1024, ReLU()),           
        Linear(1024, 1024, ReLU()),          
        Linear(1024, 512, ReLU()),
        Linear(512, 256, ReLU()),            
        Linear(256, dataset.num_channels, Sigmoid())
    ]
    model = Model(layers, loss_type='mse', learning_rate=0.05)
    
    print(f"Model Parameters: {model.count_parameters()}")
    
    # 3. Training Loop
    save_dir = f"results_{method}"
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.time()
    batch_size = 256
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        dataset.shuffle()
        batch_gen = dataset.get_batch(batch_size)

        for x, y in batch_gen:
            model.zero_grad()
            loss = model.train_step(x, y)
            model.update()
            epoch_loss += loss
            num_batches += 1
                
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
            
            # Save intermediate result
            features, _ = dataset.get_all_data()
            preds = model.predict(features)
            img = dataset.reconstruct_image(preds)
            save_reconstruction(img, f"{save_dir}/epoch_{epoch+1}.png", epoch=epoch+1, loss=avg_loss)

    print(f"Training completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    train('smiley.png', method='raw', epochs=50)
    train('smiley.png', method='fourier', epochs=50, freq=10)