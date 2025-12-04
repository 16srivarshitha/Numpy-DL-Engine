import matplotlib.pyplot as plt
import numpy as np

def save_reconstruction(img, save_path, epoch=None, loss=None):
    plt.figure(figsize=(6, 6))
    
    if img.ndim == 2:  # Grayscale
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    else:  # RGB
        plt.imshow(img)
    
    plt.axis('off')
    
    title = "Reconstruction"
    if epoch is not None:
        title += f" - Epoch {epoch}"
    if loss is not None:
        title += f" - Loss: {loss:.6f}"
    plt.title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()