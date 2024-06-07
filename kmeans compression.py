import tkinter as tk
from tkinter import filedialog
from skimage import io
from sklearn.cluster import MiniBatchKMeans
import numpy as np
def select_image():
 root = tk.Tk()
 root.withdraw() # Hide the main window
 file_path = filedialog.askopenfilename(title="Select Image File", 
filetypes=[("Image files", ".png;.jpg;*.jpeg")])
 return file_path
def select_output_path():
 root = tk.Tk()
 root.withdraw() # Hide the main window
 output_path = filedialog.asksaveasfilename(defaultextension=".png", 
filetypes=[("PNG files", "*.png")])
 return output_path
# Reading filename
filename = select_image()
# Reading the image
image = io.imread(filename)
# Preprocessing
rows, cols = image.shape[0], image.shape[1]
image_flat = image.reshape(-1, 3)
# Modeling with MiniBatchKMeans for faster processing
print('Compressing...')
print('Note: This can take a while for a large image file.')
kMeans = MiniBatchKMeans(n_clusters=16, batch_size=1000)
# Adjust batch_size as needed
kMeans.fit(image_flat)
# Getting centers and labels
centers = np.uint8(kMeans.cluster_centers_)
labels = np.uint8(kMeans.labels_.reshape(rows, cols))
print('Almost done.')
# Reconstructing the image
newImage = centers[labels]
# Asking the user for the output path
output_path = select_output_path()
# Saving the compressed image
io.imsave(output_path, newImage)
print('Image has been compressed successfully.')
