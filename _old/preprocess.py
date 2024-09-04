import os
from torchvision import transforms
from PIL import Image

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Paths to the original and preprocessed image directories
original_dir = 'data/train/car'  # Adjust this to preprocess other categories
preprocessed_dir = 'data/preprocessed/train/car'

# Ensure the preprocessed directory exists
os.makedirs(preprocessed_dir, exist_ok=True)

# Loop through each image in the original directory, preprocess, and save
for filename in os.listdir(original_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(original_dir, filename)
        image = Image.open(img_path)
        image = transform(image)

        # Convert the tensor back to a PIL image for saving
        save_image = transforms.ToPILImage()(image)
        save_image.save(os.path.join(preprocessed_dir, filename))
