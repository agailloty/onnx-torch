from torch.utils.data import Dataset
import os
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_labels = {}

        # Create a mapping of class labels to integers
        self.class_labels = {}
        class_idx = 0

        # Iterate over sub-directories
        for class_dir in os.listdir(self.root_dir):
            class_dir_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                self.class_labels[class_dir] = class_idx
                class_idx += 1

                # Iterate over images in the sub-directory
                for img_filename in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path, img_filename)
                    self.images.append(img_path)
                    self.labels.append(self.class_labels[class_dir])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        # Convert grayscale image to RGB
        if image.mode == "L":
            image = Image.merge("RGB", (image, image, image))
        if self.transform:
            image = self.transform(image)
        return image, label
    

def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc