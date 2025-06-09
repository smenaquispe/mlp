import os
import csv
import random
from PIL import Image

def images_to_csv(dataset_path, output_csv_train, output_csv_test, train_ratio=0.8):
    # Open training and test CSV files
    with open(output_csv_train, 'w', newline='') as f_train, open(output_csv_test, 'w', newline='') as f_test:
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)

        # Write headers to both files
        header = ['label'] + [f'pixel{i}' for i in range(28*28)]
        writer_train.writerow(header)
        writer_test.writerow(header)

        for label in range(10):
            label_path = os.path.join(dataset_path, str(label))
            if not os.path.isdir(label_path):
                continue

            # List all image files in the label directory
            image_files = [file for file in os.listdir(label_path) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Shuffle the list of image files
            random.shuffle(image_files)
            # Calculate the split index
            split_index = int(len(image_files) * train_ratio)

            # Process training images
            for file in image_files[:split_index]:
                image_path = os.path.join(label_path, file)
                img = Image.open(image_path).convert('L').resize((28, 28))
                pixels = list(img.getdata())
                writer_train.writerow([label] + pixels)

            # Process test images
            for file in image_files[split_index:]:
                image_path = os.path.join(label_path, file)
                img = Image.open(image_path).convert('L').resize((28, 28))
                pixels = list(img.getdata())
                writer_test.writerow([label] + pixels)

# Call the function with the paths for training and test CSV files
images_to_csv('trainingSet/trainingSet', 'mnist_train.csv', 'mnist_test.csv')
