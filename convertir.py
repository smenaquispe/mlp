import os
import csv
from PIL import Image

def images_to_csv_train(dataset_path, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = ['label'] + [f'pixel{i}' for i in range(28*28)]
        writer.writerow(header)

        for label in range(10):
            label_path = os.path.join(dataset_path, str(label))
            if not os.path.isdir(label_path):
                continue

            for file in os.listdir(label_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(label_path, file)
                    img = Image.open(image_path).convert('L').resize((28, 28)) 
                    pixels = list(img.getdata()) 
                    writer.writerow([label] + pixels)


images_to_csv_train('trainingSet/trainingSet', 'mnist_train.csv')
