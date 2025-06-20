import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple


class ImageToData:
    """
    Converts images from a labeled folder structure into a machine learning-friendly dataset.

    Arguments:
    - master_path (str): Root folder containing dataset(s).
    - encoding_type (str): 'RGB' or 'Grayscale'.
    - resolution_convert (Tuple[int, int]): Desired image resolution (width, height). Use (0, 0) to keep original.
    - output_format (str): 'csv' or 'npz'.
    - output_path (str): Path to save the output file(s).
    - data_set (str): Specific dataset to convert (subfolder under master_path). Converts all if None.
    """

    def __init__(self, master_path: str, encoding_type: str, resolution_convert: Tuple[int, int] = (0, 0),
                 output_format: str = "npz", output_path: str = "./", data_set: str = None):
        self.master_path = master_path
        self.encoding_type = encoding_type.upper()
        self.resolution_convert = resolution_convert
        self.output_format = output_format.lower()
        self.output_path = output_path
        self.data_set = data_set

        if self.encoding_type not in ("RGB", "GRAYSCALE"):
            raise ValueError("encoding_type must be either 'RGB' or 'Grayscale'")
        if self.output_format not in ("csv", "npz"):
            raise ValueError("output_format must be either 'csv' or 'npz'")

        self.process()

    def process(self):
        datasets = [self.data_set] if self.data_set else os.listdir(self.master_path)

        for dataset_name in datasets:
            dataset_path = os.path.join(self.master_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            data = []
            labels = []

            for label in os.listdir(dataset_path):
                label_path = os.path.join(dataset_path, label)
                if not os.path.isdir(label_path):
                    continue
                count = 0
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    try:
                        image = Image.open(image_path)

                        if self.encoding_type == "RGB":
                            image = image.convert("RGB")
                        else:
                            image = image.convert("L")

                        if self.resolution_convert != (0, 0):
                            image = image.resize(self.resolution_convert)

                        image_array = np.array(image).flatten()
                        data.append(image_array)
                        labels.append(label)
                    except Exception as e:
                        print(f"Failed to process image {image_path}: {e}")
                    count+=1
                    print(f"Label: {label}, Iteration: {count}")

            data = np.array(data)
            labels = np.array(labels)

            # Save output
            filename = os.path.join(self.output_path, dataset_name + "." + self.output_format)
            if self.output_format == "csv":
                df = pd.DataFrame(data)
                df["label"] = labels
                df.to_csv(filename, index=False)
            elif self.output_format == "npz":
                np.savez_compressed(filename, data=data, labels=labels)

            print(f"Dataset '{dataset_name}' processed and saved to {filename}")


ImageToData("data", encoding_type="Grayscale", output_path="output", resolution_convert=(128,128), output_format="npz", data_set="emotions_dataset")
