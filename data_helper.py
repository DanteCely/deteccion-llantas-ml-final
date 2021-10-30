import os
import cv2
import numpy as np


if __name__ == "__main__":

    flat_directory = "./dataset-tire/flat.class"
    full_directory = "./dataset-tire/full.class"

    directories = {
        1: "./dataset-tire/flat.class",
        0: "./dataset-tire/full.class"
    }

    images_list = []

    # Label used: 1 -> flat tire
    #             0 -> full tire

    for key in directories.keys():
        images_names = os.listdir(directories.get(key))

        for flat_image_name in images_names:
            img = cv2.imread(flat_directory + "/" + flat_image_name)
            gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            images_list.append(np.append(gray_image.flatten(), [key]))

    images_array = np.array(images_list)
    np.random.shuffle(images_array)

    np.savetxt('./dataset-tire/input_data.csv', images_array, delimiter=',', newline='\n', fmt='%u')

    print("Process finished. ", len(images_list),  " images processed")
