import os
import cv2
import sys
import numpy as np

# This python script is used to load the images and store them in a csv file, also it helps to resize the images
# to the desired dimensions
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ", sys.argv[0], " new_size")
        exit(-1)

    new_size = int(sys.argv[1])
    flat_directory = "./dataset-tire/flat.class"
    full_directory = "./dataset-tire/full.class"

    directories = {
        1: "./dataset-tire/flat.class",
        -1: "./dataset-tire/full.class"
    }

    images_list = []

    # Label used: 1 -> flat tire
    #            -1 -> full tire

    for key in directories.keys():
        images_names = os.listdir(directories.get(key))

        for flat_image_name in images_names:
            img = cv2.imread(flat_directory + "/" + flat_image_name)
            gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            scaled_image = cv2.resize(gray_image, (new_size, new_size), cv2.INTER_LANCZOS4)
            images_list.append(np.append(scaled_image.flatten(), [key]))

    images_array = np.array(images_list)
    rng = np.random.default_rng()
    rng.shuffle(images_array)
    filename = './dataset-tire/input_data' + str(new_size) + '_' + str(new_size) + '.csv'

    np.savetxt(filename, images_array, delimiter=',', newline='\n', fmt='%u')

    print("Process finished. ", len(images_list),  " images processed")
