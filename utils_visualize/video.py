from typing import List
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np


def save_video(images_list, video_path: str):
    """Saves a video from a list of images

    Args:
        images_list (List[Image]): A list of PIL images.
        video_path (str): The path to save to video to.
    """
    images = [np.array(img) for img in images_list]
    height, width, _ = images[0].shape

    fps = len(images) // 20
    video = cv2.VideoWriter(video_path, 0, fps, (width, height))

    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
    video.release()
    print(video_path)

img_path = "/home/daniel/diff-seg/core/checkpoints/diffseg-2023-05-31-01-52-58-232746/samples_15x256x256x3.npz"
video_path = "/home/daniel/diff-seg/core/checkpoints/diffseg-2023-05-31-01-52-58-232746"

def main():
    # Load npz file
    images = np.load(img_path)["arr_0"]
    print(images.shape)
    # Convert to PIL images
    images = [Image.fromarray(np.uint8(img)) for img in images]
    # Save images with plt
    print(images[0])
    for i, img in enumerate(images):
        plt.imshow(img)
        plt.savefig(f"{video_path}/{i}.png")




if __name__ == "__main__":
    main()