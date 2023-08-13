import json
from guided_diffusion import dist_util, logger
import blobfile as bf

def main():
    # Read the json file
    logger.configure()
    with open("./segmented-images/bounding-boxes.json") as f:
        data = json.load(f)
    keys = list(data.keys())
    bbox_dict = {k: data[k]["bbox"] for k in keys}
    

    def _list_image_files_recursively(data_dir):
        results = []
        for entry in sorted(bf.listdir(data_dir)):
            full_path = bf.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                results.append(full_path)
            elif bf.isdir(full_path):
                results.extend(_list_image_files_recursively(full_path))
        return results

    # Read file names in segmented-images/images
    img_paths = _list_image_files_recursively("./segmented-images/images")
    # Remove the prefix and .jpg extension
    img_paths = [p[26:-4] for p in img_paths]

    # Read file names in segmented-images/masks
    mask_paths = _list_image_files_recursively("./segmented-images/masks")
    # Remove the prefix and .jpg extension
    mask_paths = [p[25:-4] for p in mask_paths]
    for i, k in enumerate(keys):
        logger.log(f"{i+1}: {k} {bbox_dict[k]}")
        if k in img_paths and (k not in mask_paths):
            logger.log(f"Image {k} has no mask")

    # Copy masked images to masked-images
    # for i, k in enumerate(keys):
    #     if k in img_paths and k in mask_paths:
    #         bf.copy(f"./segmented-images/images/{k}.jpg", f"./segmented-images/masked-images/{k}.jpg")
    masked_img_paths = _list_image_files_recursively("./segmented-images/masked-images")
    logger.log(f"Number of masked images: {len(masked_img_paths)}")

if __name__ == "__main__":
    main()