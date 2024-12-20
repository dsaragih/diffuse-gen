from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments
import os


if __name__ == "__main__":
    args = get_arguments()
    # Loop through images in input directory

    if args.cluster_path:
        # expect model_path to be path/to/cluster_model/modelNNNNNN.pt
        base_model_path = args.cluster_model_dir 
        for i in range(args.start_index, args.end_index + 1):
            path = None
            # split = base_model_path.split("/")
            # filename = f"cluster_{i}_" + split[-1]
            # split[-1] = filename
            # path = "/".join(split)
            # # Check if file exists
            # if not os.path.isfile(path):
            #     print(f"Cluster {i} does not exist")
            #     continue
            # Go through base_model_path and check that file starts with cluster_{i}_
            for filename in os.listdir(base_model_path):
                if filename.startswith(f"cluster_{i}_"):
                    path = os.path.join(base_model_path, filename)
            
            assert path is not None, f"Cluster {i} does not exist"
            
            args.model_path = path
            image_editor = ImageEditor(args)
            image_editor.sample_image()
    else:
        image_editor = ImageEditor(args)
        image_editor.sample_image()
