import argparse
import os
import random
import shutil
import glob

def get_image_paths(image_dir):
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return image_paths

def sample_data(map_name, new, old, sep=False, image_dir='h:\\111\\images', output_dir='h:\\111\\sampled'):
    map_image_dir = os.path.join(image_dir, map_name)
    map_output_dir = os.path.join(output_dir, map_name)

    # Get all image paths
    image_paths = get_image_paths(map_image_dir)

    total_images = len(image_paths)
    if total_images < new + old:
        print(f"Not enough images in {map_image_dir}. Found {total_images}, need {new + old}.")
        return

    # Randomly sample images
    random.shuffle(image_paths)
    sampled_old = image_paths[:old]
    sampled_new = image_paths[old:old + new]

    # Create output directories
    if sep:
        old_dir = os.path.join(map_output_dir, 'old')
        new_dir = os.path.join(map_output_dir, 'new')
        os.makedirs(old_dir, exist_ok=True)
        os.makedirs(new_dir, exist_ok=True)
        for img_path in sampled_old:
            shutil.copy(img_path, old_dir)
        for img_path in sampled_new:
            shutil.copy(img_path, new_dir)
    else:
        os.makedirs(map_output_dir, exist_ok=True)
        for img_path in sampled_old + sampled_new:
            shutil.copy(img_path, map_output_dir)

    print(f"Sampled {len(sampled_old)} old and {len(sampled_new)} new images for {map_name}.")
    if sep:
        print(f"Images saved to {old_dir} and {new_dir}.")
    else:
        print(f"Images saved to {map_output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample data for training.')
    parser.add_argument('--map_name', type=str, required=True, help='Name of the map (e.g., village_small, village_medium, village_large).')
    parser.add_argument('--new', type=int, required=True, help='Number of new training images to sample.')
    parser.add_argument('--old', type=int, required=True, help='Number of old training images to sample.')
    parser.add_argument('--sep', action='store_true', help='Save images to separate folders for old and new.')
    parser.add_argument('--image_dir', type=str, default='h:\\111\\images', help='Directory containing the map images.')
    parser.add_argument('--output_dir', type=str, default='h:\\111\\sampled', help='Directory to save the sampled images.')

    args = parser.parse_args()

    # Validate new and old values
    valid_values = [2, 4, 6, 8, 10, 12]
    if args.new not in valid_values or args.old not in valid_values:
        print(f"Error: --new and --old must be one of {valid_values}.")
    else:
        sample_data(args.map_name, args.new, args.old, args.sep, args.image_dir, args.output_dir)
