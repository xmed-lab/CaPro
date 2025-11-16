import os
from PIL import Image


def convert_png_to_jpg(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            png_path = os.path.join(input_folder, filename)
            jpg_filename = os.path.splitext(filename)[0] + ".png"
            jpg_path = os.path.join(output_folder, jpg_filename)
            with Image.open(png_path) as img:
                img = img.convert("RGB")
                img.save(jpg_path, "JPEG")
            print(f"Converted {filename} to {jpg_filename}")


if __name__ == "__main__":
    input_folder = ''
    output_folder = ''
    convert_png_to_jpg(input_folder, output_folder)
