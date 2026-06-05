import os
from PIL import Image

# 输入文件夹路径
input_folder = 'result1-1'
# 输出文件夹路径
output_folder = 'result1-1'

os.makedirs(output_folder, exist_ok=True)

target_size = (1920, 1080)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with Image.open(input_path) as img:
            # 调整尺寸，使用缩放且保持比例，填充或裁剪也可以根据需求调整
            img_resized = img.resize(target_size, Image.BILINEAR)
            img_resized.save(output_path)
            print(f"Processed {filename}")

print("All images resized to 1024x1024.")

