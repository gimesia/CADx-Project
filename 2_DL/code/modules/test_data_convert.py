import os
import pandas as pd
import shutil

excel_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\Computer_Aided_Diagnosis\DL_2\2Create_test\binary_results_final.xlsx".replace("\\", "/")
image_folder = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\Computer_Aided_Diagnosis\Labs\Datasets\test\testX".replace("\\", "/")

image_files = sorted(os.listdir(image_folder))
df = pd.read_excel(excel_path, header = None, names = ["Label"])
print("-------------------------")
if len(image_files) != len(df):
    raise ValueError("The number of images does not match the number of labels.")

else:
    print(f"Number of images and labels match: {len(image_files)}")

print("-------------------------")
print(df.head())
print("-------------------------")
print(image_files[:10])
print("-------------------------")

#Create output folders for cases 0 and 1

output_folder_0 = os.path.join(image_folder,"class_0")
output_folder_1 = os.path.join(image_folder,"class_1")

os.makedirs(output_folder_0, exist_ok= True)
os.makedirs(output_folder_1, exist_ok= True)


#Pair each image with label and move to appropriate folder

for label, image_name in zip(df["Label"], image_files):

    source_path = os.path.join(image_folder, image_name)

    if label == 0:
        destination_path = os.path.join(output_folder_0)
    elif label == 1:
        destination_path = os.path.join(output_folder_1)
    else:
        print(f"Unknown label {label} for image {image_name}")

    shutil.move(source_path, destination_path)
    print(f"Moved image {image_name} to {destination_path}")