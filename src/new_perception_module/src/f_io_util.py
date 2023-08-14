import sys
import os

file_names = list(os.listdir('/home/mvslab/catkin_ws/Yolo_mark-master/x64/Release/data/img/'))

file_names = [file_name for file_name in file_names if file_name[-3:] == 'png' or file_name[-4:] == 'jpeg']#  and file_name[0] == 'i']

print(len(file_names))
exit()

# train_directory = r'datasets/coco128/images/train2017/'
# dest_dir_images = r'datasets/coco128/images/train2017/'
# dest_dir_label = r'datasets/coco128/labels/train2017/'

file_names = list(os.listdir(train_directory))
print(file_names)
file_names = [file_name for file_name in file_names if file_name[-3:] == 'npy']#  and file_name[0] == 'i']

for file_name in file_names:
    print(file_name)
    exit()
    os.remove(train_directory + file_name)
    # os.replace(dest_dir_label + 'labels' + file_name[6:-3] + 'txt', dest_dir_label + file_name[6:-3] + 'txt')

# class CustomDataset(Dataset):
#     def __init__(self, train_directory, batch_size = 1):
#         self.path = train_directory
#         self.file_names = list(os.listdir(train_directory))
#         self.image_names = [file_name for file_name in self.file_names if file_name[-1] == 'g']
#         self.label_names = [img_name.split('.jpg')[0] + '.txt' for img_name in self.image_names]
#         self.batch_size = batch_size
#     def __len__(self):
#         return len(self.image_names)
#     def __getitem__(self, index):
#         img_name, label_name = self.image_names[index], self.label_names[index]

#         img = torchvision.io.read_image(self.path + img_name)
#         img = img[None, :]
        
#         try:
#             with open(self.path + label_name, 'r') as f:
#                 unprocessed_labels = f.readlines()
        
#             labels = [tuple(map(eval, line[:-1].split(' '))) for line in unprocessed_labels]
#         except:
#             labels = []
        
#         return img, labels