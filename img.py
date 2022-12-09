from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval() 


if not os.path.exists('./data.pt'):
    dataset = datasets.ImageFolder('photos') # photos folder path 
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        face, prob = mtcnn0(img, return_prob=True) 
        if face is not None and prob>0.92:
            emb = resnet(face.unsqueeze(0)) 
            embedding_list.append(emb.detach()) 
            name_list.append(idx_to_class[idx])        

    # save data
    data = [embedding_list, name_list] 
    torch.save(data, 'data.pt') # saving data.pt file

# loading data.pt file
load_data = torch.load('data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

BASE_DIR = "TrainingImages"
list_imgs = os.listdir(BASE_DIR)

for im in list_imgs:
    print(im)
    img = cv2.imread(os.path.join(BASE_DIR, im)) # read image 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    name = ""
    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 

                dist_list = [] # list of matched distances, minimum distance is used to identify the person

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name = name_list[min_dist_idx] # get name corrosponding to minimum dist

                box = list(map(int, boxes[i]))
                original_image = img.copy() # storing copy of image before drawing on it

                # if min_dist<0.90:
                #    img = cv2.putText(img, name+' '+str(min_dist), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)   
                img = cv2.rectangle(img, (box[0],box[1]) , (box[2],box[3]), (0,0,255), 2)

    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"FinalImage/{im.split('.')[0]}-{name}.png", image)