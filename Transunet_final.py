import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class Resnet(nn.Module):
    def __init__(self,net_layers):
        super(Resnet, self).__init__()
        #change the first convolution since it expected 3 RGB channels as input
        self.layer0 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
        self.layer1 = nn.Sequential(*net_layers[1:3])
        self.layer2 = nn.Sequential(*net_layers[3:5])
        self.layer3 = nn.Sequential(*net_layers[5])
        self.layer4 = nn.Sequential(*net_layers[6])

    def forward(self,x):

        y1 = self.layer0(x)
        y1 = self.layer1(y1)

        y2 = self.layer2(y1)

        y3 = self.layer3(y2)

        out = self.layer4(y3)

        return y1, y2, y3, out



class MSA(nn.Module):
    def __init__(self,hidden_size):
        super(MSA, self).__init__()

        num_heads = 12   #hyperparameter to match pretrained weights size

        self.n_heads = num_heads
        self.k = hidden_size // self.n_heads

        self.query = nn.Linear(hidden_size, self.k * self.n_heads)
        self.key = nn.Linear(hidden_size, self.k * self.n_heads)
        self.value = nn.Linear(hidden_size, self.k * self.n_heads)

        self.unifyheads = nn.Linear(self.k * self.n_heads, hidden_size)


    def forward(self, x):

        b,t = x.size()[0:2]
        h = self.n_heads

        q = self.query(x).view(b, t, h, self.k)
        key = self.key(x).view(b, t, h, self.k)
        v = self.value(x).view(b, t, h, self.k)

    #fold head dimension back into batch dimension
        q = q.transpose(1,2).reshape(b*h,t,self.k)
        key = key.transpose(1,2).reshape(b*h,t,self.k)
        v = v.transpose(1,2).reshape(b*h,t,self.k)

        w_prime = torch.bmm(q, key.transpose(1, 2))
        w_prime = w_prime / math.sqrt(self.k)
        w = F.softmax(w_prime, dim=2) 

    #compute y and reshape from b*h to b,h
        y = torch.bmm(w, v).view(b, h, t, self.k)
        y = y.transpose(1,2).reshape(b,t,h*self.k)
        y = self.unifyheads(y)
   
        return y


class MLP(nn.Module):
    def __init__(self, hidden_size):

        percep_dim = 3072 #hyper parameter needed because of pretrained weights

        super(MLP, self).__init__()
        self.linear1 = nn.Linear(hidden_size, percep_dim)
        self.linear2 = nn.Linear(percep_dim, hidden_size)
        self.activate = nn.ReLU()

    def forward(self, x):

        #could add dropout after activation and linear2
        x = self.linear1(x)
        x = self.activate(x)
        x = self.linear2(x)

        return x 
    

class MSA_MLP_block(nn.Module):
    def __init__(self,hidden_size):
        super(MSA_MLP_block, self).__init__()

        self.h_size = hidden_size

        self.attent_layer = MSA(self.h_size)
        self.norm1 = nn.LayerNorm(self.h_size)

        self.percep_layer = MLP(self.h_size)
        self.norm2 = nn.LayerNorm(self.h_size)

    def forward(self,x):
        y = x
        x = self.norm1(x)
        x = self.attent_layer(x) + y

        y = x 
        x = self.norm2(x)
        x = self.percep_layer(x) + y

        return x


def reform(loaded_weights):
    if loaded_weights.ndim == 3:
        reshap = torch.from_numpy(loaded_weights).view(768,768).t()
    else:
        reshap = torch.from_numpy(loaded_weights).view(768)
    return nn.Parameter(reshap)

def change_totens(loaded_weights):
    return nn.Parameter(torch.from_numpy(loaded_weights))


class Transformer(nn.Module):
    def __init__(self, hidden_size, img_size, num=12, patch_size = 1):


        super(Transformer, self).__init__()
    
        #this part is for the embedding
        shrink_factor = 16 #R50 structure decrease H and W dimesnsion with factor of 8 and input size is 1/2 of orginial
        channels = 64 * shrink_factor #number of channels after resnet layer
    
        patch_size_2 = patch_size * shrink_factor
        num_patches = (img_size //patch_size_2) **2

        self.embed_patch = nn.Conv2d(channels,hidden_size,patch_size,patch_size)
        self.embed_pos = nn.Parameter(torch.zeros(1,num_patches,hidden_size))

        self.layers = nn.ModuleList([MSA_MLP_block(hidden_size) for i in range(num)])

        self.norm = nn.LayerNorm(hidden_size)

        self.init_weight()

    def init_weight(self):
        w = np.load('R50+ViT-B_16.npz')

        for j in range(len(self.layers)):
            trans_block = "Transformer/encoderblock_" + str(j)

            self.layers[j].attent_layer.query.weight = reform(w[trans_block + "/MultiHeadDotProductAttention_1/query/kernel"])
            self.layers[j].attent_layer.query.bias = reform(w[trans_block + "/MultiHeadDotProductAttention_1/query/bias"])

            self.layers[j].attent_layer.key.weight = reform(w[trans_block + "/MultiHeadDotProductAttention_1/key/kernel"])
            self.layers[j].attent_layer.key.bias = reform(w[trans_block + "/MultiHeadDotProductAttention_1/key/bias"])
    
            self.layers[j].attent_layer.value.weight = reform(w[trans_block + "/MultiHeadDotProductAttention_1/value/kernel"])
            self.layers[j].attent_layer.value.bias = reform(w[trans_block + "/MultiHeadDotProductAttention_1/value/bias"])

            self.layers[j].attent_layer.unifyheads.weight = reform(w[trans_block + "/MultiHeadDotProductAttention_1/out/kernel"])
            self.layers[j].attent_layer.unifyheads.bias = reform(w[trans_block + "/MultiHeadDotProductAttention_1/out/bias"])
          
            self.layers[j].norm1.weight = reform(w[trans_block + "/LayerNorm_0/scale"])
            self.layers[j].norm1.bias = reform(w[trans_block + "/LayerNorm_0/bias"])
    
            self.layers[j].percep_layer.linear1.weight = change_totens(w[trans_block + "/MlpBlock_3/Dense_0/kernel"].T)
            self.layers[j].percep_layer.linear1.bias = change_totens(w[trans_block + "/MlpBlock_3/Dense_0/bias"])

            self.layers[j].percep_layer.linear2.weight = change_totens(w[trans_block + "/MlpBlock_3/Dense_1/kernel"].T)
            self.layers[j].percep_layer.linear2.bias = change_totens(w[trans_block + "/MlpBlock_3/Dense_1/bias"])
    
            self.layers[j].norm2.weight = reform(w[trans_block + "/LayerNorm_2/scale"])
            self.layers[j].norm2.bias = reform(w[trans_block + "/LayerNorm_2/bias"])


    def forward(self, x):

        x = self.embed_patch(x)
        x = x.flatten(2) #flatten the dimension that contain the number of patches
        x = x.transpose(-1,-2) #swap around patches and hidden dimension

        x = x + self.embed_pos

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    ) 

class Decod(nn.Module):
    def __init__(self, n_class, hidden_size):

        super().__init__()
        N = 16                  

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)       

        self.dconv5 = double_conv(hidden_size, 512)
        self.dconv_up4 = double_conv(32*N+32*N, 16*N)    
        self.dconv_up3 = double_conv(16*N + 16*N, 8*N)
        self.dconv_up2 = double_conv(4*N + 8*N, 4*N)
        self.dconv_up1 = double_conv(4*N, N)
        
        self.conv_last = nn.Conv2d(N, n_class, 3, stride=1, padding=3//2)

 


    def forward(self, x, skip1,skip2,skip3):
        x = self.dconv5(x)

        x = self.upsample(x)
        x = torch.cat([x, skip3], dim=1)
 
        x = self.dconv_up4(x)
        x = self.upsample(x)     
        x = torch.cat([x, skip2], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)    
        x = torch.cat([x, skip1], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)  
        
        x = self.dconv_up1(x)
        
        y = self.conv_last(x)

        return y



class TransUnet(nn.Module):
    def __init__(self,R_layers,hidden_size=768,img_size=224):

        super(TransUnet, self).__init__()
        num_classes = 1 #8 classes and background

        self.CNN = Resnet(list(R_layers.children()))
        self.tr = Transformer(hidden_size,img_size)

        self.cup = Decod(num_classes,hidden_size)

    def forward(self,x):

        skip1, skip2, skip3, x = self.CNN(x)

        x = self.tr(x)

    #reshape the output of the transformer
        batch_size, n_patch, hid_size = x.shape
        x = torch.reshape(x,[batch_size,hid_size,int(math.sqrt(n_patch)),int(math.sqrt(n_patch))])

        predicted = self.cup(x,skip1,skip2,skip3)
    

        return predicted



# %%
# data loader 
import os
from PIL import Image 
# pillow library
import numpy as np
from torch.utils.data import Dataset

#dataset class,this is classic
class T_Dataset(Dataset):
    def __init__(self,image_directory,mask_directory,transform=True):
        # image_directory is the image path,transform is augmentations,
        # mask_directory can be changed to targets if it is image classification
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.transform = transform
        self.images = os.listdir(image_directory)
        #os.listdir() method in python is used to get the list of all files and directories
        #in the specified directory. If we don’t specify any directory, 
        #then list of files and directories in the current working directory will be returned.
    
    
    def __len__(self):
        return len(self.images)# size of image dataset 
    
    def __getitem__(self,index): # index iterates from 0 to number of dataset
        img_path=os.path.join(self.image_directory, self.images[index])
        #os.path.join() method in Python join one or more path components intelligently
        mask_path=os.path.join(self.mask_directory, self.images[index].replace('.jpg','_mask.gif'))
        image= np.array(Image.open(img_path).convert('RGB'))#convert image from BGR to RGB
        mask = np.array(Image.open(mask_path).convert('L'),dtype=np.float32) #L is the greyscale 
                                                                             # in this mask，
                                                                             # white is labeled as 255.0 
                                                                             #and black is labeled as 0.0 
                
        mask[mask == 255.0] =1.0 # preprocess for the mask, since sigmoid activation is used, so transfer 255 to 1
        
        
        #image=np.transpose(image,(2,0,1).astype(np.float32)) # change the array of image to channel first
        #tensor.unsqueeze(0) #if the iamge is greyscale, you need to add one dimension to it
        if self.transform is not None:# data augmentaion
            augmentations =self.transform(image = image, mask = mask)
            image = augmentations['image']
            mask = augmentations ['mask']
        return image,mask
            
            

# %%
# create training part
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm #loading progress bar
import torch.optim as optim

#save & load mode
def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    print('=> saving checkpoint')
    torch.save(state,filename)

def load_checkpoint(checkpoint,model):
    print('=> loading checkpoint')
    model.load.state.dict(checkpoint['state_dict'])

def check_accuracy(loader,model,device='cuda'):
    num_correct=0
    num_pixels=0
    dice_score=0
    model.eval()
    
    with torch.no_grad():
        for image,mask in loader:
            images=image.to(device)
            masks=mask.to(device).unsqueeze(1) # since this is greyscale so add one dimension with unsqueeze
            preds=torch.sigmoid(model(images))
            preds=(preds>0.5).float()
            num_correct+=(preds==masks).sum()
            num_pixels+=torch.numel(preds)#numel: number of elements
            dice_score+= (2*(preds*masks).sum())/((preds+masks).sum()+1e-8) 
            # this is for binary to evaluate the output, google for multiclass
    print(f' got{num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}' ) 
    print(f' Dice score:{dice_score/len(loader)}' )
    model.train()

def save_predictions_as_imgs(loader,model,folder='C:/Users/clarq/Desktop/data/saved images',device='cuda'):
    model.eval()
    for idx,(images,masks) in enumerate(loader):
        images=images.to(device=Device)
        with torch.no_grad():
            preds=torch.sigmoid(model(images))
            preds=(preds>0.5).float()
        torchvision.utils.save_image(
            preds,f'{folder}/pred_{idx}.png')
        torchvision.utils.save_image(masks.unsqueeze(1),f'{folder}/true_{idx}.png')
        
    model.train() 

# %%
# hyperparameters
learning_rate=1e-4
Device='cuda' if torch.cuda.is_available() else 'cpu'

Batch_size=2
num_epochs=1
num_workers=0
Image_height=224  # 1280 originally, use 160 in this example(only use a small part of the image),
                # in real competition, use 1280
Image_width=224 #1918 originally
Pin_memory= True
Load_model= True
val_percent: float = 0.3
train_image_directory='C:/Users/clarq/Desktop/data/train'
train_mask_directory='C:/Users/clarq/Desktop/data/train_masks'
# val_img_directory='C:/Users/clarq/Desktop/data/val'
# val_mask_directory='C:/Users/clarq/Desktop/data/val_masks'

# %%
#data augmentation,create model
from torch.utils.data import DataLoader
def main():
    train_transform = A.Compose([
        A.Resize(height=Image_height, width= Image_width),
        A.Rotate(limit=35,p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0,0,0],std=[1,1,1],max_pixel_value=225),
        ToTensorV2(),
    ])
#     val_transform = A.Compose([
#         A.Resize(height=Image_height, width= Image_width),
#         A.Normalize(mean=[0,0,0],std=[1,1,1],max_pixel_value=225),
#         ToTensorV2(),
#     ])
    

    Full_Rnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    model = TransUnet(Full_Rnet).to(Device)
    loss_function=nn.BCEWithLogitsLoss()# bianry cross entropy, sigmoid is included in the loss function
                                         # use CROSS entropy loss for multiple classification

    optimizer= optim.Adam(model.parameters(),lr=learning_rate)
    

    ds=T_Dataset(image_directory=train_image_directory,mask_directory=train_mask_directory,
                     transform=train_transform)

    
    n_val = int(len(ds) * val_percent)
    n_train = len(ds) - n_val
    train_set, val_set = torch.utils.data.random_split(ds, [n_train, n_val],
                                                   generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, shuffle=True,batch_size=Batch_size,pin_memory=Pin_memory,num_workers=0)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=Batch_size,pin_memory=Pin_memory,num_workers=0)
      
    
    if Load_model: #Load model = False as set, change to True to use this if loop
        load_checkpoint(torch.load('my_checkpoint.pth.tar'),model)
        
    check_accuracy(val_loader,model,device=Device)
        
        
        
    scaler=torch.cuda.amp.GradScaler()
    
  
    for epoch in range(num_epochs):
        loop=tqdm(train_loader)    
        for batch_idx, (data,targets) in enumerate(loop):
            data= data.to(device=Device)# data is the input image
            targets=targets.float().unsqueeze(1).to(device=Device)#target is the masks
            #forward
            with torch.cuda.amp.autocast():
                predictions=model(data)
                loss=loss_function(predictions,targets)
            
        
        
            #backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            #update tadm loop
            loop.set_postfix(loss=loss.item())
         
        #save model
        checkpoint={'state_dict': model.state_dict(),'optimizer':optimizer.state_dict(),}
        save_checkpoint(checkpoint)
        
        #check accuracy
        check_accuracy(val_loader, model,device=Device)
        
        # print some exmaples to the folder
        save_predictions_as_imgs(val_loader,model,folder='C:/Users/clarq/Desktop/data/saved images',device=Device)
  
    
   
    print('Finished Training')  
    
        
if __name__=='__main__':
    main()

# %%
# '''Dataloader'''
# class SynDataset(Dataset):
#     def __init__(self,path,transforms=None):
#         self.image_list = listdir(path)
#         #listdir() method in python is used to get the list of all files and directories
#         #in the specified directory. If we don’t specify any directory, 
#         #then list of files and directories in the current working directory will be returned.
#         self.path = path
#         self.transforms=transforms
    
#     def __len__(self):
#         return len(self.image_list)# size of image dataset 
  
#     def __getitem__(self,idx): # index iterates from 0 to number of dataset
#         img = np.load(self.path+self.image_list[idx])
        
#         im, la = img['image'], img['label']
       
#         sample = {'image': im, 'label': la} #make a dictionary
    
#         if self.transforms is not None:
#             sample=self.transforms(sample) #在后面调用SynDataset train model 的时候会define 具体的 transform 
        
#         return sample

    
    
# '''data augmentation'''
# class Resize(object):
#     def __init__(self, output_size):
        
#         self.size = output_size
      
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label'] #call the dictionary sample['image']= im
    
#         im = transform.resize(image, (self.size, self.size))
#         la = transform.resize(label, (self.size, self.size),order=0,anti_aliasing=False)
        
#         return {'image': im, 'label': la}
    
# class Rotate(object):

#     def __init__(self, angle):
        
#         self.angle = angle
      
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
        
#         if np.random.rand()<0.25:        
#             alpha = np.random.randint(-self.angle, self.angle)
#             image = transform.rotate(image, alpha, resize=False)
#             label = transform.rotate(label, alpha, resize=False)
        
#         return {'image': image, 'label': label}
    
# class flip(object):
    
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
            
#         if np.random.rand()<0.25: 
#             image = np.flipud(image).copy()
#             label = np.flipud(label).copy()
#         elif np.random.rand()<0.25: 
#             image = np.fliplr(image).copy()
#             label = np.fliplr(label).copy()
            
#         return {'image': image, 'label': label}


# """
# load the train data and train the network
# """

# train_path = '/home/dewolf151/train_npz/'
# train_files = listdir(train_path)
# batch_size = 24
# num_epochs = 250 

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('the available device is:', device, flush=True)

           
# composed=transforms.Compose([Resize(224),
#                           Rotate(15),
#                           flip()])

# trainloader=DataLoader(SynDataset(train_path,composed),batch_size=batch_size,shuffle=True)

# Full_Rnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
# model = TransUnet(Full_Rnet)
# model.to(device)

# ce_loss = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

# store_loss = np.zeros((num_epochs))
# acc = []

# for epoch in range(num_epochs):
#   epoch_loss = 0
#   for ii, data in enumerate(trainloader):
#     image, labels = data['image'].to(device), data['label'].to(device)
    
#     im_in = image.unsqueeze(1)
#     prediction = model(im_in)
    
#     loss = ce_loss(prediction, labels[:].long())
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     epoch_loss += loss.item()    
        
#   avg_loss = epoch_loss / (ii+1)
#   print('epoch %d loss: %.3f' % (epoch + 1, avg_loss ), flush=True)
    
#   store_loss[epoch] = avg_loss
  
#   if epoch % 5 == 0:
#       max_pred = torch.argmax(prediction,1)
#       num_pix = labels.nelement()
#       corr = max_pred.eq(labels).sum().item()
#       acc.append(100 * corr / num_pix)
      
        
# torch.save(model.state_dict(), 'saved_models/mod.pth')

# np.save('saved_data/train_acc',acc)
# np.save('saved_data/train_loss',store_loss)

# """
# Define the metrics used to quantify the performance during the testing time
# """

# def diceCoef(pred, gt):
#     nclas = 8 #the number of classes except background
#     N = gt.size(0) #the batch size
    
#     pred_flat = pred.view(N, -1)
#     gt_flat = gt.view(N, -1)
    
#     Dice = np.zeros((N,nclas))
#     for ii in range(nclas):
#         logit_pred = pred_flat == ii+1
#         logit_gt = gt_flat == ii+1
     
#         intersection = (logit_pred * logit_gt).sum(1)
#         unionset = logit_pred.sum(1) + logit_gt.sum(1)
#         Dice[:,ii] = 2 * (intersection) / (unionset)
    
#     #Dice[np.isnan(Dice)] = 0 #remove the Nan when class is not present
#     return 100*Dice

# def Hausdorff(pred, gt):
#     batch_s = pred.size(0)
#     HDC = np.zeros(batch_s)
    
#     pred[pred>0] = 1
#     pred = pred.detach().cpu().numpy()
    
#     gt[gt>0] = 1
#     gt = gt.detach().cpu().numpy()
    
#     for jj in range(batch_s): 
#         if pred[jj,:,:].sum() > 0 and gt[jj,:,:].sum()>0:
#             HDC[jj] = binary.hd95(pred[jj,:,:], gt[jj,:,:])
#         else:
#             HDC[jj] = 0
        
#     return HDC

# """
# Rescale the input images back for the test data set
# """
# def Test_rescale(sample,size):

#     image, label = sample['image'][:], sample['label'][:]
    
#     N = image.shape[0]
#     im = transform.resize(image, (N, size, size))
#     la = transform.resize(label, (N, size, size),order=0,preserve_range=True,anti_aliasing=False).astype(np.uint8)
        
#     return im, la

# """
# Load the test data and run it through the model
# """

# #Full_Rnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
# #model = TransUnet(Full_Rnet)
# #model.to(device)
# #model.load_state_dict(torch.load('saved_models\\mod.pth'))

# test_path = '/home/dewolf151/test_vol_h5/'
# test_files = listdir(test_path)  

# HD = []
# first_it = True
# for nn in range(len(test_files)):

#     test_data = h5py.File(test_path + test_files[nn])
    
#     T_image, T_label = Test_rescale(test_data,224)
    
#     #only send the images to gpu for running it through network
#     dataset = torch.utils.data.TensorDataset(torch.Tensor(T_image).to(device), 
#                                              torch.Tensor(T_label) )
        
#     testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                               shuffle=False, num_workers=0)
    
    
#     for i, data in enumerate(testloader):
#         image, labels = data
        
#         im_in = image.unsqueeze(1)
#         prediction = model(im_in)
        
#         prediction = torch.argmax(prediction, dim=1).cpu() #get tensor back to cpu
        
                    
#         if first_it:
#             DSC = diceCoef(prediction, labels)
#             first_it = False
#         else:
#             DSC = np.concatenate((DSC, diceCoef(prediction, labels)), axis=0)
            
#         HD = np.append(HD, Hausdorff(prediction, labels))
        
# np.save('saved_data/Dice_scores',DSC)
# np.save('saved_data/HD_scores',HD)

# print('training and testing has finished', flush=True)
        

# %%


# %%


# %%
