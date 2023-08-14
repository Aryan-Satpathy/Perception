import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as func
# from torchvision.utils.data import Dataset, DataLoader

import cv2
import numpy as np

BatchSize = 1

Latent_Size = 128
n_filters = 16

class MyDataset(Dataset):
    def __init__(self, path_to_video):
        cap = cv2.VideoCapture(path_to_video)
        self.total_frames = int(cap.get(7))

        # self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.width = 960
        self.height = 600

        self.data = []# np.ones((self.total_frames, self.height, self.width, 3), np.uint8)

        for i in range(self.total_frames):
            # _, self.data[i, :, :, :] = cap.read()
            _, data = cap.read()
            data  = cv2.resize(data, (600, 960))
            self.data.append(func.to_tensor(data)) #.transpose(1,2,0)))

        cap.release()

    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        return self.data[idx]

class Encoder(nn.Module):
    def __init__(self, latent_size, n_filters, img_size):
        super(Encoder, self).__init__()

        self.w, self.h = img_size

        # w, h, 3 -> w,  h, n_filters
        self.l0 = nn.Conv2d(in_channels = 3, out_channels = n_filters, kernel_size = 3, stride = 1, padding = 1)
        self.l1 = nn.LeakyReLU()

        # w, h, n_filters -> w / 2,  h / 2, 2 * n_filters
        self.l2 = nn.Conv2d(in_channels = n_filters, out_channels = 2 * n_filters, kernel_size = 3, stride = 2, padding = 1)
        self.l3 = nn.LeakyReLU()

        # w / 2, h / 2, 2 * n_filters -> w / 2,  h / 2, 2 * n_filters
        self.l4 = nn.Conv2d(in_channels = 2 * n_filters, out_channels = 2 * n_filters, kernel_size = 3, stride = 1, padding = 1)
        self.l5 = nn.LeakyReLU()

        # w / 2, h / 2, 2 * n_filters -> w / 4,  h / 4, 4 * n_filters
        self.l6 = nn.Conv2d(in_channels = 2 * n_filters, out_channels = 4 * n_filters, kernel_size = 3, stride = 2, padding = 1)
        self.l7 = nn.LeakyReLU()

        # w / 4,  h / 4, 4 * n_filters -> w / 4,  h / 4, 4 * n_filters
        self.l8 = nn.Conv2d(in_channels = 4 * n_filters, out_channels = 4 * n_filters, kernel_size = 3, stride = 1, padding = 1)
        self.l9 = nn.LeakyReLU()

        # w / 4,  h / 4, 4 * n_filters -> w / 8, h / 8, 8 * n_filters
        self.l10 = nn.Conv2d(in_channels = 4 * n_filters, out_channels = 8 * n_filters, kernel_size = 3, stride = 2, padding = 1)
        self.l11 = nn.LeakyReLU()

        # w / 8, h / 8, 8 * n_filters -> w x h x n_filters / 8
        self.l12 = nn.Flatten()

        self.size = self.get_size()

        self.mu = nn.Linear(in_features = self.size, out_features = Latent_Size)
        self.var = nn.Linear(in_features = self.size, out_features = Latent_Size)

    def get_size(self):
        dummy_in = torch.zeros((1, 3, self.w, self.h))

        out = self.l0(dummy_in)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        out = self.l8(out)
        out = self.l9(out)
        out = self.l10(out)
        out = self.l11(out)
        print(out.shape)
        out = self.l12(out)

        print(out.shape[1], (self.w // 8) * (self.h // 8) * n_filters * 8)

        return out.shape[1]

    def forward(self, x: torch.tensor):
        out = self.l0(x)

        print(out.shape)
        out = self.l1(out)

        print(out.shape)
        out = self.l2(out)

        print(out.shape)
        out = self.l3(out)

        print(out.shape)
        out = self.l4(out)

        print(out.shape)
        out = self.l5(out)

        print(out.shape)
        out = self.l6(out)

        print(out.shape)
        out = self.l7(out)

        print(out.shape)
        out = self.l8(out)

        print(out.shape)
        out = self.l9(out)

        print(out.shape)
        out = self.l10(out)

        print(out.shape)
        out = self.l11(out)

        print(out.shape)
        out = self.l12(out)

        print(out.shape)
        
        mu = self.mu(out)
        std = torch.exp(self.var(out) / 2)

        eps = torch.randn_like(mu)

        kl_div = -0.5 * (1 + torch.log(std**2) - mu**2 - std**2)

        z = mu + eps * std
        return z, kl_div
    
class Decoder(nn.Module):
    def __init__(self, latent_size, n_filters, img_size):
        super(Decoder, self).__init__()

        self.w, self.h = img_size

        self.l13 = nn.Linear(in_features = Latent_Size, out_features = (self.w // 8) * (self.h // 8) * n_filters * 8)
        self.l12 = nn.Unflatten(dim = 1, unflattened_size = (8 * n_filters, self.h // 8, self.w // 8))

        self.l11 = nn.ConvTranspose2d(in_channels = 8 * n_filters, out_channels = 4 * n_filters, kernel_size = 3, stride = 2, padding = 0)
        self.l10 = nn.LeakyReLU()
 
        self.l9 = nn.ConvTranspose2d(in_channels = 4 * n_filters, out_channels = 4 * n_filters, kernel_size = 3, stride = 1, padding = 0)
        self.l8 = nn.LeakyReLU()

        self.l7 = nn.ConvTranspose2d(in_channels = 4 * n_filters, out_channels = 2 * n_filters, kernel_size = 3, stride = 2, padding = 1)
        self.l6 = nn.LeakyReLU()

        self.l5 = nn.ConvTranspose2d(in_channels = 2 * n_filters, out_channels = 2 * n_filters, kernel_size = 3, stride = 1, padding = 0)
        self.l4 = nn.LeakyReLU()

        self.l3 = nn.ConvTranspose2d(in_channels = 2 * n_filters, out_channels = n_filters, kernel_size = 3, stride = 2, padding = 1)
        self.l2 = nn.LeakyReLU()

        self.l1 = nn.ConvTranspose2d(in_channels = n_filters, out_channels = 3, kernel_size = 3, stride = 1, padding = 0)
        self.l0 = nn.Sigmoid()

    def forward(self, x: torch.tensor):
        out = self.l13(x)

        print(out.shape)
        out = self.l12(out)

        print(out.shape)
        out = self.l11(out)

        print(out.shape)
        out = self.l10(out)

        print(out.shape)
        out = self.l9(out)

        print(out.shape)
        out = self.l8(out)

        print(out.shape)
        out = self.l7(out)

        print(out.shape)
        out = self.l6(out)

        print(out.shape)
        out = self.l5(out)

        print(out.shape)
        out = self.l4(out)

        print(out.shape)
        out = self.l3(out)

        print(out.shape)
        out = self.l2(out)

        print(out.shape)
        out = self.l1(out)

        print(out.shape)
        out = self.l0(out)

        print(out.shape)
        
        return out
    
class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()

        self.enc = encoder
        self.dec = decoder
    
    def forward(self, x):
        out, loss = self.enc(x)
        return self.dec(out), loss 

def train():
    dataset = MyDataset("UndistortedView.avi")
    dataloader = DataLoader(dataset, batch_size = BatchSize, shuffle = True)

    print(dataset.get_height(), dataset.get_width())
    # exit()

    encoder = Encoder(Latent_Size, n_filters = n_filters, img_size = (dataset.get_height(), dataset.get_width()))
    decoder = Decoder(Latent_Size, n_filters = n_filters, img_size = (dataset.get_height(), dataset.get_width()))

    ae = AE(encoder = encoder, decoder = decoder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ae.to(device = device)
    
    loss_fn = nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(ae.parameters(), 0.0003)

    num_epochs = 30

    train_losses = []

    reconstruction_loss_factor = 1

    for epoch in range(1, num_epochs+1):
        batch_losses = []
        for i, x in enumerate(dataloader):
            ae.train()
            
            x = x.to(device)

            # Step 1 - Computes our model's predicted output - forward pass
            y, kl_loss = ae(x)

            print(x.shape, y.shape)

            # Step 2 - Computes the loss
            # reduce (sum) over pixels (dim=[1, 2, 3])
            # and then reduce (sum) over batch (dim=0)
            loss = loss_fn(y, x).sum(dim=[1, 2, 3]).sum(dim=0)
            # reduce (sum) over z (dim=1)
            # and then reduce (sum) over batch (dim=0)
            kl_loss = kl_loss.sum(dim=1).sum(dim=0)
            # we're adding the KL loss to the original MSE loss
            total_loss = reconstruction_loss_factor * loss + kl_loss

            # Step 3 - Computes gradients
            total_loss.backward()
            
            # Step 4 - Updates parameters using gradients and the learning rate
            optim.step()
            optim.zero_grad()
            
            batch_losses.append(np.array([total_loss.data.item(), 
                                        loss.data.item(), 
                                        kl_loss.data.item()]))

        # Average over batches
        train_losses.append(np.array(batch_losses).mean(axis=0))

        print(f'Epoch {epoch:03d} | Loss >> {train_losses[-1][0]:.4f}/ \
            {train_losses[-1][1]:.4f}/{train_losses[-1][2]:.4f}')
        
if __name__ == '__main__':
    train()