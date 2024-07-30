from audioop import bias
from ssl import CHANNEL_BINDING_TYPES
from turtle import forward
from unittest import TestCase
import torch 
import torch.nn as nn
from torch.autograd import Variable 



#-----------------STAGE I: GENERATOR (LOW RESOLUTION IMAGE PROD.)-------------------------#
class stageI_Gen(nn.Module):
    def __init__(self):
        super(stageI_Gen,self).__init__()
        self.gf_dim = 192*8
        self.ef_dim = 128
        self.z_dim = 100
        self.define_module()
    
    def define_module(self):
        n_input = self.z_dim + self.ef_dim
        ngf = self.gf_dim
        self.ca_net = Ca_Net()
        
        self.fc = nn.Sequential(
            nn.Linear(n_input, ngf*4*4,bias=False),
            nn.BatchNorm1d(ngf*4*4),
            nn.ReLU(True)
        )
        
        self.upsamples = [
            upBlock(ngf, ngf//2),
            upBlock(ngf//2, ngf//4),
            upBlock(ngf//4, ngf//8),
            upBlock(ngf//8, ngf//16)
        ]
        self.img = nn.Sequential(conv3x3(ngf//16,3),nn.Tanh())
        
    def forward(self, text_embedding, noise):
         c_code , mu, logvar = self.ca_net(text_embedding)
         z_c_code = torch.cat((noise,c_code),1)
         h_code = self.fc(z_c_code)
         
         h_code = h_code.view(-1, self.gf_dim,4,4)
         for upsample in self.upsamples:
             h_code = upsample(h_code)
         fake_img = self.img(h_code)
         return None, fake_img, mu, logvar

#-----------------STAGE I: DISCRIMINATOR (LOW RESOLUTION IMAGE PROD.)-------------------------#

class stageI_Dis(nn.Module):
    def __init__(self):
        super(stageI_Dis, self).__init__()
        self.df_dim = 96
        self.ef_dim = 128
        self.define_module()
    
    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.get_cond_logits = D_logits(ndf,nef)
        self.get_uncond_logits = None
        
    def forward(self,image):
        img_embedding = self.encode_img(image)
        return img_embedding

#-----------------STAGE II: GENERATOR (HIGH RESOLUTION IMAGE PROD.)-------------------------#

  
class StageII_GeN(nn.Module):
    def __init__(self,stageII_GeN):
        super(stageII_GeN,self).__init__()
        self.gf_dim = GAN_GF_DIM
        self.ef_dim = GAN_CONDITION_DIM 
        self.z_dim = Z_DIM 
        self.stageI_GeN = stageI_Gen
        for param in self.stageI_GeN.parameters():
            param.requires_grad = False
        self.define_module()
        
    def _make_layer(self,block,channel_num):
        layers = []
        for _ in range(GAN_R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.ca_net = Ca_Net()
        self.encoder = nn.Sequential(
            conv3x3(3,ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf,ngf*2,4,2,1,bias=False),
            nn.Batchnorm2d(ngf*2),
            nn.ReLU(True),
            nn.Conv2d(ngf*2,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)     
            )
        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + ngf*4, ngf*4),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
        )
        self.residual = self._make_layer(ResBlock,ngf*4)
        
        self.upsample1 = upBlock(ngf*4,ngf*2)
        self.upsample2 = upBlock(ngf*2,ngf)
        self.upsample3 = upBlock(ngf,ngf//2)
        self.upsample4 = upBlock(ngf//2,ngf//4)
        self.img = nn.Sequential(
            conv3x3(ngf//4,3),
            nn.Tanh()
        )
        
    def forward(self, text_embedding, noise):
        _, stage1_img, _, _ = self.stageI_Gen(text_embedding, noise)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)

        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = c_code.view(-1,self.ef_dim,1,1)
        c_code = c_code.repeat(1,1,16,16)
        i_c_code = torch.cat([encoded_img,c_code],1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        fake_img = self.img(h_code)
        return stage1_img, fake_img, mu, logvar
    
        
     
#-----------------STAGE II: DISCRIMINATOR (HIGH RESOLUTION IMAGE PROD.)-------------------------#

class StageII_Disc(nn.Module):
    def __init__(self,StageII_Disc):
        super(StageII_Disc,self).__init__()
        self.df_dim = GAN_DF_DIM 
        self.ef_dim = GAN_CONDITION_DIM
        self.define_module()
        
        
    def define_module(self):
        ndf,nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.Batchnorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.Batchnorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.Batchnorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*8,ndf*16,4,2,1,bias=False),
            nn.Batchnorm2d(ndf*16),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*16,ndf*32,4,2,1,bias=False),
            nn.Batchnorm2d(ndf*32),
            nn.LeakyReLU(0.2,inplace=True),
            conv3x3(ndf*32,ndf*16),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2,inplace=True),
            conv3x3(ndf*16,ndf*8),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.get_cond_logits = D_GET_LOGITS(ndf,nef,bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf,nef,bcondition=False)
    
    def forward(self,image):
        img_embedding = self.encode_img(image)
        return img_embedding
    
    
        
#-----------------------------HELPER FUNCTIONS---------------------------------------------------#  

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
             
def upBlock(in_channels, out_channels):
    block = nn.Sequential(
        nn.Upsample(scale_factor =2, mode='nearest'),
        conv3x3(in_channels,out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )   
    return block    
        
        
class Ca_Net(nn.Module):
    def __init__(self):
        super(Ca_Net,self).__init__()
        self.t_dim = 1024
        self.c_dim = 128
        self.fc = nn.Linear(self.t_dim, self.c_dim*2, bias=True)
        self.relu = nn.ReLU()
        
    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:,:self.c_dim]
        logvar = x[:,self.c_dim:]
        return mu, logvar 
    
    def reprametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reprametrize(mu,logvar)
        return c_code, mu, logvar 

class D_Logits(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_Logits,self).__init__()
        self.df_dim = ndf 
        self.ef_dim = nef 
        self.bcondition = bcondition 
        
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf*8 + nef, ndf*8),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(ndf*8,1,kernel_size=4,stride=4),
                nn.Sigmoid()
            )
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf*8,1,kernel_size=4,stride=4),
                nn.Sigmoid()
            )
    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim,1,1)
            c_code = c_code.repeat(1,1,4,4)
            h_c_code = torch.cat((h_code,c_code),1)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output.view(-1)
                
            
class ResBlock(nn.Module):
    def __init__(self,channel_num):
        super(ResBlock,self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.Batchnorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num,channel_num),
            nn.Batchnorm2d(channel_num)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        residual = x
        out = self.block(x)
        out += residual 
        out = self.relu(out)
        return out 
    
class D_GET_LOGITS(nn.Module):
    def __init__(self,ndf,nef,bcondition=True):
        super(D_GET_LOGITS,self).__init__()
        self.df_dim = ndf 
        self.ef_dim = nef 
        self.bcondition = bcondition 
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf*8+nef, ndf*8),
                nn.Batchnorm2d(ndf*8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(ndf*8,1,kernel_size=4),
                nn.Sigmodi()
            )
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf*8,1,kernel_size=4,stride=4),
                nn.Sigmoid()
            )
    def forward(self,h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code  = c_code.view(-1,self.ef_dim,1,1)
            c_code = c_code.repeat(1,1,4,4)
            h_c_code = torch.cat((h_code,c_code),1)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output.view(-1)