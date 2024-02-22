from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class VoxelDecoder(nn.Module):
    def __init__(self):
        super(VoxelDecoder, self).__init__()
        
        # layer 1
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(4),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, encoded_feat, debug=False):
        """
        This function takes in the encoded feature and returns the voxel prediction
        
        @param encoded_feat: b x 512
        @param debug: boolean
        
        @return out: b x 32 x 16 x 16 x 16
        """
        # Input: b x 512
        # Output: b x 32 x 32 x 32
        # size: 64 x 2 x 2 x 2
        out1 = self.layer1(encoded_feat.view(-1, 64, 2, 2, 2))
        # size: 32 x 4 x 4 x 4
        out2 = self.layer2(out1)
        # size: 16 x 8 x 8 x 8
        out3 = self.layer3(out2)
        # size: 8 x 16 x 16 x 16
        out4 = self.layer4(out3)
        # size: 1 x 32 x 32 x 32
        out5 = self.layer5(out4)
        
        if debug:
            print("Layer 1: ", out1.shape)
            print("Layer 2: ", out2.shape)
            print("Layer 3: ", out3.shape)
            print("Layer 4: ", out4.shape)
            print("Layer 5: ", out5.shape)
            
        return out5

class PointCloudDecoder(nn.Module):
    def __init__(self, n_points):
        super(PointCloudDecoder, self).__init__()
        self.n_points = n_points
        
        self.layer1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(2048, n_points*3),
            nn.ReLU()
        ) 
        
    def forward(self, encoded_feat, debug=False):
        """
        This function takes in the encoded feature and returns the point cloud prediction
        
        @param encoded_feat: b x 512
        @param debug: boolean
        
        @return out: b x n_points x 3
        """
        # Input: b x 512
        # Output: b x n_points x 3
        out1 = self.layer1(encoded_feat)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        
        if debug:
            print("Layer 1: ", out1.shape)
            print("Layer 2: ", out2.shape)
            print("Layer 3: ", out3.shape)
        return out3
        
class MeshDecoder(nn.Module):
    def __init__(self, device, out_shape):
        super(MeshDecoder, self).__init__()
        self.device = device
        self.out_shape = out_shape

        self.layer1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(2048, self.out_shape),
            nn.Tanh()
        )
        # tanh activation
    
    def forward(self, encoded_feat, debug=False):
        """
        This function takes in the encoded feature and returns the mesh prediction
        
        @param encoded_feat: b x 512
        @param debug: boolean
        
        @return out: b x mesh_pred.verts_packed().shape[0] x 3
        """
        # Input: b x 512
        # Output: b x mesh_pred.verts_packed().shape[0] x 3
        out1 = self.layer1(encoded_feat)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        
        if debug:
            print("Layer 1: ", out1.shape)
            print("Layer 2: ", out2.shape)
            print("Layer 3: ", out3.shape)
        
        return out3     

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # TODO:
            self.decoder = VoxelDecoder() 
                        
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = PointCloudDecoder(self.n_point)
            
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            out_shape = 3*mesh_pred.verts_packed().shape[0]
            self.decoder = MeshDecoder(self.device, out_shape)

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat)             
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)       
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

