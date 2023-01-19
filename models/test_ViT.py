import torch
from thop import profile
from torchstat import stat
from vit_model import *

vit_pose_model = ViT_Pose_Model(img_size=368,patch_size=16,embed_dim=512,depth=12, num_heads=8,use_deconv=False).cuda()
x_in = torch.ones(size=(8,3,368,368)).cuda()
x_out = vit_pose_model(x_in)
print(x_out.shape)

flops, params = profile(vit_pose_model, inputs=(x_in, ))
print('flops:{}'.format(flops/1e9))
print('params:{}'.format(params/1e6))
#
# from torchsummary import summary
# summary(vit_pose_model, input_size=(3, 368, 368), batch_size=-1)