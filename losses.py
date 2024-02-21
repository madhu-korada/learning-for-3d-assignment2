import torch
import pytorch3d
import pytorch3d.loss

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# implement some loss for binary voxel grids
	
 	# Binary cross entropy loss
	BCE_loss = torch.nn.BCELoss()
	loss = BCE_loss(voxel_src, voxel_tgt)	
	return loss

def chamfer_loss(point_cloud_src, point_cloud_tgt, debug=False):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# implement chamfer loss from scratch

	# If the point clouds are not in the right shape, reshape them
	# Fix for mesh vs point cloud
	if point_cloud_src.shape != point_cloud_tgt.shape:
		point_cloud_src = point_cloud_src.reshape(-1, point_cloud_src.shape[-1]//3, 3)
  
	len_src = torch.full((point_cloud_src.shape[0],), point_cloud_src.shape[1], dtype=torch.int64, device=point_cloud_src.device)
	len_tgt = torch.full((point_cloud_tgt.shape[0],), point_cloud_tgt.shape[1], dtype=torch.int64, device=point_cloud_tgt.device)
	# d_chamfer = sum_i(min||x_j - y_j||^2) + sum_j(||x_i - y_i||^2)
	# calculate pairwise distances
	# for each point in source, find the closest point in target
	knn_src = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, lengths1=len_src, lengths2=len_tgt, K=1)
	# for each point in target, find the closest point in source
	knn_tgt = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, lengths1=len_tgt, lengths2=len_src, K=1)
	# sum the distances and return
	src_dists = knn_src.dists[..., 0]  # (N, S)
	tgt_dists = knn_tgt.dists[..., 0]  # (N, S)
	
 	# we can do mean because the number of points is same
	loss_chamfer = src_dists.mean() + tgt_dists.mean()
	if debug:
		chamfer_dist_p3d = pytorch3d.loss.chamfer_distance(point_cloud_src, point_cloud_tgt)
		print(f'Chamfer distance from pytorch3d: {chamfer_dist_p3d[0]}')
		print(f'Chamfer distance from scratch: {loss_chamfer}')
		assert torch.allclose(chamfer_dist_p3d[0], loss_chamfer)
  
	return loss_chamfer

def smoothness_loss(mesh_src, debug=True):	
	# implement laplacian smoothening loss
	# Compute the Laplacian of the vertices
	verts_laplacian = pytorch3d.loss.mesh_laplacian_smoothing(mesh_src)#, method="uniform")
	# The Laplacian smoothing loss is the mean squared norm of the Laplacian
	loss_laplacian = torch.mean(verts_laplacian ** 2) # TODO: check if this is correct

	return loss_laplacian