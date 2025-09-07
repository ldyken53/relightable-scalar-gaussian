import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import rotation_to_quaternion, quaternion_multiply, pcast_i16_to_f32
from utils.sh_utils import RGB2SH, eval_sh
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from arguments import OptimizationParams
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from diff_gaussian_rasterization._C import kmeans_cuda
from collections import OrderedDict
class Codebook():
    def __init__(self, ids, centers):
        self.ids = ids
        self.centers = centers
    
    def evaluate(self):
        return self.centers[self.ids.flatten().long()]

def generate_codebook(values, inverse_activation_fn=lambda x: x, num_clusters=256, tol=0.0001):
    shape = values.shape
    values = values.flatten().view(-1, 1)
    centers = values[torch.randint(values.shape[0], (num_clusters, 1), device="cuda").squeeze()].view(-1,1)

    ids, centers = kmeans_cuda(values, centers.squeeze(), tol, 500)
    ids = ids.byte().squeeze().view(shape)
    centers = centers.view(-1,1)

    return Codebook(ids.cuda(), inverse_activation_fn(centers.cuda()))

class ScalarGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        # self.normal_activation = torch.nn.functional.normalize
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.scalar_activation = torch.sigmoid
        self.inverse_scalar_activation = inverse_sigmoid

        if self.use_phong:
            self.diffuse_factor_activation = torch.sigmoid
            self.shininess_activation = torch.sigmoid
            self.ambient_activation = torch.sigmoid
            self.specular_factor_activation = torch.sigmoid

    def __init__(self, render_type='render', TFscalar=0.5):
        print(TFscalar)
        self.render_type = render_type
        self.use_phong = render_type in ['phong']
        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)  # normal
        self._scaling = torch.empty(0)
        self._scaling_q = torch.empty(0)
        self._rotation = torch.empty(0) # quaternion 4-dim
        self._rotation_q = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.normal_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._scalar = torch.empty(0)
        self._scalar2 = torch.empty(0)
        self.TFscalar = TFscalar

        self.setup_functions()
        self.transform = {}

        if self.use_phong:
            self._diffuse_factor = torch.empty(0) 
            self._shininess = torch.empty(0) 
            self._ambient_factor = torch.empty(0) 
            self._specular_factor = torch.empty(0)
            self._codebook_dict = None

        #*for composing
        self._num_GSs_TF = [-1]
        
    def set_num_GSs_TF(self, num_GSs_TF):
        self._num_GSs_TF = num_GSs_TF
        
    @torch.no_grad()
    def set_transform(self, rotation=None, center=None, scale=None, offset=None, transform=None):
        if transform is not None:
            scale = transform[:3, :3].norm(dim=-1)

            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
            xyz_homo = torch.cat([self._xyz.data, torch.ones_like(self._xyz[:, :1])], dim=-1)
            self._xyz.data = (xyz_homo @ transform.T)[:, :3]
            rotation = transform[:3, :3] / scale[:, None]
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
            return

        if center is not None:
            self._xyz.data = self._xyz.data - center
        if rotation is not None:
            self._xyz.data = (self._xyz.data @ rotation.T)
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
        if scale is not None:
            self._xyz.data = self._xyz.data * scale
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
        if offset is not None:
            self._xyz.data = self._xyz.data + offset
    
    

    def capture(self):
        captured_list = [
            self._xyz,
            self._normal,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.normal_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._scalar,
            self._scalar2
        ]
        if self.use_phong:
            captured_list.extend([
                self._diffuse_factor, # 1
                self._shininess, # 1
                self._ambient_factor, # 1
                self._specular_factor
            ])

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        (self._xyz,
         self._normal,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         self._scalar,
         self._scalar2) = model_args[:13]
        if len(model_args) > 13 and self.use_phong:
            (self._diffuse_factor,
             self._shininess,
             self._ambient_factor,
             self._specular_factor) = model_args[13:]

        if is_training:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.normal_gradient_accum = normal_gradient_accum
            self.denom = denom
            if restore_optimizer:
                # TODO automatically match the opt_dict
                try:
                    self.optimizer.load_state_dict(opt_dict)
                except:
                    pass
    
    @property
    def get_num_GSs_TF(self):
        return self._num_GSs_TF

    @property
    def get_gaussian_nums(self):
        return self._xyz.shape[0]
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_diffuse_factor(self):
        # return self.diffuse_factor_activation(self._diffuse_factor)
        return self._diffuse_factor

    @property
    def get_shininess(self):
        return self.shininess_activation(self._shininess)*50 # for high specular part
        # return self._shininess
    
    @property
    def get_ambient_factor(self):
        # return self.ambient_activation(self._ambient_factor)
        return self._ambient_factor
    
    @property
    def get_specular_factor(self):
        return self.specular_factor_activation(self._specular_factor)*5 # for high specular part
        # return self._specular_factor # original
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_normal(self):
        return self.normal_activation(self._normal)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scalar(self):
        return self.scalar_activation(self._scalar)

    @property
    def get_scalar2(self):
        return self.scalar_activation(self._scalar2)
    

    def get_by_names(self, names):
        if len(names) == 0:
            return None
        fs = []
        for name in names:
            fs.append(getattr(self, "get_" + name))
        return torch.cat(fs, dim=1)

    def split_by_names(self, features, names):
        results = {}
        last_idx = 0
        for name in names:
            current_shape = getattr(self, "_" + name).shape[1]
            results[name] = features[last_idx:last_idx + current_shape]
            last_idx += getattr(self, "_" + name).shape[1]
        return results

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling,
                                          scaling_modifier,
                                          self.get_rotation)

    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self.get_rotation)

    @property
    def attribute_names(self):
        attribute_names = ['xyz', 'normal', 'scaling', 'rotation', 'opacity', 'scalar', 'scalar2']
        if self.use_phong:
            attribute_names.extend(['diffuse_factor', 'shininess', "ambient_factor", "specular_factor"])
        return attribute_names

    
    @classmethod
    def create_from_gaussians(cls, gaussians_list, dataset):
        assert len(gaussians_list) > 0
        gaussians = ScalarGaussianModel(render_type=gaussians_list[0].render_type)
        attribute_names = gaussians.attribute_names
        num_GSs_TF = [g.get_gaussian_nums for g in gaussians_list]
        # ic("gaussians_nums", num_GSs_TF)
        for attribute_name in attribute_names:
            setattr(gaussians, "_" + attribute_name,
                    nn.Parameter(torch.cat([getattr(g, "_" + attribute_name).data for g in gaussians_list],
                                           dim=0).requires_grad_(True)))
        gaussians.set_num_GSs_TF(num_GSs_TF) # set the number of GSs in each TF, used for editing control
        # ic("gaussians_nums", gaussians.get_num_GSs_TF) 
        # exit()
        return gaussians

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path, weights_only=False)

        (self._xyz,
         self._normal,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         self._scalar,
        self._scalar2) = model_args[:13]

        self.xyz_gradient_accum = xyz_gradient_accum
        self.normal_gradient_accum = normal_gradient_accum
        self.denom = denom

        if self.use_phong:
            if len(model_args) > 13:
                (self._diffuse_factor, 
                 self._shininess,
                 self._ambient_factor,
                 self._specular_factor) = model_args[13:]
            else: #* default init all zeros
                self._diffuse_factor = nn.Parameter((torch.zeros_like(self._xyz[..., :1])).requires_grad_(True))
                self._shininess = nn.Parameter(torch.ones_like(self._xyz[..., :1]).requires_grad_(True))
                self._ambient_factor = nn.Parameter(torch.zeros_like(self._xyz[..., :1]).requires_grad_(True))
                self._specular_factor = nn.Parameter((torch.zeros_like(self._xyz[..., :1])).requires_grad_(True))

        if restore_optimizer:
            # TODO automatically match the opt_dict
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        print(f"min and max: {fused_point_cloud.max()} {fused_point_cloud.min()}")
        fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(fused_normal.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # scalars = inverse_sigmoid(torch.clamp(self.TFscalar + (torch.rand((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") - 0.5) * 2 * 0.01, 
        #                                       0.0, 1.0))
        if pcd.values is not None:
            print("Dropout values given")
            scalars = inverse_sigmoid(
                torch.tensor(pcd.values, dtype=torch.float, device="cuda").reshape(-1, 1)
            )
        else:
            scalars = inverse_sigmoid((torch.rand((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")))
        self._scalar = nn.Parameter(scalars.clone().requires_grad_(True))
        self._scalar2 = nn.Parameter(scalars.clone().requires_grad_(True))

        if self.use_phong: #* default init all zeros
            diffuse_factor = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            shininess = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            ambient = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            specular = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

            self._diffuse_factor = nn.Parameter(diffuse_factor.requires_grad_(True))
            self._shininess = nn.Parameter(shininess.requires_grad_(True))
            self._ambient_factor = nn.Parameter(ambient.requires_grad_(True))
            self._specular_factor = nn.Parameter(specular.requires_grad_(True))
            
        
    def training_setup(self, training_args: OptimizationParams):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scalar], 'lr': training_args.scalar_lr, "name": "scalar"},
            {'params': [self._scalar2], 'lr': training_args.scalar_lr, "name": "scalar2"}
        ]

        if self.use_phong:
            l.extend([
                {'params': [self._diffuse_factor], 'lr': training_args.diffuse_factor_lr, "name": "diffuse_factor"},
                {'params': [self._shininess], 'lr': training_args.shininess_lr, "name": "shininess"},
                {'params': [self._ambient_factor], 'lr': training_args.ambient_lr, "name": "ambient_factor"},
                {'params': [self._specular_factor], 'lr': training_args.specular_lr, "name": "specular_factor"},
            ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self): #todo: when you save ply, elements should keep this order
        # l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l = ['x', 'y', 'z']
        l.append('opacity')
        for i in range(self._normal.shape[1]): 
            # ic('normal_{}'.format(i))
            l.append('normal_{}'.format(i)) # normal_0, normal_1, normal_2
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i)) # scale_0, scale_1, scale_2
        for i in range(self._rotation.shape[1]): 
            l.append('rot_{}'.format(i)) # rot_0, rot_1, rot_2, rot_3
        l.append('scalar')
        l.append('scalar2')
        if self.use_phong:
            l.append('diffuse_factor')
            l.append('shininess')
            l.append('ambient_factor') #* _ambient_factor
            l.append('specular_factor') #* _specular_factor
        return l

    def my_save_ply(self, path, quantised=False, half_float=False):
        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type
        mkdir_p(os.path.dirname(path))
        elements_list = []

        if quantised:
            # Read codebook dict to extract ids and centers
            if self._codebook_dict is None:
                print("Clustering codebook missing. Returning without saving")
                return

            opacity = self._codebook_dict["opacity"].ids
            scaling = self._codebook_dict["scaling"].ids
            normal = self._codebook_dict["normal"].ids #* quantised normal
            rot = torch.cat((self._codebook_dict["rotation_re"].ids,
                            self._codebook_dict["rotation_im"].ids),
                            dim=1)
            diffuse_factor = self._codebook_dict["diffuse_factor"].ids
            shininess = self._codebook_dict["shininess"].ids
            ambient_factor = self._codebook_dict["ambient_factor"].ids
            specular_factor = self._codebook_dict["specular_factor"].ids
            scalar = self._codebook_dict["scalar"].ids
            scalar2 = self._codebook_dict["scalar2"].ids
            
            dtype_full = [(k, float_type) for k in self._codebook_dict.keys()]
            # ic(dtype_full)
            # exit()
            codebooks = np.empty(256, dtype=dtype_full)

            centers_numpy_list = [v.centers.detach().cpu().numpy() for v in self._codebook_dict.values()]

            if half_float:
                # No float 16 for plydata, so we just pointer cast everything to int16
                for i in range(len(centers_numpy_list)):
                    centers_numpy_list[i] = np.asarray(centers_numpy_list[i], dtype=np.float16).view(dtype=np.int16)
                
            codebooks[:] = list(map(tuple, np.concatenate([ar for ar in centers_numpy_list], axis=1)))
                
        else:
            normal = self._normal #* quantised normal
            opacity = self._opacity
            scaling = self._scaling
            rot = self._rotation
            diffuse_factor = self._diffuse_factor
            shininess = self._shininess
            ambient_factor = self._ambient_factor
            specular_factor = self._specular_factor
            scalar = self._scalar
            scalar2 = self._scalar2

        #  Position&Normal is not quantised
        if half_float:
            xyz = self._xyz.detach().cpu().half().view(dtype=torch.int16).numpy()
            # normal = self._normal.detach().cpu().half().view(dtype=torch.int16).numpy()
        else:
            xyz = self._xyz.detach().cpu().numpy()
            # normal = self._normal.detach().cpu().numpy()

        opacities = opacity.detach().cpu().numpy()
        scaling = scaling.detach().cpu().numpy()
        normal = normal.detach().cpu().numpy()#* quantised normal
        rotation = rot.detach().cpu().numpy()
        diffuse_factor = diffuse_factor.detach().cpu().numpy()
        shininess = shininess.detach().cpu().numpy()
        ambient_factor = ambient_factor.detach().cpu().numpy()
        specular_factor = specular_factor.detach().cpu().numpy()
        scalar = scalar.detach().cpu().numpy()
        scalar2 = scalar2.detach().cpu().numpy()

        # dtype_full = [(attribute, float_type) 
        #                 if attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz'] else (attribute, attribute_type) 
        #                 for attribute in self.construct_list_of_attributes()]
        dtype_full = [(attribute, float_type) 
                        if attribute in ['x', 'y', 'z'] else (attribute, attribute_type) 
                        for attribute in self.construct_list_of_attributes()]
        gaussian_nums = xyz.shape[0]

        elements = np.empty(gaussian_nums, dtype=dtype_full)
        #!Note: the order need to be aligned with the order of construct_list_of_attributes
        attributes = np.concatenate((xyz, opacities, normal, scaling, rotation, scalar, scalar2, diffuse_factor, shininess, ambient_factor, specular_factor), axis=1)
        elements[:] = list(map(tuple, attributes))
        elements_list.append(PlyElement.describe(elements, f'gaussians'))
            
        if quantised:
            elements_list.append(PlyElement.describe(codebooks, f'codebook_centers'))
        PlyData(elements_list).write(path)
    
    
    #todo: this is new
    def _parse_codebook(self,vertex_group,
                            float_type,
                            attribute_type,
                            quantised,
                            half_precision,                            
                            codebook_centers_torch=None):
        num_primitives = vertex_group.count

        xyz = np.stack((np.asarray(vertex_group["x"], dtype=float_type),
                        np.asarray(vertex_group["y"], dtype=float_type),
                        np.asarray(vertex_group["z"], dtype=float_type)), axis=1)
        # normal = np.stack((np.asarray(vertex_group["nx"], dtype=float_type),
        #                    np.asarray(vertex_group["ny"], dtype=float_type),
        #                     np.asarray(vertex_group["nz"], dtype=float_type)), axis=1)

        opacity = np.asarray(vertex_group["opacity"], dtype=attribute_type)[..., np.newaxis]
    
        # Stacks the separate components of a vector attribute into a joint numpy array
        # Defined just to avoid visual clutter
        def stack_vector_attribute(name, count):
            return np.stack([np.asarray(vertex_group[f"{name}_{i}"], dtype=attribute_type)
                            for i in range(count)], axis=1)

        # features_dc = stack_vector_attribute("f_dc", 3).reshape(-1, 1, 3)
        scaling = stack_vector_attribute("scale", 3)
        normal = stack_vector_attribute("normal", 3) #* quantised normal
        rotation = stack_vector_attribute("rot", 4)
        diffuse_factor = np.asarray(vertex_group["diffuse_factor"], dtype=attribute_type)[..., np.newaxis]
        shininess = np.asarray(vertex_group["shininess"], dtype=attribute_type)[..., np.newaxis]
        ambient_factor = np.asarray(vertex_group["ambient_factor"], dtype=attribute_type)[..., np.newaxis]
        specular_factor = np.asarray(vertex_group["specular_factor"], dtype=attribute_type)[..., np.newaxis]
        scalar = np.asarray(vertex_group["scalar"], dtype=attribute_type)[..., np.newaxis]
        scalar2 = np.asarray(vertex_group["scalar2"], dtype=attribute_type)[..., np.newaxis]

        xyz = torch.from_numpy(xyz).cuda()
        # normal = torch.from_numpy(normal).cuda()
        if half_precision:
            xyz = pcast_i16_to_f32(xyz)
            # normal = pcast_i16_to_f32(normal)
  
        opacity = torch.from_numpy(opacity).cuda()
        scaling = torch.from_numpy(scaling).cuda()
        rotation = torch.from_numpy(rotation).cuda()
        normal = torch.from_numpy(normal).cuda() #* quantised normal
        diffuse_factor = torch.from_numpy(diffuse_factor).cuda()
        shininess = torch.from_numpy(shininess).cuda()
        ambient_factor = torch.from_numpy(ambient_factor).cuda()
        specular_factor = torch.from_numpy(specular_factor).cuda()
        scalar = torch.from_numpy(scalar).cuda()
        scalar2 = torch.from_numpy(scalar2).cuda()

        # If quantisation has been used, it is needed to index the centers
        if quantised:
            # This is needed as we might have padded the features_rest tensor with zeros before        
            # The gather operation indexes a 256x15 tensor with a (P*3)features_rest index tensor,
            # in a column-wise fashion
            # Basically this is equivalent to indexing a single codebook with a P*3 index
            # features_rest times inside a loop
            opacity = codebook_centers_torch['opacity'][opacity.long()]
            scaling = codebook_centers_torch['scaling'][scaling.view(num_primitives*3).long()].view(num_primitives, 3)
            normal = codebook_centers_torch['normal'][normal.view(num_primitives*3).long()].view(num_primitives, 3) #* quantised normal
            # Index the real and imaginary part separately
            rotation = torch.cat((
                codebook_centers_torch['rotation_re'][rotation[:, 0:1].long()],
                codebook_centers_torch['rotation_im'][rotation[:, 1:].reshape(num_primitives*3).long()].view(num_primitives,3)
                ), dim=1)
            diffuse_factor = codebook_centers_torch['diffuse_factor'][diffuse_factor.long()]
            shininess = codebook_centers_torch['shininess'][shininess.long()]
            ambient_factor = codebook_centers_torch['ambient_factor'][ambient_factor.long()]
            specular_factor = codebook_centers_torch['specular_factor'][specular_factor.long()]
            scalar = codebook_centers_torch['scalar'][scalar.long()]
            scalar2 = codebook_centers_torch['scalar2'][scalar2.long()]


        return {'xyz': xyz,
                'normal': normal,
                'opacity': opacity,
                'scaling': scaling,
                'rotation': rotation,
                'diffuse_factor': diffuse_factor,
                'shininess': shininess,
                'ambient_factor': ambient_factor,
                'specular_factor': specular_factor,
                'scalar': scalar,
                'scalar2': scalar2
        }
        
    def my_load_ply(self, path, half_float=False, quantised=False):
        plydata = PlyData.read(path)

        xyz_list = []
        normal_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        diffuse_factor_list = []
        shininess_list = []
        ambient_factor_list = []
        specular_factor_list = []
        scalar_list = []
        scalar2_list = []

        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type

        codebook_centers_torch = None
        if quantised:
            # Parse the codebooks.
            # The layout is 256 x 20, where 256 is the number of centers and 20 number of codebooks
            # In the future we could have different number of centers
            
            codebook_centers = plydata.elements[-1]

            codebook_centers_torch = OrderedDict()

            codebook_centers_torch['opacity'] = torch.from_numpy(np.asarray(codebook_centers['opacity'], dtype=float_type)).cuda()
            codebook_centers_torch['scaling'] = torch.from_numpy(np.asarray(codebook_centers['scaling'], dtype=float_type)).cuda()
            codebook_centers_torch['normal'] = torch.from_numpy(np.asarray(codebook_centers['normal'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_re'] = torch.from_numpy(np.asarray(codebook_centers['rotation_re'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_im'] = torch.from_numpy(np.asarray(codebook_centers['rotation_im'], dtype=float_type)).cuda()
            codebook_centers_torch['diffuse_factor'] = torch.from_numpy(np.asarray(codebook_centers['diffuse_factor'], dtype=float_type)).cuda()
            codebook_centers_torch['shininess'] = torch.from_numpy(np.asarray(codebook_centers['shininess'], dtype=float_type)).cuda()
            codebook_centers_torch['ambient_factor'] = torch.from_numpy(np.asarray(codebook_centers['ambient_factor'], dtype=float_type)).cuda()
            codebook_centers_torch['specular_factor'] = torch.from_numpy(np.asarray(codebook_centers['specular_factor'], dtype=float_type)).cuda()
            codebook_centers_torch['scalar'] = torch.from_numpy(np.asarray(codebook_centers['scalar'], dtype=float_type)).cuda()
            codebook_centers_torch['scalar2'] = torch.from_numpy(np.asarray(codebook_centers['scalar2'], dtype=float_type)).cuda()


            # If use half precision then we have to pointer cast the int16 to float16
            # and then cast them to floats, as that's the format that our renderer accepts
            if half_float:
                for k, v in codebook_centers_torch.items():
                    codebook_centers_torch[k] = pcast_i16_to_f32(v)

            # Iterate over the point clouds that are stored on top level of plyfile
            # to get the various fields values 
        attributes_dict = self._parse_codebook(plydata.elements[0],
                                                    float_type,
                                                    attribute_type,
                                                    quantised,
                                                    half_float,
                                                    codebook_centers_torch)

        xyz_list.append(attributes_dict['xyz'])
        normal_list.append(attributes_dict['normal']) #* quantised normal

        opacity_list.append(attributes_dict['opacity'])
        scaling_list.append(attributes_dict['scaling'])
        rotation_list.append(attributes_dict['rotation'])
        diffuse_factor_list.append(attributes_dict['diffuse_factor'])
        shininess_list.append(attributes_dict['shininess'])
        ambient_factor_list.append(attributes_dict['ambient_factor'])
        specular_factor_list.append(attributes_dict['specular_factor'])
        scalar_list.append(attributes_dict['scalar'])
        scalar2_list.append(attributes_dict['scalar2'])
        
        # Concatenate the tensors into one, to be used in our program
        xyz = torch.cat((xyz_list), dim=0)
        normal = torch.cat((normal_list), dim=0) #* quantised normal

        opacity = torch.cat((opacity_list), dim=0)
        scaling = torch.cat((scaling_list), dim=0)
        rotation = torch.cat((rotation_list), dim=0)
        diffuse_factor = torch.cat((diffuse_factor_list), dim=0)
        shininess = torch.cat((shininess_list), dim=0)
        ambient_factor = torch.cat((ambient_factor_list), dim=0)
        specular_factor = torch.cat((specular_factor_list), dim=0)
        scalar = torch.cat((scalar_list), dim=0)
        scalar2 = torch.cat((scalar2_list), dim=0)

        # normal_save = load_tensor('normal.pt')
        # scaling_save = load_tensor('scaling.pt')
        # rotation_save = load_tensor('rotation.pt')
        # offset_color_save = load_tensor('offset_color.pt')
        # diffuse_factor_save = load_tensor('diffuse_factor.pt')
        # shininess_save = load_tensor('shininess.pt')
        # ambient_factor_save = load_tensor('ambient_factor.pt')
        # specular_factor_save = load_tensor('specular_factor.pt')
        # ic(torch.allclose(normal, normal_save))
        # ic(torch.allclose(scaling, scaling_save))
        # ic(torch.allclose(rotation, rotation_save))
        # ic(torch.allclose(offset_color, offset_color_save))
        # ic(torch.allclose(diffuse_factor, diffuse_factor_save))
        # ic(torch.allclose(shininess, shininess_save))
        # ic(torch.allclose(ambient_factor, ambient_factor_save))
        # ic(torch.allclose(specular_factor, specular_factor_save))
        
        # ic(normal)
        # ic(normal_save)
        # ic(torch.allclose(normal, normal_save))
        # exit()
        
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True)) #* quantised normal
        
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._diffuse_factor = nn.Parameter(diffuse_factor.requires_grad_(True))
        self._shininess = nn.Parameter(shininess.requires_grad_(True))
        self._ambient_factor = nn.Parameter(ambient_factor.requires_grad_(True))
        self._specular_factor = nn.Parameter(specular_factor.requires_grad_(True))
        self._scalar = nn.Parameter(scalar.requires_grad_(True))
        self._scalar2 = nn.Parameter(scalar2.requires_grad_(True))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.normal_gradient_accum = self.normal_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._scalar = optimizable_tensors["scalar"]
        self._scalar2 = optimizable_tensors["scalar2"]

        if self.use_phong:
            self._diffuse_factor = optimizable_tensors["diffuse_factor"]
            self._shininess = optimizable_tensors["shininess"]
            self._ambient_factor = optimizable_tensors["ambient_factor"]
            self._specular_factor = optimizable_tensors["specular_factor"]
            

    def cat_tensors_to_optimizer(self, tensors_dict, source_indices, mode):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # ic(group["name"])
            # ic(extension_tensor.shape)
            # ic(group["params"][0].shape)
            # ic(stored_state["exp_avg"].shape)
            # ic(stored_state["exp_avg_sq"].shape)
            if stored_state is not None:
                if group["name"].startswith("scalar"):
                    exp_avg = stored_state["exp_avg"][source_indices].clone()
                    exp_avg_sq = stored_state["exp_avg_sq"][source_indices].clone()
                    if mode == "split":
                        exp_avg = exp_avg.repeat(2, 1)
                        exp_avg_sq = exp_avg_sq.repeat(2, 1)
                else:
                    exp_avg = torch.zeros_like(extension_tensor)
                    exp_avg_sq = torch.zeros_like(extension_tensor)
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]

                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_normal, new_opacities, new_scaling, new_rotation,
                              new_diffuse_factor=None, new_shininess=None, new_ambient_factor=None, 
                              new_specular_factor=None, new_scalar=None, new_scalar2=None, source_indices=None, mode=None):
        d = {
            "xyz": new_xyz,
            "normal": new_normal,
            "rotation": new_rotation,
            "scaling": new_scaling,
            "opacity": new_opacities,
            "scalar": new_scalar,
            "scalar2": new_scalar2
        }

        if self.use_phong:
            d.update({
                "diffuse_factor": new_diffuse_factor,
                "shininess": new_shininess,
                "ambient_factor": new_ambient_factor,
                "specular_factor": new_specular_factor
            })

        optimizable_tensors = self.cat_tensors_to_optimizer(d, source_indices, mode)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._rotation = optimizable_tensors["rotation"]
        self._scaling = optimizable_tensors["scaling"]
        self._opacity = optimizable_tensors["opacity"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._scalar = optimizable_tensors["scalar"]
        self._scalar2 = optimizable_tensors["scalar2"]

        if self.use_phong:
            self._diffuse_factor = optimizable_tensors["diffuse_factor"]
            self._shininess = optimizable_tensors["shininess"]
            self._ambient_factor = optimizable_tensors["ambient_factor"]
            self._specular_factor = optimizable_tensors["specular_factor"]


    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grad_normal = torch.zeros((n_init_points), device="cuda")
        padded_grad_normal[:grads_normal.shape[0]] = grads_normal.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask_normal = torch.where(padded_grad_normal >= grad_normal_threshold, True, False)
        # print("densify_and_split_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])

        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        # print("densify_and_split:", selected_pts_mask.sum().item(), "/", self.get_xyz.shape[0])

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)  # (N, 3)
        means = torch.zeros((stds.size(0), 3), device="cuda")  # (N, 3)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        kwargs = {
            "new_xyz": new_xyz,
            "new_normal": new_normal,
            "new_opacities": new_opacity,
            "new_scaling": new_scaling,
            "new_rotation": new_rotation,
        }

        kwargs["new_scalar"] = self._scalar[selected_pts_mask].repeat(N, 1)
        kwargs["new_scalar2"] = self._scalar2[selected_pts_mask].repeat(N, 1)
        kwargs["mode"] = "split"
        kwargs["source_indices"] = selected_pts_mask

        if self.use_phong:
            kwargs.update(
                new_diffuse_factor = self._diffuse_factor[selected_pts_mask].repeat(N, 1),
                new_shininess = self._shininess[selected_pts_mask].repeat(N, 1),
                new_ambient_factor = self._ambient_factor[selected_pts_mask].repeat(N, 1),
                new_specular_factor = self._specular_factor[selected_pts_mask].repeat(N, 1),
            )

        self.densification_postfix(**kwargs)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_normal = torch.where(torch.norm(grads_normal, dim=-1) >= grad_normal_threshold, True, False)
        # print("densify_and_clone_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        # print("densify_and_clone:", selected_pts_mask.sum().item(), "/", self.get_xyz.shape[0])

        new_xyz = self._xyz[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        kwargs = {
            "new_xyz": new_xyz,
            "new_normal": new_normal,
            "new_opacities": new_opacity,
            "new_scaling": new_scaling,
            "new_rotation": new_rotation,
        }
        kwargs["mode"] = "clone"
        kwargs["source_indices"] = selected_pts_mask

        kwargs["new_scalar"] = self._scalar[selected_pts_mask]
        kwargs["new_scalar2"] = self._scalar2[selected_pts_mask]

        if self.use_phong:
            kwargs.update(
                new_diffuse_factor = self._diffuse_factor[selected_pts_mask],
                new_shininess = self._shininess[selected_pts_mask],
                new_ambient_factor = self._ambient_factor[selected_pts_mask],
                new_specular_factor = self._specular_factor[selected_pts_mask],
            )

        self.densification_postfix(**kwargs)
    

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_normal):
        # print(self.xyz_gradient_accum.shape)
        grads = self.xyz_gradient_accum / self.denom
        grads_normal = self.normal_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_normal[grads_normal.isnan()] = 0.0

        # if self._xyz.shape[0] < 1000000:
        self.densify_and_clone(grads, max_grad, extent, grads_normal, max_grad_normal)
        self.densify_and_split(grads, max_grad, extent, grads_normal, max_grad_normal)
        # self.densify_and_compact()

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # prune_mask2 = (torch.max(self.get_scaling, dim=1).values < 0.001).squeeze()
        # prune_mask = torch.logical_or(prune_mask, prune_mask2)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.normal_gradient_accum[update_filter] += torch.norm(
            self.normal_activation(self._normal.grad)[update_filter], dim=-1,
            keepdim=True)
        self.denom[update_filter] += 1
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        attributes_list = [xyz, normal, opacities, scale, rotation]
        attributes_list.extend([
            self._scalar.detach().cpu().numpy(),
            self._scalar2.detach().cpu().numpy()
        ])
        if self.use_phong:
            attributes_list.extend([
                self._diffuse_factor.detach().cpu().numpy(),
                self._shininess.detach().cpu().numpy(),
                self._ambient_factor.detach().cpu().numpy(),
                self._specular_factor.detach().cpu().numpy()
            ])

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def produce_clusters(self, num_clusters=256, store_dict_path=None):
        codebook_dict = OrderedDict({})

        #* Basis GS attributes
        codebook_dict["opacity"] = generate_codebook(self.get_opacity.detach(),
                                                     self.inverse_opacity_activation, num_clusters=num_clusters)
        codebook_dict["scaling"] = generate_codebook(self.get_scaling.detach(),
                                                     self.scaling_inverse_activation, num_clusters=num_clusters)
        codebook_dict["rotation_re"] = generate_codebook(self.get_rotation.detach()[:, 0:1],
                                                         num_clusters=num_clusters)
        codebook_dict["rotation_im"] = generate_codebook(self.get_rotation.detach()[:, 1:],
                                                         num_clusters=num_clusters)
        codebook_dict["normal"] = generate_codebook(self.get_normal.detach(),
                                                    num_clusters=num_clusters)

        codebook_dict["scalar"] = generate_codebook(self.get_scalar.detach(), self.inverse_scalar_activation, 
                                                    num_clusters=num_clusters)
        codebook_dict["scalar2"] = generate_codebook(self.get_scalar2.detach(), self.inverse_scalar_activation, 
                                                    num_clusters=num_clusters)

        #* Phong attributes
        codebook_dict["diffuse_factor"] = generate_codebook(self.get_diffuse_factor.detach(),
                                                            num_clusters=num_clusters)
        codebook_dict["shininess"] = generate_codebook(self.get_shininess.detach(),
                                                      num_clusters=num_clusters)
        codebook_dict["ambient_factor"] = generate_codebook(self.get_ambient_factor.detach(),
                                                              num_clusters=num_clusters)
        codebook_dict["specular_factor"] = generate_codebook(self.get_specular_factor.detach(),
                                                                num_clusters=num_clusters)

        if store_dict_path is not None:
            torch.save(codebook_dict, os.path.join(store_dict_path, 'codebook.pt'))
        
        self._codebook_dict = codebook_dict
        
    #todo: this is new
    def apply_clustering(self, codebook_dict=None):
        if codebook_dict is None:
            return

        opacity = codebook_dict["opacity"].evaluate().requires_grad_(True)
        scaling = codebook_dict["scaling"].evaluate().view(-1, 3).requires_grad_(True)
        rotation = torch.cat((codebook_dict["rotation_re"].evaluate(),
                            codebook_dict["rotation_im"].evaluate().view(-1, 3)),
                            dim=1).squeeze().requires_grad_(True)
        normal = codebook_dict["normal"].evaluate().view(-1, 3).requires_grad_(True)
        
        scalar = codebook_dict["scalar"].evaluate().requires_grad_(True)
        scalar2 = codebook_dict["scalar2"].evaluate().requires_grad_(True)

        diffuse_factor = codebook_dict["diffuse_factor"].evaluate().requires_grad_(True)
        shininess = codebook_dict["shininess"].evaluate().requires_grad_(True)
        ambient_factor = codebook_dict["ambient_factor"].evaluate().requires_grad_(True)
        specular_factor = codebook_dict["specular_factor"].evaluate().requires_grad_(True)
        # save_tensor(normal, "normal.pt") 
        # save_tensor(opacity, "opacity.pt")
        # save_tensor(scaling, "scaling.pt")
        # save_tensor(rotation, "rotation.pt")
        # save_tensor(offset_color, "offset_color.pt")
        # save_tensor(diffuse_factor, "diffuse_factor.pt")
        # save_tensor(shininess, "shininess.pt")
        # save_tensor(ambient_factor, "ambient_factor.pt")
        # save_tensor(specular_factor, "specular_factor.pt")

        with torch.no_grad():
            self._opacity = opacity
            self._scaling = scaling
            self._rotation = rotation
            self._normal = normal
            self._scalar = scalar
            self._scalar2 = scalar2
            self._diffuse_factor = diffuse_factor
            self._shininess = shininess
            self._ambient_factor = ambient_factor
            self._specular_factor = specular_factor
    
    def freeze_attributes(self):
        self._opacity.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._normal.requires_grad_(False)
        self._scalar.requires_grad_(False)
        self._scalar2.requires_grad_(False)
        self._diffuse_factor.requires_grad_(False)
        self._shininess.requires_grad_(False)
        self._ambient_factor.requires_grad_(False)
        self._specular_factor.requires_grad_(False)

def save_tensor(tensor, path):
    torch.save(tensor, path)

def load_tensor(path):
    return torch.load(path)
