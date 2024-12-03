import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt 
import sys
from math import exp
from scipy.ndimage import label, find_objects
from scipy.ndimage import label, find_objects
from PIL import Image
import torchvision.transforms as transforms
import scipy.stats as st
import torchvision.transforms as T
def gauss_kernel(x, kernlen=3, nsig=3):

    if not isinstance(x, torch.Tensor):
        raise ValueError("Input x must be a PyTorch tensor.")

    device = x.device
    channels = x.shape[1]
    
    # Generate a 1D Gaussian kernel
    interval = (2 * nsig + 1.) / kernlen
    x_vals = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x_vals))
    
    # Create a 2D Gaussian kernel by taking the outer product of the 1D kernel
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()  # Normalize the kernel
    
    # Convert the kernel to a PyTorch tensor
    out_filter = torch.tensor(kernel, dtype=torch.float32, device=device)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))  # Reshape to 4D
    out_filter = out_filter.repeat(channels, 1, 1, 1)  # Repeat the kernel for all channels
    
    # Apply the Gaussian kernel to the input tensor using convolution
    return F.conv2d(x, out_filter, stride=1, padding=kernlen // 2, groups=channels)

class CustomBatchCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomBatchCrossEntropyLoss, self).__init__()

    def forward(self, input, target, ignore_classes):
        """
        Compute the loss with dynamic batch-wise ignore classes.

        Args:
            input (torch.Tensor): The model outputs with shape (N, C, H, W).
            target (torch.Tensor): The target labels with shape (N, H, W).
            ignore_classes (list of list): List of classes to ignore for each batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Compute the CrossEntropyLoss for each pixel
        criterion = nn.CrossEntropyLoss(reduction='none')
        # Note: CrossEntropyLoss expects input of shape (N, C, H, W) and target of shape (N, H, W)
        loss = criterion(input, target)

        # Prepare a tensor to accumulate the loss
        batch_size = target.size(0)
        total_loss = 0.0

        for b in range(batch_size):
            # Create a mask to ignore specific classes
            mask = ~torch.isin(target[b].view(-1), torch.tensor(ignore_classes[b], dtype=torch.long, device=target.device))
            # Apply the mask to the loss
            batch_loss = loss[b].view(-1)[mask]
            # Accumulate the loss
            total_loss += batch_loss.mean()

        return total_loss / batch_size

def convert_keep_to_ignore(num_classes, keep_classes_list):
    """
    Convert keep classes to ignore classes.

    Args:
        num_classes (int): The total number of classes.
        keep_classes_list (list of list): List of lists specifying classes to keep for each batch.

    Returns:
        ignore_classes (list of list): List of lists specifying classes to ignore for each batch.
    """
    ignore_classes = []
    for keep_classes in keep_classes_list:
        # Generate a list of all classes
        all_classes = set(range(num_classes))
        # Compute the set of classes to ignore
        ignore_classes_batch = list(all_classes - set(keep_classes))
        ignore_classes.append(ignore_classes_batch)
    
    return ignore_classes

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_classes):
        super(CustomCrossEntropyLoss, self).__init__()
        self.ignore_classes = ignore_classes

    def forward(self, input, target):
       
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(input, target)
        
        
        mask = ~torch.isin(target, torch.tensor(self.ignore_classes, dtype=torch.long, device=target.device))
        loss = loss[mask]
        
       
        return loss.mean()


class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()


class CLIP_MatchingLoss(nn.Module):
    def __init__(self, logit_scale):
        super().__init__()
        self.logit_scale = logit_scale

    def forward(self, image_features, text_features):
        # features is normalized
        if text_features.ndim > 2:
            text_features = rearrange(text_features, 'b c L -> b (c L)').contiguous()
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        logits = logits.softmax(dim=-1)
        
        n = logits.shape[1]
        labels = torch.arange(n).to(logits.device)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        return loss

class fusion_base_loss(nn.Module):
    def __init__(self):
        super(fusion_base_loss, self).__init__()
        self.loss_func_Int = L_intensity()
        self.loss_func_Grad = L_gradient()      
        self.loss_func_color = L_color()

    def forward(self, image_A, image_B, image_fused,int_ratio=1, grad_ratio=2, color_ratio=10):
     
        print('This is Base_model train loss')
        image_A_y,image_A_cbcr = self.rgb_to_y(image_A)
        image_B_y,image_B_cbcr = self.rgb_to_y(image_B)
        image_fuse_y,image_fuse_cbcr = self.rgb_to_y(image_fused)
        
        loss_int = int_ratio * F.l1_loss(image_fuse_y, torch.max(image_A_y,image_B_y))

        loss_grad = grad_ratio * self.loss_func_Grad(image_A_y, image_B_y, image_fuse_y)
            
        loss_color = color_ratio * self.loss_func_color(image_A, image_fused)

        total_loss = 10*(loss_int+loss_grad+loss_color)
        print('totoal',total_loss,'loss_grad',loss_grad,'loss_color',loss_color)
        
        
        return total_loss, loss_int,loss_grad, loss_color

    def rgb_to_y(self, image):
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b

        cb = 0.564 * (b - y)

        cr = 0.713 * (r - y)
    

        cbcr = torch.cat((cb, cr), dim=1)
    
        return y, cbcr


class fusion_modulated_loss(nn.Module):
    def __init__(self):
        super(fusion_modulated_loss, self).__init__()
        self.loss_func_Int = L_intensity()
        self.loss_func_Grad = L_gradient()      
        self.loss_func_color = L_color()
        self.ce_loss = CustomBatchCrossEntropyLoss()
    def forward(self, image_A, image_B, image_fused,g_label,image_Seg_Label,Fusion_Base,text_seg_list, int_ratio=1, grad_ratio=2, color_ratio=10):
        if torch.max(image_Seg_Label)==0:
            print('This is base loss',torch.max(image_Seg_Label),image_Seg_Label.shape)
            image_A_y,image_A_cbcr = self.rgb_to_y(image_A)
            image_B_y,image_B_cbcr = self.rgb_to_y(image_B)
            image_fuse_y,image_fuse_cbcr = self.rgb_to_y(image_fused)
            Fusion_Base_y,Fusion_Base_cbcr = self.rgb_to_y(Fusion_Base)
        
            loss_int = int_ratio * F.l1_loss(image_fuse_y, Fusion_Base_y)

            loss_grad = grad_ratio * self.loss_func_Grad(Fusion_Base_y, Fusion_Base_y, image_fuse_y)
            
            loss_color = color_ratio * self.loss_func_color(Fusion_Base, image_fused)

            total_loss = 10*(loss_int+loss_grad+loss_color)
            print('totoal',total_loss,'loss_grad',loss_grad,'loss_color',loss_color)
        
        
            return total_loss
        else:
        
            print('This is with enchance loss',torch.max(image_Seg_Label),image_Seg_Label.shape)
            image_A_y,image_A_cbcr = self.rgb_to_y(image_A)
            image_B_y,image_B_cbcr = self.rgb_to_y(image_B)
            image_fuse_y,image_fuse_cbcr = self.rgb_to_y(image_fused)
            Fusion_Base_y,Fusion_Base_cbcr = self.rgb_to_y(Fusion_Base)
            text_seg_list=convert_keep_to_ignore(9, text_seg_list)    
            seg_mask = image_Seg_Label > 0
            non_seg_mask = image_Seg_Label == 0
            seg_mask=seg_mask
            seg_mask3=(image_Seg_Label > 0).float().expand(-1, 3, -1, -1) 
            seg_mask31=(image_Seg_Label > 0).expand(-1, 3, -1, -1) 
            non_seg_mask3=(image_Seg_Label == 0).expand(-1, 3, -1, -1) 
            non_seg_mask = non_seg_mask
            print(g_label.shape,image_Seg_Label.squeeze(1).shape,g_label.dtype,image_Seg_Label.dtype)
            seg_loss = self.ce_loss(g_label, image_Seg_Label.squeeze(1).long(),text_seg_list) 
            
            loss_int1 = int_ratio*F.l1_loss(image_fuse_y[non_seg_mask],Fusion_Base_y[non_seg_mask]) 
            loss_int2=int_ratio*F.l1_loss((image_fuse_y[seg_mask]-torch.mean(image_fuse_y[non_seg_mask])),torch.max((image_A_y[seg_mask]-torch.mean(image_A_y[non_seg_mask])),(image_B_y[seg_mask]-torch.mean(image_B_y[non_seg_mask]))))
           
            loss_int=loss_int1+loss_int2
            
            #Fusion_Base_y[seg_mask]=((Fusion_Base_y - gauss_kernel(Fusion_Base_y))* 25 +gauss_kernel(Fusion_Base_y))[seg_mask]
            
            
            loss_grad1 = grad_ratio * self.loss_func_Grad(Fusion_Base_y, Fusion_Base_y, image_fuse_y,non_seg_mask)
            loss_grad2 = grad_ratio*F.l1_loss((image_fuse_y-gauss_kernel(image_fuse_y) )[seg_mask],((Fusion_Base_y-gauss_kernel(Fusion_Base_y))*25 )[seg_mask])
            
            loss_grad=loss_grad1+loss_grad2
            
            
            loss_color = color_ratio * self.loss_func_color(image_A, image_fused)
            total_loss = 10*(loss_int+loss_grad+loss_color)+seg_loss
            print('totoal',total_loss,'loss_grad',loss_grad,loss_grad1,loss_grad2,'loss_color',loss_color,'loss_int',loss_int,loss_int1,loss_int2,seg_loss)
        
            return total_loss

    def rgb_to_y(self, image):
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b

        cb = 0.564 * (b - y)

        cr = 0.713 * (r - y)
    

        cbcr = torch.cat((cb, cr), dim=1)
    
        return y, cbcr
##### Color loss by Chrominance Component: Cb Cr 
class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_X, image_fused):   
        ############### get the color vector ###################
        X_color_V = F.normalize(image_X, p=2.0, dim=1, eps=1e-12) 
        Fuse_color_V = F.normalize(image_fused, p=2.0, dim=1, eps=1e-12) 
    
        loss_color = torch.mean((1 - torch.sum(Fuse_color_V * X_color_V, dim=1)))

        return loss_color



###### Intensity Loss by select the most significant pixel intensity 
class L_intensity(nn.Module):
    def __init__(self):
        super(L_intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused, max_mode="l1"):
        A_sal = self.saliency_com(image_A)
        B_sal = self.saliency_com(image_B)
        
        concat_sal = torch.cat([A_sal/0.1, B_sal/0.1], dim=1)
        Weight_map = F.softmax(concat_sal, dim=1)
        
        W_a = Weight_map[:, :1, :, :] 
        W_b = Weight_map[:, 1:, :, :]
        
        #W_a = A_sal/(A_sal+B_sal)
        
        int_tar = W_a * image_A  + W_b * image_B
        
        if max_mode == "l1":
            Loss_intensity = F.l1_loss(int_tar, image_fused)
        else:
            Loss_intensity = F.mse_loss(int_tar, image_fused)
        return Loss_intensity

    def saliency_com(self, image):
        B, C, H, W = image.size()
        
        single_channel = torch.mean(image, dim=1, keepdim=True)
        mean_com = torch.mean(single_channel, dim=(1, 2, 3), keepdim=True)
        
        diff = torch.abs(image - mean_com)
        diff_min_init, _ = torch.min(diff.view(B, -1), dim=-1, keepdim=True)
        diff_min = diff_min_init.unsqueeze(-1).unsqueeze(-1)
        diff_max_init, _ = torch.max(diff.view(B, -1), dim=-1, keepdim=True)
        diff_max = diff_max_init.unsqueeze(-1).unsqueeze(-1)
        diff_norm = (diff - diff_min) / (diff_max - diff_min)
        
        return diff_norm






##### Gradient Loss by select the most significant pixel gradient 
class L_gradient(nn.Module):
    def __init__(self):
        super(L_gradient, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse,non_seg_mask=None):    
        if  non_seg_mask==None:
            gradient_A_x, gradient_A_y = self.gradient(image_A)
        
            gradient_B_x, gradient_B_y = self.gradient(image_B)
    
            gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)

            loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
            return loss
        else:
            gradient_A_x, gradient_A_y = self.gradient(image_A)
        
            gradient_B_x, gradient_B_y = self.gradient(image_B)
    
            gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)

            loss = F.l1_loss(gradient_fuse_x[non_seg_mask], torch.max(gradient_A_x[non_seg_mask], gradient_B_x[non_seg_mask])) + F.l1_loss(gradient_fuse_y[non_seg_mask], torch.max(gradient_A_y[non_seg_mask], gradient_B_y[non_seg_mask]))
            return loss      

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)



#class L_gradient(nn.Module):
#    def __init__(self):
#        super(L_gradient, self).__init__()
#        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
#                                                       [-2, 0, 2],
#                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
#        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
#                                                       [0, 0, 0],
#                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
#        self.padding = (1, 1, 1, 1)
#
#    def forward(self, image_A, image_B, image_fuse):     
#        gradient_A = self.gradient(image_A)
#        
#        gradient_B = self.gradient(image_B)
#    
#        gradient_fuse = self.gradient(image_fuse)
#
#        loss = F.l1_loss(gradient_fuse, torch.max(gradient_A, gradient_B))
#        return loss
#
#    def gradient(self, image):
#        image = F.pad(image, self.padding, mode='replicate')
#        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
#        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
#        return torch.abs(gradient_x)+torch.abs(gradient_y)
        


###get int loss#################################################
def sum_connected_components(components_list):

    if not components_list:
        raise ValueError("1")
    
   
    tensor_size = components_list[0][0].shape
    combined_tensor = torch.zeros(tensor_size, dtype=torch.float32)
    num=0
    for sublist in components_list:
        if num==0:
            num=num+1
        else:
            for tensor in sublist:
                combined_tensor=combined_tensor.to(tensor.device)
                combined_tensor += tensor
    
    return combined_tensor
def generate_minimum_bounding_box(mask):

    mask_np = mask.cpu().numpy()
    labeled_array, num_features = label(mask_np)
    slices = find_objects(labeled_array)
    B,C, H, W = mask.shape
   
    if slices is not None:
        min_rect_mask = np.zeros_like(mask_np)
        for s in slices:
            
            min_row, min_col = s[2].start, s[3].start
            max_row, max_col = s[2].stop - 1, s[3].stop - 1
            
                # Expand the bounding box by 1 pixel in all directions
            min_row = max(min_row - 1, 0)
            min_col = max(min_col - 1, 0)
            max_row = min(max_row + 1, H - 1)
            max_col = min(max_col + 1, W - 1)
                
                # Create the expanded bounding box mask
            min_rect_mask[:,:, min_row:max_row+1, min_col:max_col+1] = 1

        return torch.tensor(min_rect_mask, dtype=torch.float)
    return mask

def compute_background_mean(image, min_rect_mask, mask_n):

    background = (min_rect_mask - mask_n) > 0
    return torch.mean(image[background])

def compute_difference(image, mask_n, background_mean):
   
    masked_image = image * mask_n
    return torch.abs(masked_image - background_mean)

def compare_and_select_components(masks, image_A_y, image_B_y):
    
    extract_connected_components_A = []
    extract_connected_components_B = []

    for mask_list in masks:
        components_A = []
        components_B = []

        for mask_n in mask_list:
            mask_n=mask_n.unsqueeze(1).to(image_A_y.device)
            min_rect_mask = generate_minimum_bounding_box(mask_n).to(image_A_y.device)
            
            mean_A = compute_background_mean(image_A_y, min_rect_mask, mask_n)
            mean_B = compute_background_mean(image_B_y, min_rect_mask, mask_n)
            

            diff_A = compute_difference(image_A_y, mask_n, mean_A)
            diff_B = compute_difference(image_B_y, mask_n, mean_B)
            
           
            mask_A_n = mask_n * (diff_A > diff_B).float().to(image_A_y.device)
            mask_B_n = (mask_n - mask_A_n).to(image_A_y.device)

            components_A.append(mask_A_n)
            components_B.append(mask_B_n)
        
        extract_connected_components_A.append(components_A)
        extract_connected_components_B.append(components_B)
    
    return extract_connected_components_A, extract_connected_components_B




def extract_connected_components(masks):

    all_components = []

    for mask in masks:
       
        mask_np = mask.cpu().numpy()
        
       
        labeled_array, num_features = label(mask_np)
        

        components = []
        
        for feature in range(1, num_features + 1):
         
            component = np.zeros_like(mask_np)
            component[labeled_array == feature] = 1
            
          
            components.append(torch.tensor(component, dtype=torch.float))
        
       
        all_components.append(components)
    
    return all_components

def generate_masks_from_labels(image_Seg_Label):

    unique_values = torch.unique(image_Seg_Label)
    
    masks = [(image_Seg_Label == value).float() for value in unique_values]
    
    return masks

def save_tensor_as_image(tensor, file_path):

    if tensor.dim() != 3 or tensor.size(0) != 1:
        raise ValueError("Tensor must have shape (1, H, W)")
    
    
    image_array = tensor.squeeze().cpu().numpy()
    
    
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0
    image_array = image_array.astype(np.uint8)
    
    
    image = Image.fromarray(image_array)
    
    
    image.save(file_path)


def loss_int_compass(image_Seg_Label,image_A_y,image_B_y,image_fused_y):
    total_loss=0
    for b in range(image_Seg_Label.shape[0]):
        all_mask_binary = (image_Seg_Label>0).float()
        masks = generate_masks_from_labels(image_Seg_Label[b:b+1,:,:])
        connected_components = extract_connected_components(masks)
        extract_connected_components_A, extract_connected_components_B = compare_and_select_components(connected_components, image_A_y[b:b+1,:,:,:], image_B_y[b:b+1,:,:,:])
        combined_tensor_A = sum_connected_components(extract_connected_components_A).to(image_A_y.device)
        combined_tensor_B = sum_connected_components(extract_connected_components_B).to(image_A_y.device)
        co_A_B=combined_tensor_A*image_A_y[b:b+1,:,:,:]+combined_tensor_B*image_B_y[b:b+1,:,:,:]
        
        #save_tensor_as_image(all_mask_binary[b:b+1,:,:], '1.png')
        #save_tensor_as_image(combined_tensor_A.squeeze(1), '2.png')
        #save_tensor_as_image(combined_tensor_B.squeeze(1), '3.png')
        if torch.mean(image_Seg_Label[b:b+1,:,:].float())==0:
            loss=0
        else:
            loss=F.l1_loss(image_fused_y[b:b+1,:,:,:][(image_Seg_Label[b:b+1,:,:]>0).unsqueeze(1)],co_A_B[(image_Seg_Label[b:b+1,:,:]>0).unsqueeze(1)])
        #print('loss',loss)
        total_loss=total_loss+loss
    return total_loss/image_Seg_Label.shape[0]