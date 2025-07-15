from .groundingdino.util.slconfig import SLConfig
from .groundingdino.models import build_model
from .groundingdino.util.utils import  get_phrases_from_posmap,clean_state_dict

import torch
import random
from PIL import Image


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    transform = T.Compose(
        [
            #T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def generation_promotlist_for_train(
    seg_label_tensor, 
    classes_list=["Background", "car", "person", "bikes", "Curve", "Car Stop", "Guardrail", "cone", "Bump"]
):
    """
    Generate a list of formatted prompts and target segment lists.

    Parameters:
        classes_list (list): List of class names with 'Background' as the first element.
        seg_label_tensor (torch.Tensor): 4D tensor of shape [batch_size, C, H, W] representing segmentation labels.

    Returns:
        tuple: (formatted_promot_list, text_seg_list)
    """
    # Ensure seg_label_tensor is a 4D tensor
    if not isinstance(seg_label_tensor, torch.Tensor) or len(seg_label_tensor.shape) != 4:
        raise ValueError("seg_label_tensor must be a 4D tensor with shape [batch_size, C, H, W].")
    
    # Define the specific target classes to select from
    specific_target_classes = [1, 2, 3, 7]  # Corresponding to Car, Person, Bike, Color Cone
    
    # Initialize lists
    text_seg_list = []
    formatted_promot_list = []
    
    batch_size = seg_label_tensor.shape[0]

    for b in range(batch_size):
        # Get unique pixel values for the current batch
        unique_values = torch.unique(seg_label_tensor[b].flatten()).tolist()

        # Convert to set for faster membership checking
        unique_values_set = set(unique_values)

        # Find available target classes
        available_target_classes = [c for c in specific_target_classes if c in unique_values_set]

        if available_target_classes:
            # Select only one class randomly from available target classes
            selected_class = random.choice(available_target_classes)

            # Create segment list with background (0) and selected class
            seg_list = [0, selected_class]  # Include background and selected class
        else:
            seg_list = [0]  # Only background

        text_seg_list.append(seg_list)

        # Generate formatted prompts based on selected classes
        if len(seg_list) > 1:
            selected_class_names = [classes_list[c] for c in seg_list[1:] if c < len(classes_list)]
            highlight_string = ' and '.join(selected_class_names)
        else:
            highlight_string = "\n"

        formatted_promot_list.append(highlight_string)
    return formatted_promot_list, text_seg_list



def generation_promotlist_for_train1(seg_label_tensor,classes_list = ["Background", "car", "person", "bikes","Curve", "Car Stop","Cuardrail", "cone", "Bump"], probability_list=[1500, 914, 1271, 582, 718, 371, 42, 344, 113], no_target_prob=0.05, multi_class_prob=0.3):
    """
    Generate a list of formatted prompts and target segment lists based on probabilities.

    Parameters:
        classes_list (list): List of class names with 'Background' as the first element.
        seg_label_tensor (torch.Tensor): 4D tensor of shape [batch_size, C, H, W] representing segmentation labels.
        probability_list (list): List of integers representing the probability inversely proportional to the values.
        no_target_prob (float): Probability of not selecting any target (i.e., returning background only).
        multi_class_prob (float): Probability of selecting two or more target classes.

    Returns:
        tuple: (formatted_promot_list, text_seg_list)
    """
    # Ensure seg_label_tensor is a 4D tensor
    if not isinstance(seg_label_tensor, torch.Tensor) or len(seg_label_tensor.shape) != 4:
        raise ValueError("seg_label_tensor must be a 4D tensor with shape [batch_size, C, H, W].")
    
    # Ensure classes_list is valid
    if not classes_list or len(classes_list) < 2:
        raise ValueError("classes_list must contain at least two classes: 'Background' and at least one target class.")

    # Ensure probability_list is valid
    if len(probability_list) != len(classes_list):
        raise ValueError("probability_list must have the same length as the number of classes (including background).")

    # Exclude the background class (0) from probability calculations
    target_classes = list(range(1, len(classes_list)))
    target_probabilities = probability_list[1:]

    # Compute inverse probabilities for target classes
    inverse_probabilities = [1 / (p + 1e-6) for p in target_probabilities]  # Adding small epsilon to avoid division by zero
    total_inverse_prob = sum(inverse_probabilities)
    probabilities = [inv_prob / total_inverse_prob for inv_prob in inverse_probabilities]

    # Adjust probabilities to include the "no target" option
    probabilities = [(1 - no_target_prob) * prob for prob in probabilities]
    probabilities.append(no_target_prob)  # Add the "no target" probability
    
    # Initialize lists
    text_seg_list = []
    formatted_promot_list = []

    batch_size = seg_label_tensor.shape[0]

    for b in range(batch_size):
        # Get unique pixel values for the current batch
        unique_values = torch.unique(seg_label_tensor[b].flatten()).tolist()

        # Convert to set for faster membership checking
        unique_values_set = set(unique_values)

        # Filter out the background class (0), keeping only target classes
        target_classes_set = set(target_classes)
        available_target_classes = list(target_classes_set & unique_values_set)

        # Add a special class ID for "no target" (i.e., background only)
        no_target_class_id = len(target_classes) + 1

        if available_target_classes:
            available_target_classes.append(no_target_class_id)  # Append the "no target" option

            # Map available target classes to their corresponding probabilities
            class_probs = [probabilities[target_classes.index(c)] if c in target_classes else no_target_prob for c in available_target_classes]

            # Randomly decide if multiple classes should be selected
            if random.random() < multi_class_prob and len(available_target_classes) > 1:
                # Select 2 or more classes (without repetition)
                num_selected_classes = random.randint(2, len(available_target_classes))
                selected_classes = random.sample(available_target_classes, k=num_selected_classes)
            else:
                # Select only one class
                selected_classes = random.choices(available_target_classes, weights=class_probs, k=1)

            if no_target_class_id in selected_classes:
                seg_list = [int(0)]  # Only background
            else:
                seg_list = [int(0)] + [int(c) for c in selected_classes if c != no_target_class_id]  # Include selected classes, exclude no target

        else:
            seg_list = [int(0)]  # Only background

        text_seg_list.append(seg_list)

        # Generate formatted prompts based on selected classes
        if len(seg_list) > 1:
            selected_class_names = [classes_list[c] for c in seg_list[1:] if c < len(classes_list)]
            highlight_string = ' and '.join(selected_class_names)
        else:
            highlight_string = "\n"

        formatted_promot_list.append(highlight_string)

    return formatted_promot_list[0], text_seg_list





def compute_dice_loss_with_textlist(self_Seg_Label, seg1, text_list):
    # Step 1: One-hot encode self_Seg_Label to shape (B, 9, H, W)
    B, H, W = self_Seg_Label.shape  # Extract shape information
    num_classes = seg1.shape[1]  # Assume seg1's second dimension is the number of classes (9)
    
    # One-hot encode self_Seg_Label
    self_Seg_Label_onehot = F.one_hot(self_Seg_Label, num_classes=num_classes)  # Shape: (B, H, W, 9)
    self_Seg_Label_onehot = self_Seg_Label_onehot.permute(0, 3, 1, 2)  # Shape: (B, 9, H, W)

    # Step 2: Sort text_list and extract corresponding layers
    sorted_text_list = [sorted(sublist) for sublist in text_list]  # Sort each sublist
    
    # Initialize lists to hold the extracted layers
    seg1_new = []
    self_Seg_Label_onehot_new = []

    # Step 3: Extract layers from seg1 and self_Seg_Label_onehot based on sorted_text_list
    for b in range(B):
        indices = sorted_text_list[b]  # Get the sorted indices for this batch element

        # Extract layers from seg1 and self_Seg_Label_onehot
        seg1_selected = seg1[b, indices, :, :]  # Shape: (4, H, W)
        self_Seg_Label_onehot_selected = self_Seg_Label_onehot[b, indices, :, :]  # Shape: (4, H, W)

        seg1_new.append(seg1_selected)
        self_Seg_Label_onehot_new.append(self_Seg_Label_onehot_selected)

    # Stack the extracted layers to get the final tensors
    seg1_new = torch.stack(seg1_new)  # Shape: (B, 4, H, W)
    self_Seg_Label_onehot_new = torch.stack(self_Seg_Label_onehot_new)  # Shape: (B, 4, H, W)

    # Step 4: Compute Dice Loss
    dice_loss_fn = DiceLoss(include_background=False, to_onehot_y=False, softmax=False)
    
    # Calculate the Dice loss between seg1_new and self_Seg_Label_onehot_new
    dice_loss_value = dice_loss_fn(seg1_new, self_Seg_Label_onehot_new)
    
    return dice_loss_value#, seg1_new, self_Seg_Label_onehot_new


    
    


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    image = image.to(device) 
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    #print(load_res)
    model = model.to(device) 
    _ = model.eval()
    return model    

def tensor2img(tensor):
    """ To change the decoded latent into real format image in [0, 1] """
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return img


def calculate_clip(image_features, text_features):
    if text_features.ndim > 2:
        text_features = rearrange(text_features, 'b c L -> b (c L)').contiguous()
    logits = image_features @ text_features.t()
    
    return logits