from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
from scipy import ndimage
from io import BytesIO
import io
import base64

# Load both models once
binary_model = load_model('./binary_classifier.keras')      # Classifies brain vs non-brain
tumor_model = load_model('./braintumor.keras')              # Classifies tumor type


# requirements classes

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):  # Changed from 16 to 8
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

# Enhanced Conv Block with Residual Connections and Dropout

def conv_block(in_channels, out_channels, kernel_size=3, activation=nn.ReLU(inplace=True), dropout_p=0.2):
    """Create a conv block with CBAM attention, residual connection, and dropout"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
        nn.BatchNorm2d(out_channels),
        activation,
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
        nn.BatchNorm2d(out_channels),
        activation,
        CBAM(out_channels, reduction_ratio=8),  # Reduced ratio for better attention
        nn.Dropout2d(p=dropout_p)  # Added spatial dropout for regularization
    )

# UNet++ with Enhanced Architecture

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, deep_supervision=True, n_filters=32):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Encoder (Downsampling Path)
        self.enc1 = conv_block(in_channels, 1 * n_filters, activation=self.activation, dropout_p=0.1)
        self.enc2 = conv_block(1 * n_filters, 2 * n_filters, activation=self.activation, dropout_p=0.1)
        self.enc3 = conv_block(2 * n_filters, 4 * n_filters, activation=self.activation, dropout_p=0.2)
        self.enc4 = conv_block(4 * n_filters, 8 * n_filters, activation=self.activation, dropout_p=0.2)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Center / Bottleneck
        self.center = conv_block(8 * n_filters, 16 * n_filters, activation=self.activation, dropout_p=0.3)

        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Nested decoder blocks (UNet++ architecture)
        # Level 1 skip connections (up from level 2)
        self.dec1_1 = conv_block(1 * n_filters + 2 * n_filters, 1 * n_filters, activation=self.activation, dropout_p=0.1)

        # Level 2 skip connections
        self.dec2_1 = conv_block(2 * n_filters + 4 * n_filters, 2 * n_filters, activation=self.activation, dropout_p=0.1)
        self.dec1_2 = conv_block(1 * n_filters + 2 * n_filters, 1 * n_filters, activation=self.activation, dropout_p=0.1)

        # Level 3 skip connections
        self.dec3_1 = conv_block(4 * n_filters + 8 * n_filters, 4 * n_filters, activation=self.activation, dropout_p=0.2)
        self.dec2_2 = conv_block(2 * n_filters + 4 * n_filters, 2 * n_filters, activation=self.activation, dropout_p=0.2)
        self.dec1_3 = conv_block(1 * n_filters + 2 * n_filters, 1 * n_filters, activation=self.activation, dropout_p=0.1)

        # Level 4 skip connections
        self.dec4_1 = conv_block(8 * n_filters + 16 * n_filters, 8 * n_filters, activation=self.activation, dropout_p=0.2)
        self.dec3_2 = conv_block(4 * n_filters + 8 * n_filters, 4 * n_filters, activation=self.activation, dropout_p=0.2)
        self.dec2_3 = conv_block(2 * n_filters + 4 * n_filters, 2 * n_filters, activation=self.activation, dropout_p=0.1)
        self.dec1_4 = conv_block(1 * n_filters + 2 * n_filters, 1 * n_filters, activation=self.activation, dropout_p=0.1)

        # Output layers for deep supervision
        if self.deep_supervision:
            self.output1 = nn.Conv2d(1 * n_filters, out_channels, kernel_size=1)
            self.output2 = nn.Conv2d(1 * n_filters, out_channels, kernel_size=1)
            self.output3 = nn.Conv2d(1 * n_filters, out_channels, kernel_size=1)
            self.output4 = nn.Conv2d(1 * n_filters, out_channels, kernel_size=1)
        else:
            self.output = nn.Conv2d(1 * n_filters, out_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Bottleneck
        bottle = self.center(self.pool(x4))

        # Decoder (UNet++ nested architecture)
        # Level 4
        x4_1 = self.dec4_1(torch.cat([x4, self.up(bottle)], dim=1))

        # Level 3
        x3_1 = self.dec3_1(torch.cat([x3, self.up(x4)], dim=1))
        x3_2 = self.dec3_2(torch.cat([x3, self.up(x4_1)], dim=1))

        # Level 2
        x2_1 = self.dec2_1(torch.cat([x2, self.up(x3)], dim=1))
        x2_2 = self.dec2_2(torch.cat([x2, self.up(x3_1)], dim=1))
        x2_3 = self.dec2_3(torch.cat([x2, self.up(x3_2)], dim=1))

        # Level 1
        x1_1 = self.dec1_1(torch.cat([x1, self.up(x2)], dim=1))
        x1_2 = self.dec1_2(torch.cat([x1, self.up(x2_1)], dim=1))
        x1_3 = self.dec1_3(torch.cat([x1, self.up(x2_2)], dim=1))
        x1_4 = self.dec1_4(torch.cat([x1, self.up(x2_3)], dim=1))

        # Deep supervision
        if self.deep_supervision:
            output1 = self.output1(x1_1)
            output2 = self.output2(x1_2)
            output3 = self.output3(x1_3)
            output4 = self.output4(x1_4)
            return [output1, output2, output3, output4]
        else:
            output = self.output(x1_4)
            return output

def preprocess_image(image_path, target_size=(240, 240)):
    """
    Preprocess a single MRI image for the brain tumor segmentation model
    """
    # Load image and convert to grayscale if needed
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')

    # Resize to target size
    img = img.resize(target_size)

    # Convert to numpy array and normalize to [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0

    # Stack the same image 4 times to simulate 4 MRI modalities
    # (T1, T1ce, T2, FLAIR) - in a real scenario, you'd have separate images
    img_stacked = np.stack([img_np] * 4)

    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_stacked).unsqueeze(0)

    return img_tensor

def run_inference(model, image_tensor, device, test_time_augmentation=True):
    """
    Run inference with optional test-time augmentation
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        if not test_time_augmentation:
            # Standard inference
            if model.deep_supervision:
                output = model(image_tensor)[-1]  # Take the last output if deep supervision
            else:
                output = model(image_tensor)
            return output

        # Test-time augmentation (TTA)
        # Original image
        if model.deep_supervision:
            pred_orig = model(image_tensor)[-1]
        else:
            pred_orig = model(image_tensor)

        # Horizontal flip
        x_hflip = torch.flip(image_tensor, dims=[3])
        if model.deep_supervision:
            pred_hflip = model(x_hflip)[-1]
        else:
            pred_hflip = model(x_hflip)
        pred_hflip = torch.flip(pred_hflip, dims=[3])

        # Vertical flip
        x_vflip = torch.flip(image_tensor, dims=[2])
        if model.deep_supervision:
            pred_vflip = model(x_vflip)[-1]
        else:
            pred_vflip = model(x_vflip)
        pred_vflip = torch.flip(pred_vflip, dims=[2])

        # Average predictions
        output = (pred_orig + pred_hflip + pred_vflip) / 3.0

        return output

def postprocess_prediction(pred_tensor):
    """
    Convert predictions to binary segmentation mask
    """
    # Apply sigmoid and threshold
    pred_sigmoid = torch.sigmoid(pred_tensor)
    binary_mask = (pred_sigmoid > 0.5).float()

    return binary_mask

def visualize_prediction(image, pred_mask):
    """
    Visualize prediction results with individual tumor regions
    - ET: Enhancing Tumor (orange)
    - ED: Edema (green)
    - NCR: Necrotic Core (blue)
    """
    if isinstance(image, torch.Tensor):
        # Use only one channel for display
        image = image[0, 0].cpu().numpy()

    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    tumor_classes = ['NCR', 'ED', 'ET']
    colors = [(0, 0, 1), (0, 1, 0), (1, 0.6, 0)]  # Blue, Green, Orange

    fig = plt.figure(figsize=(20, 6))

    # Original image
    ax_orig = plt.subplot2grid((1, 5), (0, 0))
    ax_orig.imshow(image, cmap='gray')
    ax_orig.set_title("Original")
    ax_orig.axis('off')

    # Individual regions
    for c, (class_name, color) in enumerate(zip(tumor_classes, colors)):
        ax = plt.subplot2grid((1, 5), (0, c+1))
        ax.imshow(image, cmap='gray')

        pred_region = pred_mask[0, c]
        overlay = np.zeros((*image.shape, 3), dtype=np.float32)
        for h in range(pred_region.shape[0]):
            for w in range(pred_region.shape[1]):
                if pred_region[h, w] > 0:
                    overlay[h, w] = color

        ax.imshow(overlay, alpha=0.6)
        ax.set_title(f"{class_name}")
        ax.axis('off')

    # Combined prediction
    ax_pred = plt.subplot2grid((1, 5), (0, 4))
    ax_pred.imshow(image, cmap='gray')
    overlay_pred = np.zeros((*image.shape, 3), dtype=np.float32)
    for c, color in enumerate(colors):
        mask = pred_mask[0, c]
        for h in range(mask.shape[0]):
            for w in range(mask.shape[1]):
                if mask[h, w] > 0:
                    overlay_pred[h, w] = color

    ax_pred.imshow(overlay_pred, alpha=0.6)
    ax_pred.set_title("Combined Prediction")
    ax_pred.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.2, top=0.9)
    plt.suptitle("Brain Tumor Segmentation (Single Model)", fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig

def compare_with_ensemble(image, single_pred, ensemble_pred):
    """
    Compare predictions from single model and ensemble
    """
    if isinstance(image, torch.Tensor):
        image = image[0, 0].cpu().numpy()

    if isinstance(single_pred, torch.Tensor):
        single_pred = single_pred.cpu().numpy()

    if isinstance(ensemble_pred, torch.Tensor):
        ensemble_pred = ensemble_pred.cpu().numpy()

    tumor_classes = ['NCR', 'ED', 'ET']
    colors = [(0, 0, 1), (0, 1, 0), (1, 0.6, 0)]  # Blue, Green, Orange

    fig = plt.figure(figsize=(15, 8))

    # Original image
    ax_orig = plt.subplot2grid((2, 4), (0, 0), rowspan=2)
    ax_orig.imshow(image, cmap='gray')
    ax_orig.set_title("Original MRI", fontsize=14)
    ax_orig.axis('off')

    # Single model predictions by region
    for i, (class_name, color) in enumerate(zip(tumor_classes, colors)):
        ax = plt.subplot2grid((2, 4), (0, i+1))
        ax.imshow(image, cmap='gray')

        pred_region = single_pred[0, i]
        overlay = np.zeros((*image.shape, 3), dtype=np.float32)
        for h in range(pred_region.shape[0]):
            for w in range(pred_region.shape[1]):
                if pred_region[h, w] > 0:
                    overlay[h, w] = color

        ax.imshow(overlay, alpha=0.6)
        ax.set_title(f"Single: {class_name}", fontsize=12)
        ax.axis('off')

    # Ensemble model predictions by region
    for i, (class_name, color) in enumerate(zip(tumor_classes, colors)):
        ax = plt.subplot2grid((2, 4), (1, i+1))
        ax.imshow(image, cmap='gray')

        pred_region = ensemble_pred[0, i]
        overlay = np.zeros((*image.shape, 3), dtype=np.float32)
        for h in range(pred_region.shape[0]):
            for w in range(pred_region.shape[1]):
                if pred_region[h, w] > 0:
                    overlay[h, w] = color

        ax.imshow(overlay, alpha=0.6)
        ax.set_title(f"Ensemble: {class_name}", fontsize=12)
        ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.suptitle("Single Model vs Ensemble Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig


def load_pretrained_model(checkpoint_path, device):
    """Load a single pretrained UNetPlusPlus model"""
    model = UNetPlusPlus(in_channels=4, out_channels=3, deep_supervision=True, n_filters=32)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def calculate_tumor_size(binary_mask, pixel_spacing=None):
    """Calculate tumor size metrics from binary segmentation mask"""
    tumor_metrics = {
        'ET': {'name': 'Enhancing Tumor', 'color': (1, 0.6, 0)},
        'ED': {'name': 'Edema', 'color': (0, 1, 0)},
        'NCR': {'name': 'Necrotic Core', 'color': (0, 0, 1)}
    }

    with torch.no_grad():
        mask_np = binary_mask.squeeze().cpu().numpy()

        for i, class_key in enumerate(tumor_metrics.keys()):
            pixel_area = np.sum(mask_np[i] > 0.5)
            tumor_metrics[class_key]['pixel_area'] = pixel_area

            if pixel_spacing:
                mm2_per_pixel = pixel_spacing[0] * pixel_spacing[1]
                tumor_metrics[class_key]['physical_area_mm2'] = pixel_area * mm2_per_pixel

    return tumor_metrics

def save_masks(output_dir, image_tensor, probability_masks, binary_masks):
    """Save input image and all masks to directory"""
    os.makedirs(output_dir, exist_ok=True)

    # Save input image
    input_image = image_tensor[0, 0].cpu().numpy()
    plt.imsave(f"{output_dir}/input_image.png", input_image, cmap='gray')

    # Save masks
    classes = ['NCR', 'ED', 'ET']
    for i, class_name in enumerate(classes):
        # Probability mask
        prob_mask = probability_masks[0, i].cpu().numpy()
        plt.imsave(f"{output_dir}/prob_{class_name}.png", prob_mask, cmap='jet', vmin=0, vmax=1)

        # Binary mask
        bin_mask = binary_masks[0, i].cpu().numpy()
        plt.imsave(f"{output_dir}/binary_{class_name}.png", bin_mask, cmap='gray', vmin=0, vmax=1)

def visualize_prediction(image, pred_mask, tumor_sizes=None, pixel_spacing=None):
    """Visualization with size annotations"""
    # ... [Keep the visualization function from previous answer] ...
    
    
def calculate_tumor_size(binary_mask, pixel_spacing=(1.0, 1.0), slice_thickness=1.0):
    """
    Calculate the size of the tumor from a binary segmentation mask

    Args:
        binary_mask: Binary segmentation mask of shape [B, C, H, W] where C is the number of classes
                    (in this case, 3 for NCR, ED, ET)
        pixel_spacing: Tuple of (x, y) pixel spacing in mm
        slice_thickness: Thickness of the slice in mm (for 3D volume calculation)

    Returns:
        Dict containing area (in mm²) and volume (in mm³) for each tumor region and total
    """
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()

    # Get image dimensions
    _, num_classes, height, width = binary_mask.shape

    # Area per pixel in mm²
    area_per_pixel = pixel_spacing[0] * pixel_spacing[1]

    # Volume per voxel in mm³
    volume_per_voxel = pixel_spacing[0] * pixel_spacing[1] * slice_thickness

    # Initialize results dictionary
    results = {
        'regions': [],
        'total_area_mm2': 0,
        'total_volume_mm3': 0,
        'max_diameter_mm': 0
    }

    region_names = ['Necrotic Core (NCR)', 'Edema (ED)', 'Enhancing Tumor (ET)']

    # Calculate metrics for each class
    for c in range(num_classes):
        region_mask = binary_mask[0, c]  # Take first batch

        # Count pixels
        num_pixels = np.sum(region_mask)

        # Calculate area
        area_mm2 = num_pixels * area_per_pixel

        # Calculate volume
        volume_mm3 = num_pixels * volume_per_voxel

        # Calculate maximum diameter
        if num_pixels > 0:
            # Label connected components
            labeled_mask, num_features = ndimage.label(region_mask)

            # Find properties of the largest connected component
            if num_features > 0:
                largest_label = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
                largest_component = (labeled_mask == largest_label)

                # Find the maximum diameter using the distance transform
                distance_map = ndimage.distance_transform_edt(largest_component, sampling=pixel_spacing)
                max_distance = np.max(distance_map)
                max_diameter_mm = 2 * max_distance  # Diameter = 2 * radius
            else:
                max_diameter_mm = 0
        else:
            max_diameter_mm = 0

        # Store region results
        region_result = {
            'name': region_names[c],
            'area_mm2': float(area_mm2),
            'volume_mm3': float(volume_mm3),
            'max_diameter_mm': float(max_diameter_mm)
        }
        results['regions'].append(region_result)

        # Add to total
        results['total_area_mm2'] += area_mm2
        results['total_volume_mm3'] += volume_mm3
        results['max_diameter_mm'] = max(results['max_diameter_mm'], max_diameter_mm)

    return results

def visualize_tumor_sizes(image, binary_mask, size_results, save_path=None, return_image=True):
    """
    Visualize tumor segmentation with size information and return as PIL image.

    Args:
        image: Original image as numpy array or tensor.
        binary_mask: Binary segmentation mask (tensor or numpy).
        size_results: Dictionary with size/volume information.
        save_path: Optional path to save figure.
        return_image: If True, return PIL image for base64 or web use.

    Returns:
        PIL.Image if return_image is True; else None.
    """
    if isinstance(image, torch.Tensor):
        image = image[0, 0].cpu().numpy()

    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original MRI", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image, cmap='gray')
    colors = [(0, 0, 1), (0, 1, 0), (1, 0.6, 0)]  # Blue, Green, Orange
    overlay = np.zeros((*image.shape, 3), dtype=np.float32)

    for c, color in enumerate(colors):
        mask = binary_mask[0, c]
        overlay[mask > 0] = color

    axes[0, 1].imshow(overlay, alpha=0.6)
    axes[0, 1].set_title("Tumor Segmentation", fontsize=12)
    axes[0, 1].axis('off')

    # Prepare region info text
    region_lines = []
    for region in size_results['regions']:
        region_lines.append(
            f"{region['name']}:\n"
            f"  Area: {region['area_mm2']:.2f} mm²\n"
            f"  Volume: {region['volume_mm3']:.2f} mm³\n"
            f"  Max Diameter: {region['max_diameter_mm']:.2f} mm"
        )
    region_info = "\n\n".join(region_lines)

    total_info = (
        f"Total Tumor:\n"
        f"  Total Area: {size_results['total_area_mm2']:.2f} mm²\n"
        f"  Total Volume: {size_results['total_volume_mm3']:.2f} mm³\n"
        f"  Maximum Diameter: {size_results['max_diameter_mm']:.2f} mm"
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1, 0].axis('off')
    axes[1, 0].text(0.05, 0.95, region_info + "\n\n" + total_info,
                   transform=axes[1, 0].transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

    axes[1, 1].axis('off')
    rano_info = assess_rano_criteria(size_results)
    axes[1, 1].text(0.05, 0.95, rano_info, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    
    # plt.show()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)

    if return_image:
        # Convert figure to PIL image
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        plt.close(fig)
        return pil_img
    

def assess_rano_criteria(size_results):
    """
    Assess tumor based on simplified RANO criteria
    (Response Assessment in Neuro-Oncology)

    Note: This is a simplified interpretation and should not be used for clinical decisions.
    A proper RANO assessment requires multiple time points and clinical context.
    """
    max_diameter = size_results['max_diameter_mm']

    # Simplified RANO-inspired assessment (for demonstration purposes)
    rano_text = "RANO-Inspired Assessment:\n"

    # This is just for demonstration - actual RANO criteria are more complex
    # and require comparison between timepoints
    if max_diameter < 10:
        rano_text += "  Small tumor burden (<10mm)\n"
    elif max_diameter < 20:
        rano_text += "  Moderate tumor burden (10-20mm)\n"
    else:
        rano_text += "  Significant tumor burden (>20mm)\n"

    # Add enhancing vs non-enhancing assessment
    et_region = [r for r in size_results['regions'] if "Enhancing" in r['name']][0]
    ncr_ed_area = sum(r['area_mm2'] for r in size_results['regions'] if "Enhancing" not in r['name'])

    enhancement_ratio = et_region['area_mm2'] / (ncr_ed_area + 1e-6)  # Avoid division by zero

    if enhancement_ratio > 0.5:
        rano_text += "  High enhancing component\n"
    else:
        rano_text += "  Predominantly non-enhancing\n"

    rano_text += "\nDisclaimer: This is a simplified assessment for demonstration purposes only. "
    rano_text += "Actual clinical decisions require expert interpretation and follow-up scans."

    return rano_text

def estimate_pixel_spacing(image_path):
    """
    Estimate pixel spacing from image metadata if available,
    otherwise use default values.

    In real clinical scenarios, this would come from DICOM metadata.
    """
    # Try to get pixel spacing from image metadata
    # For demonstration, we'll use default values
    # In real scenarios, this should come from DICOM headers

    # Default values (1mm x 1mm)
    pixel_spacing = (1.0, 1.0)
    slice_thickness = 1.0

    # Here you would add code to extract actual values from DICOM metadata

    return pixel_spacing, slice_thickness

def run_tumor_size_prediction(model, image_path, save_results=True, output_dir=None):
    """
    Main function to run tumor segmentation and size prediction
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Estimate pixel spacing (in a real scenario, this would come from DICOM metadata)
    pixel_spacing, slice_thickness = estimate_pixel_spacing(image_path)
    print(f"Using pixel spacing: {pixel_spacing} mm, slice thickness: {slice_thickness} mm")

    # Preprocess image
    try:
        print(f"Processing image: {image_path}")
        image_tensor = preprocess_image(image_path)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None

    # Run inference
    try:
        print("Running segmentation inference...")
        # Run with test-time augmentation
        prediction = run_inference(model, image_tensor, device, test_time_augmentation=True)

        # Process prediction
        processed_pred = postprocess_prediction(prediction)
        print("Segmentation completed")
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None, None

    # Calculate tumor size
    try:
        print("Calculating tumor size metrics...")
        size_results = calculate_tumor_size(processed_pred, pixel_spacing, slice_thickness)

        # Print results
        print("\nTumor Size Results:")
        for region in size_results['regions']:
            print(f"{region['name']}:")
            print(f"  Area: {region['area_mm2']:.2f} mm²")
            print(f"  Volume: {region['volume_mm3']:.2f} mm³")
            print(f"  Max Diameter: {region['max_diameter_mm']:.2f} mm")

        print(f"\nTotal Tumor:")
        print(f"  Total Area: {size_results['total_area_mm2']:.2f} mm²")
        print(f"  Total Volume: {size_results['total_volume_mm3']:.2f} mm³")
        print(f"  Maximum Diameter: {size_results['max_diameter_mm']:.2f} mm")
    except Exception as e:
        print(f"Error calculating tumor size: {e}")
        return processed_pred, None, None

    # Visualize results
    try:
        print("Visualizing results with size information...")
        save_path = None
        if save_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(output_dir, f"{base_name}_tumor_size.png")

        fig = visualize_tumor_sizes(image_tensor, processed_pred, size_results, save_path, return_image=True)
    except Exception as e:
        print(f"Error visualizing results: {e}")

    return processed_pred, size_results, fig
        
        

def predict_image(img_path):
    """
    Step 1: Check if it's a brain MRI.
    Step 2: If yes, classify tumor type.
    """

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict brain vs non-brain
    binary_result = binary_model.predict(img_array)



    if binary_result[0][0] > 0.5:
      return {
            "classification": "non_brain",
            "message": "The uploaded image is not a brain MRI. Tumor classification skipped."
        }
    else:
        img = cv2.imread(img_path)
        img_s = cv2.resize(img.copy(), (150, 150))
        img_array = np.array(img_s).reshape(1, 150, 150, 3)
        tumor_result = tumor_model.predict(img_array)
        tumor_index = tumor_result.argmax()

        tumor_labels = {
            0: "Glioma",
            1: "Melignoma",
            2: "No Tumor",
            3: "Pituitary"
        }
        
        # segmentation 
        
        
        output = {
            "classification": "brain_mri",
            "tumor_prediction": tumor_labels[tumor_index]
        }
        
        
        if tumor_index in [0,1,2,3]:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = load_pretrained_model('best_brain_segmentation_model.pth', device)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                return

            # Run tumor size prediction
            segmentation, size_results, fig = run_tumor_size_prediction(
                model, img_path, save_results=True, output_dir='output'
            )
            
            if size_results:
                print("Tumor size prediction completed successfully")
                output['sizes'] = size_results
            else:
                print("Failed to complete tumor size prediction")
                
            if fig and isinstance(fig, Image.Image):
                buf = io.BytesIO()
                
                # Save the PIL Image to the buffer in PNG format
                fig.save(buf, format='PNG')
                buf.seek(0)

                # Encode as base64
                image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                output['segmentation_image'] = image_base64
            else:
                print('no image segmentation output retuned from the visualizer')

        
        return output