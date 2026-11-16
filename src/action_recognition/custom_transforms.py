import random
import torchvision.transforms.functional as F
import torch
import torchvision
import numbers
import torch.nn.functional as tF
import numpy as np
import cv2


class CenterCropResizeWithPadding(torch.nn.Module):
    def __init__(self, size, crop_size):
        """
        Args:
            size (int): Resize shorter side to this dimension.
            crop_size (int): Final crop size around the bounding box or center.
            perc_bbox_within_crop (float): Minimum percentage of the bbox area that must be within the crop.
        """
        super().__init__()
        self.size = size  # Short side target size for resizing
        self.crop_size = crop_size  # Target size after center crop

    def resize_to_short_side(self, clip, h, w):
        """Resize the clip to the specified short side size, maintaining aspect ratio."""
        scale = self.size / min(h, w)
        new_h, new_w = int(scale * h), int(scale * w)
        return (
            tF.interpolate(
                clip, size=(new_h, new_w), mode="bilinear", align_corners=False
            ),
            new_h,
            new_w,
        )

    def center_crop(self, clip, h, w, crop_x_center, crop_y_center):
        """Center crop around the specified point (usually the bbox center or frame center)."""
        crop_w, crop_h = self.crop_size, self.crop_size
        x1 = max(0, min(w - crop_w, crop_x_center - crop_w // 2))
        y1 = max(0, min(h - crop_h, crop_y_center - crop_h // 2))
        x2, y2 = x1 + crop_w, y1 + crop_h

        return clip[:, :, y1:y2, x1:x2]

    def forward(self, clip, bbox=None):
        """
        Resize, then center crop around the bounding box if provided, and resize to the specified crop size.

        Args:
            clip (Tensor): Video tensor of shape (C, T, H, W).
            bbox (list): Bounding box in YOLO format [class_id, x_center, y_center, width, height].

        Returns:
            Tensor: Resized and center-cropped video clip of shape (C, T, crop_size, crop_size).
        """
        c, t, h, w = clip.shape

        # Resize the clip based on the shorter side
        resized_clip, new_h, new_w = self.resize_to_short_side(clip, h, w)

        if bbox is None:
            # Center crop around the middle of the frame if no bbox
            crop_x_center, crop_y_center = new_w // 2, new_h // 2
        else:
            # Convert bbox from normalized coordinates to pixel coordinates on the resized frame
            _, x_center, y_center, bbox_width, bbox_height = bbox
            bbox_x_center = int(x_center * new_w)
            bbox_y_center = int(y_center * new_h)

            # Center the crop around the bounding box center
            crop_x_center, crop_y_center = bbox_x_center, bbox_y_center

        # Apply center crop around the determined center
        cropped_clip = self.center_crop(
            resized_clip, new_h, new_w, crop_x_center, crop_y_center
        )

        return cropped_clip


class RandomCropResize(torch.nn.Module):
    def __init__(self, size, perc_bbox_within_crop=0.8):
        """
        Args:
            size (int): Final output size for resizing (size x size).
            perc_bbox_within_crop (float): Minimum percentage of the bbox area that must be within the crop.
        """
        super().__init__()
        self.size = size  # Final size to resize the crop to
        self.perc_bbox_within_crop = perc_bbox_within_crop

    def resize_with_padding(self, clip, crop_h, crop_w):
        """Resize clip with padding to maintain aspect ratio."""
        scale = min(self.size / crop_h, self.size / crop_w)
        new_h, new_w = int(scale * crop_h), int(scale * crop_w)
        resized_clip = tF.interpolate(
            clip, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        # Calculate padding to make the output size (self.size, self.size)
        pad_h = self.size - new_h
        pad_w = self.size - new_w
        padding_top = pad_h // 2
        padding_bottom = pad_h - padding_top
        padding_left = pad_w // 2
        padding_right = pad_w - padding_left

        return tF.pad(
            resized_clip,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )

    def forward(self, clip, bbox):
        """
        Perform random crop around the bounding box and resize to the specified output size.

        Args:
            clip (Tensor): Video tensor of shape (C, T, H, W).
            bbox (list): Bounding box in YOLO format [class_id, x_center, y_center, width, height].

        Returns:
            Tensor: Resized video clip of shape (C, T, self.size, self.size).
        """
        c, t, h, w = clip.shape

        # If no bbox is provided, draw a random crop with min size as 'size' and max size as full image size
        if bbox is None:
            crop_w = (
                np.random.randint(self.size, w + 1) if w + 1 > self.size else self.size
            )
            crop_h = (
                np.random.randint(self.size, h + 1) if h + 1 > self.size else self.size
            )
            x1 = np.random.randint(0, w - crop_w + 1) if w - crop_w + 1 > 0 else 0
            y1 = np.random.randint(0, h - crop_h + 1) if h - crop_h + 1 > 0 else 0
            cropped_clip = clip[:, :, y1 : y1 + crop_h, x1 : x1 + crop_w]
            return self.resize_with_padding(cropped_clip, crop_h, crop_w)

        # Convert bbox from normalized coordinates to pixel coordinates
        _, x_center, y_center, bbox_width, bbox_height = bbox
        bbox_x_center = int(x_center * w)
        bbox_y_center = int(y_center * h)
        if bbox_width > 1:
            raise ValueError(f"bbox_width is greater than 1: {bbox_width}")
        if bbox_height > 1:
            raise ValueError(f"bbox_height is greater than 1: {bbox_height}")
        bbox_w = int(bbox_width * w)
        bbox_h = int(bbox_height * h)
        bbox_area = bbox_w * bbox_h

        # Attempt to find a valid crop around the bbox with the specified overlap condition
        for _ in range(50):
            # Draw random crop dimensions between bbox size and full image size
            crop_w = w if bbox_w == w else np.random.randint(bbox_w, w + 1)
            crop_h = h if bbox_h == h else np.random.randint(bbox_h, h + 1)

            # Calculate random top-left corner within bounds around the bbox center
            x1_low = max(0, min(w - crop_w, bbox_x_center - crop_w // 2))
            x1_high = min(w - crop_w, bbox_x_center + crop_w // 2) + 1
            y1_low = max(0, min(h - crop_h, bbox_y_center - crop_h // 2))
            y1_high = min(h - crop_h, bbox_y_center + crop_h // 2) + 1

            if x1_low > x1_high or y1_low > y1_high:
                raise ValueError("Min pos is larger than high")
            x1 = np.random.randint(x1_low, x1_high) if x1_low < x1_high else x1_low
            y1 = np.random.randint(y1_low, y1_high) if y1_low < y1_high else y1_low
            x2, y2 = x1 + crop_w, y1 + crop_h

            # Calculate intersection area between bbox and crop
            inter_x1 = max(x1, bbox_x_center - bbox_w // 2)
            inter_y1 = max(y1, bbox_y_center - bbox_h // 2)
            inter_x2 = min(x2, bbox_x_center + bbox_w // 2)
            inter_y2 = min(y2, bbox_y_center + bbox_h // 2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            # Check if the crop contains the required percentage of the bbox area
            if inter_area >= self.perc_bbox_within_crop * bbox_area:
                # Crop and resize to the target size
                cropped_clip = clip[:, :, y1:y2, x1:x2]
                return self.resize_with_padding(cropped_clip, crop_h, crop_w)

        # If no valid crop is found, resize the entire clip as a fallback
        return self.resize_with_padding(clip, h, w)


class ResizeAndPadVideo:
    def __init__(self, size):
        self.size = size  # Desired output size (size x size)

    def __call__(self, clip):
        try:
            # clip is a Tensor of shape [C, T, H, W]
            # Rearrange to [T, C, H, W] for processing
            clip = clip.permute(1, 0, 2, 3)  # Now shape is [T, C, H, W]
            t, c, h, w = clip.shape

            # Calculate scale factor to resize the longer side to self.size
            scale = self.size / max(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            # Resize the clip
            clip = tF.interpolate(
                clip, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

            # Calculate padding to make the clip square
            pad_h = self.size - new_h  # pad_h >= 0
            pad_w = self.size - new_w  # pad_w >= 0

            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            # Pad the spatial dimensions (H, W)
            clip = tF.pad(
                clip,
                pad=(pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )

            # Rearrange back to [C, T, H, W]
            clip = clip.permute(1, 0, 2, 3)
            return clip
        except Exception as e:
            print(f"Error in ResizeAndPadVideo: {e}")
            return clip


class VideoColorJitter(torch.nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            value = [max(bound[0], center - value), min(bound[1], center + value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
            if value[0] == value[1]:
                value = None
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [max(bound[0], value[0]), min(bound[1], value[1])]
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with length 2."
            )
        return value

    def get_params(self):
        transforms = []
        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(lambda img: F.adjust_brightness(img, brightness_factor))

        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(lambda img: F.adjust_contrast(img, contrast_factor))

        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(lambda img: F.adjust_saturation(img, saturation_factor))

        if self.hue is not None:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(lambda img: F.adjust_hue(img, hue_factor))

        random.shuffle(transforms)
        return transforms

    def forward(self, video):
        if random.random() < self.p:
            transforms = self.get_params()
            # video shape: (C, T, H, W)
            c, t, h, w = video.shape
            video = video.permute(1, 0, 2, 3)  # (T, C, H, W)
            # Apply the same transforms to each frame
            for i in range(len(video)):
                frame = video[i]
                for transform in transforms:
                    frame = transform(frame)
                video[i] = frame
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        return video


class VideoRandomGrayscale(torch.nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, video):
        if random.random() < self.p:
            # Apply grayscale conversion to all frames
            c, t, h, w = video.shape
            video = video.permute(1, 0, 2, 3)  # (T, C, H, W)
            for i in range(len(video)):
                frame = video[i]
                frame = F.rgb_to_grayscale(frame, num_output_channels=3)
                video[i] = frame
            video = video.permute(1, 0, 2, 3)
        return video


class VideoGaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def forward(self, video):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            gaussian_blur = torchvision.transforms.GaussianBlur(self.kernel_size, sigma)
            c, t, h, w = video.shape
            video = video.permute(1, 0, 2, 3)
            for i in range(len(video)):
                frame = video[i]
                frame = gaussian_blur(frame)
                video[i] = frame
            video = video.permute(1, 0, 2, 3)
        return video


class VideoRandomAffine(torch.nn.Module):
    def __init__(self, degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p

    def forward(self, video):
        if random.random() < self.p:
            c, t, h, w = video.shape
            angle = random.uniform(-self.degrees, self.degrees)
            max_dx = self.translate[0] * w
            max_dy = self.translate[1] * h
            tx = random.uniform(-max_dx, max_dx)
            ty = random.uniform(-max_dy, max_dy)
            scale = random.uniform(self.scale[0], self.scale[1])
            video = video.permute(1, 0, 2, 3)
            for i in range(len(video)):
                video[i] = F.affine(
                    video[i],
                    angle=angle,
                    translate=[int(tx), int(ty)],
                    scale=scale,
                    shear=0,
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                )
            video = video.permute(1, 0, 2, 3)
        return video


class VideoRandomErasing(torch.nn.Module):
    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def forward(self, video):
        if random.random() < self.p:
            c, t, h, w = video.shape
            area = h * w
            for _ in range(10):
                erase_area = random.uniform(self.scale[0], self.scale[1]) * area
                aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
                erase_h = int(round((erase_area * aspect_ratio) ** 0.5))
                erase_w = int(round((erase_area / aspect_ratio) ** 0.5))
                if erase_h >= h or erase_w >= w:
                    continue
                x1 = random.randint(0, w - erase_w)
                y1 = random.randint(0, h - erase_h)
                video[:, :, y1:y1 + erase_h, x1:x1 + erase_w] = self.value
                break
        return video



class VideoTemporalReverse(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, video):
        if random.random() < self.p:
            # video shape: (C, T, H, W) — reverse along time axis
            video = video.flip(1)
        return video


class VideoCompressionArtifacts(torch.nn.Module):
    def __init__(self, quality_range=(10, 50), p=0.5):
        super().__init__()
        self.quality_range = quality_range
        self.p = p

    def forward(self, video):
        if random.random() < self.p:
            quality = random.randint(self.quality_range[0], self.quality_range[1])
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            c, t, h, w = video.shape
            video = video.permute(1, 0, 2, 3)
            for i in range(len(video)):
                frame_np = (video[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                _, encoded = cv2.imencode(".jpg", frame_bgr, encode_param)
                decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
                video[i] = torch.from_numpy(decoded_rgb).permute(2, 0, 1).float() / 255.0
            video = video.permute(1, 0, 2, 3)
        return video
