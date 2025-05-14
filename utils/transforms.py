from torchvision.transforms import functional as TF
import random
import torchvision.transforms as T

class JointTransform:
    def __init__(self, hflip=True, rotation=True, colour=True):
        self.hflip = hflip
        self.rotation = rotation
        self.colour_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __call__(self, image, mask):
        # Random horizontal flip
        if self.hflip and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random rotation
        if self.rotation:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Random resized crop
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(1.0, 1.0))
        image = TF.resized_crop(image, i, j, h, w, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resized_crop(mask, i, j, h, w, self.size, interpolation=TF.InterpolationMode.NEAREST)

        # Random color jitter
        if self.colour_jitter:
            image = self.colour_jitter(image)

        # Convert to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        mask = TF.pil_to_tensor(mask).squeeze(0).long()

        return image, mask
