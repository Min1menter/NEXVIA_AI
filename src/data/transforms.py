"""Image transforms shared by training AND serving (keeps preprocessing identical)."""
from torchvision import transforms as T

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transforms(size=224, train=True):
    if train:
        # No horizontal flip: laterality (heart on the left) is clinically meaningful.
        return T.Compose([
            T.Grayscale(3),
            T.Resize((size, size)),
            T.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Grayscale(3), T.Resize((size, size)), T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
