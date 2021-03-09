from PIL import Image
from torch.utils.data import Dataset


class MapillarySegmentation(Dataset):
    def __init__(self, root_dir='', image_set='train', transforms=None):

        self.root_dir = root_dir
        self.image_set = image_set
        self.transforms = transforms

        self.images = []
        self.targets = []

        with open(f'list/{image_set}_mapillary_1920_1080.txt', 'r') as f:
            for line in f:
                img, trg = line.split(';')
                self.images.append(img)
                self.targets.append(trg[:-1])

        self.images = list(sorted(self.images, key=lambda x: x.split('/')[-1]))
        self.targets = list(sorted(self.targets, key=lambda x: x.split('/')[-1]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
