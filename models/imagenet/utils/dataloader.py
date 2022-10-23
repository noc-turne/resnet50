import io
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from petrel_client.client import Client
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .dataset import McDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CephDataset(Dataset):
    r"""
    Dataset using ceph to read data.

    Arguments
        * image_dir (string): Root directory of the Dataset.
        * meta_file (string): The meta file of the Dataset. Each line has a image path
          and a label. Eg: ``nm091234/image_56.jpg 18``
        * transform (callable, optional): A function/transform that takes in an PIL image
          and returns a transformed image.
    """
    def __init__(self, image_dir, meta_file, transform=False):
        self.image_dir = image_dir
        self.transform = transform

        self.client = Client()
        meta_file = self.client.Get(meta_file, update_cache=True)
        self.meta_list = bytes.decode(meta_file).split('\n')
        if self.meta_list[-1] == '':
            self.meta_list.pop()
        self.num = len(self.meta_list)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        filename = self.image_dir + self.meta_list[index].split()[0]
        cls = int(self.meta_list[index].split()[1])

        img = Image.open(io.BytesIO(self.client.Get(filename, update_cache=True)))
        img = img.convert('RGB')

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls


def build_augmentation(cfg):
    compose_list = []
    if cfg.random_resize_crop:
        compose_list.append(transforms.RandomResizedCrop(cfg.random_resize_crop))
    if cfg.resize:
        compose_list.append(transforms.Resize(cfg.resize))
    if cfg.random_crop:
        compose_list.append(transforms.RandomCrop(cfg.random_crop))
    if cfg.center_crop:
        compose_list.append(transforms.CenterCrop(cfg.center_crop))

    if cfg.mirror:
        compose_list.append(transforms.RandomHorizontalFlip())
    if cfg.colorjitter:
        compose_list.append(transforms.ColorJitter(*cfg.colorjitter))

    compose_list.append(transforms.ToTensor())

    data_normalize = transforms.Normalize(mean=cfg.get('mean', [0.485, 0.456, 0.406]),
                                          std=cfg.get('std', [0.229, 0.224, 0.225]))
    compose_list.append(data_normalize)

    return transforms.Compose(compose_list)


def build_dataloader(cfg, world_size, data_reader):
    train_aug = build_augmentation(cfg.train)
    test_aug = build_augmentation(cfg.test)

    if data_reader == 'MemcachedReader':
        train_dataset = McDataset(cfg.train.image_dir, cfg.train.meta_file, train_aug)
    elif data_reader == 'CephReader':
        ceph_image_dir = 's3://parrots_model_data/imagenet/images/train/'
        ceph_meta_file = 's3://parrots_model_data/imagenet/images/meta/train.txt'
        train_dataset = CephDataset(ceph_image_dir, ceph_meta_file, train_aug)
    elif data_reader == 'DirectReader':
        raise NotImplementedError
    else:
        assert data_reader, "data reader should be provided."

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)

    if data_reader == 'MemcachedReader':
        test_dataset = McDataset(cfg.test.image_dir, cfg.test.meta_file, test_aug)
    elif data_reader == 'CephReader':
        ceph_image_dir = 's3://parrots_model_data/imagenet/images/val/'
        ceph_meta_file = 's3://parrots_model_data/imagenet/images/meta/val.txt'
        test_dataset = CephDataset(ceph_image_dir, ceph_meta_file, test_aug)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=(test_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=test_sampler, drop_last=False)
    return train_loader, train_sampler, test_loader, test_sampler
