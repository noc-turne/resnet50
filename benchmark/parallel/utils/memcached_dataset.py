import mc
from torch.utils.data import Dataset
import io
from PIL import Image


class McDataset(Dataset):
    def __init__(self, root_dir, meta_file_or_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if isinstance(meta_file_or_list, str):
            with open(meta_file_or_list) as f:
                meta_list = f.readlines()
        else:
            meta_list = meta_file_or_list

        self.num = len(meta_list)
        self.metas = []
        for line in meta_list:
            path, cls = line.strip().split()
            self.metas.append((path, int(cls)))
        self.initialized = False

    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]

        # memcached
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_buf = mc.ConvertBuffer(value)
        buff = io.BytesIO(value_buf)
        with Image.open(buff) as img:
            img = img.convert('RGB')

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
