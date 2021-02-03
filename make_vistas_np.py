import json
import os
import shutil
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as tf
from tqdm import tqdm

from cityscapes_labels import create_name_to_id, create_id_to_name

img_transformation = tf.Compose([tf.ToTensor()])
label_transformation = tf.Compose([tf.ToTensor()])


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]


class MapillaryVistasDataset(data.Dataset):
    n_classes = 66
    vistas_to_cityscapes = {
        'construction--barrier--curb': 'sidewalk',
        'construction--barrier--fence': 'fence',
        'construction--barrier--guard-rail': 'fence',
        'construction--barrier--wall': 'wall',
        'construction--flat--bike-lane': 'road',
        'construction--flat--crosswalk-plain': 'road',
        'construction--flat--curb-cut': 'sidewalk',
        'construction--flat--parking': 'road',
        'construction--flat--pedestrian-area': 'sidewalk',
        'construction--flat--rail-track': 'road',
        'construction--flat--road': 'road',
        'construction--flat--service-lane': 'road',
        'construction--flat--sidewalk': 'sidewalk',
        'construction--structure--bridge': 'building',
        'construction--structure--building': 'building',
        'construction--structure--tunnel': 'building',
        'human--person': 'person',
        'human--rider--bicyclist': 'rider',
        'human--rider--motorcyclist': 'rider',
        'human--rider--other-rider': 'rider',
        'marking--crosswalk-zebra': 'road',
        'marking--general': 'road',
        'nature--sand': 'terrain',
        'nature--sky': 'sky',
        'nature--snow': 'terrain',
        'nature--terrain': 'terrain',
        'nature--vegetation': 'vegetation',
        'object--support--pole': 'pole',
        'object--support--traffic-sign-frame': 'traffic sign',
        'object--support--utility-pole': 'pole',
        'object--traffic-light': 'traffic light',
        'object--traffic-sign--front': 'traffic sign',
        'object--vehicle--bicycle': 'bicycle',
        'object--vehicle--bus': 'bus',
        'object--vehicle--car': 'car',
        'object--vehicle--motorcycle': 'motorcycle',
        'object--vehicle--on-rails': 'train',
        'object--vehicle--truck': 'truck',
    }

    def __init__(self, root, split='training'):
        """
        :param root:         path to the datasets root
        :param split:        dataset split -- 'training', 'validation' or 'testing'
        :param transforms:   torchvision transformations
        """
        self.root = root
        self.split = split

        self.images_base = os.path.join(self.root, self.split, 'images')
        self.annotations_base = os.path.join(self.root, self.split, 'v1.2/labels')

        self.files = {}
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.jpg')

        self.class_names, self.class_ids, self.class_colors, self.vistas_id_to_name, self.labels = self._parse_config(os.path.join(self.root, 'config.json'))
        self.cityscapes_name_to_id = create_name_to_id()
        self.cityscapes_id_to_name = create_id_to_name()

        self.ignore_class = 19
        self.cityscapes_classes_mapper = torch.zeros(self.n_classes).long().fill_(self.ignore_class)
        for i in range(len(self.labels)):
            self.cityscapes_classes_mapper[i] = self.to_cityscapes_class(i)

        if not self.files[split]:
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files[split]), split))

    @staticmethod
    def _parse_config(config_path):
        # read in config file
        with open(config_path) as config_file:
            config = json.load(config_file)

        labels = config['labels']

        class_names = []
        class_ids = []
        class_colors = []
        id_to_name = {}
        print("> There are {} labels in the config file".format(len(labels)))
        for label_id, label in enumerate(labels):
            class_names.append(label["readable"])
            class_ids.append(label_id)
            class_colors.append(label["color"])
            id_to_name[label_id] = label["name"]
        return class_names, class_ids, class_colors, id_to_name, labels

    def __len__(self):
        return len(self.files[self.split])

    def to_cityscapes_class(self, id):
        vistas_name = self.vistas_id_to_name[id]
        cityscapes_name = self.vistas_to_cityscapes.get(vistas_name)
        if cityscapes_name == None:
            return self.cityscapes_name_to_id['ignore']
        else:
            return self.cityscapes_name_to_id[cityscapes_name]

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png"))

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        # img = Image.open(img_path)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        lbl = Image.open(lbl_path)

        label = label_transformation(lbl)
        label = (label * 255).long()
        label = self.cityscapes_classes_mapper[label]
        return img_path, lbl_path, label


def transform_dataset_split(dataset, split):
    cnt_in = 0
    allclasses = set()
    with tqdm(total=len(dataset)) as progress_bar:
        for i in range(len(dataset)):
            img_path, lbl_path, lbl = dataset[i]
            classes = torch.unique(lbl.view(-1)).numpy().tolist()
            if 11 in classes or 12 in classes:
                shutil.copy(img_path, './vistas-np/test/images/')
                shutil.copy(lbl_path, './vistas-np/test/labels/')
            else:
                shutil.copy(img_path, './vistas-np/' + split + '/images/')
                shutil.copy(lbl_path, './vistas-np/' + split + '/labels/')
                cnt_in += 1
                allclasses.update(classes)
            progress_bar.update(1)

    print(f"Images left in {split} split: {cnt_in}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Vistas-NP maker')
    parser.add_argument('dataroot', help='vistas dataset root', type=str, default='.')
    args = parser.parse_args()

    print('Loading Vistas dataset..')
    train_dataset = MapillaryVistasDataset(args.dataroot, split='training')
    val_dataset = MapillaryVistasDataset(args.dataroot, split='validation')

    print('Creating Vistas-NP. This could take a while..')
    shutil.copy(f"{args.dataroot}/config.json", './vistas-np/')
    transform_dataset_split(val_dataset, 'validation')
    transform_dataset_split(train_dataset, 'training')
