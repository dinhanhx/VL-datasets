import json
import logging

from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from tqdm import tqdm

logging.basicConfig(filename='uitviic_sanity_check.log',
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@dataclass
class Meta:
    img_dir = Path('/mnt/disks/nlpvnhub/dinhanhx/train2017')
    # where all images of COCO train2017 are located

    json_dir = Path('/mnt/disks/nlpvnhub/dinhanhx/UIT-ViIC')
    # where all json files that are downloaded from
    # https://sites.google.com/uit.edu.vn/uit-nlp/datasets-projects?authuser=0#h.p_Uj6Wqs5dCpc4

    train_file = json_dir.joinpath('uitviic_captions_train2017.json')
    val_file = json_dir.joinpath('uitviic_captions_val2017.json')
    test_file = json_dir.joinpath('uitviic_captions_test2017.json')


class DataUnpacker:
    def __init__(self, meta: Meta, sanity_check=False) -> None:
        self.meta = meta
        if sanity_check:
            self.run_sanity_check()

    def get_item(self, index: int = 0, target='train_file'):
        """Get data by index coresponding in json files

        Parameters
        ----------
        index : int, optional
            index of data in json file, by default 0
        target : str, optional
            'train_file', 'test_file', by default 'train_file'

        Returns
        -------
        A tuple of Dict containing data, and Path to image file
        """
        target = self.meta.__getattribute__(target)

        with open(target) as target_file:
            dataset = json.load(target_file)
            data = dataset['annotations'][index]
            img_file = self.meta.img_dir.joinpath(str(data['image_id']).zfill(12)+'.jpg')
            return data, img_file

    def run_sanity_check(self):
        """Check files, directories, and images path exist or not
        """
        meta_file_list = [self.meta.train_file, self.meta.val_file, self.meta.test_file]
        for target in meta_file_list:
            with open(target) as target_file:
                dataset = json.load(target_file)
                for d in tqdm(dataset['annotations']):
                    img_file = self.meta.img_dir.joinpath(str(d['image_id']).zfill(12)+'.jpg')
                    if not img_file.is_file():
                        logger.warn(f'{d} @ {target} has no image')

    def get_image_list(self):
        meta_file_list = [self.meta.train_file, self.meta.val_file, self.meta.test_file]
        image_set = set()
        for target in meta_file_list:
            with open(target) as target_file:
                dataset = json.load(target_file)
                for d in tqdm(dataset['annotations']):
                    image_set.add(self.meta.img_dir.joinpath(str(d['image_id']).zfill(12)+'.jpg'))

        return list(image_set)


if '__main__' == __name__:
    meta = Meta()
    data_unpacker = DataUnpacker(meta)
    image_set = data_unpacker.get_image_list()
    l = len(image_set)
    h, w = 0, 0
    min_h, min_w = Image.open(image_set[0]).height, Image.open(image_set[0]).width
    max_h, max_w = 0, 0
    for img_path in tqdm(image_set):
        img = Image.open(img_path)
        h += img.height
        w += img.width
        if img.height * img.width <= min_h * min_w:
            min_h, min_w = img.height, img.width

        if img.height * img.width >= max_h * max_w:
            max_h, max_w = img.height, img.width

    print(f'Number of image-text pairs: {l}')
    print(f'Average H W: {h/l} {w/l}')
    print(f'Min H W: {min_h} {min_w}')
    print(f'Max H W: {max_h} {max_w}')
