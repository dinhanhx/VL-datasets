import csv
import logging

from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from tqdm import tqdm

logging.basicConfig(filename='ViVQA_sanity_check.log',
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@dataclass
class Meta:
    img_root_dir = Path('/mnt/disks/nlpvnhub/dinhanhx/')
    # where COCO train2017 and val2017 dirs are located

    csv_dir = Path('/mnt/disks/nlpvnhub/dinhanhx/ViVQA-main')
    # where github.com/kh4nh12/ViVQA is cloned

    img_train_dir = img_root_dir.joinpath('train2017')
    img_val_dir = img_root_dir.joinpath('val2017')

    train_file = csv_dir.joinpath('train.csv')
    test_file = csv_dir.joinpath('test.csv')


class DataUnpacker:
    def __init__(self, meta: Meta, sanity_check=False) -> None:
        self.meta = meta
        if sanity_check:
            self.run_sanity_check()

    def get_item(self, index: int = 0, target='train_file'):
        """Get data by index coresponding in csv files

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
        meta_dir_list = [self.meta.img_train_dir, self.meta.img_val_dir]

        with open(target) as target_file:
            reader = csv.DictReader(target_file)
            for i, line in enumerate(reader):
                if i == index:
                    for img_dir in meta_dir_list:
                        img_file = img_dir.joinpath(str(line['img_id']).zfill(12)+'.jpg')
                        if img_file.is_file():
                            return line, img_file

        return (None, None)

    def run_sanity_check(self):
        """Check files, directories, and images path exist or not
        """
        meta_file_list = [self.meta.train_file, self.meta.test_file]
        meta_dir_list = [self.meta.img_train_dir, self.meta.img_val_dir]

        for p in meta_dir_list+meta_dir_list:
            if not p.exists():
                logger.warn(f'{p} does not exist')

        for target in meta_file_list:
            with open(target) as target_file:
                reader = csv.DictReader(target_file)
                for line in tqdm(reader):
                    file_found = False
                    for img_dir in meta_dir_list:
                        img_file = img_dir.joinpath(str(line['img_id']).zfill(12)+'.jpg')
                        if img_file.is_file():
                            file_found = True
                            if img_dir != self.meta.img_train_dir:
                                logger.info(f'{line} @ {target} has {img_file}')

                    if not file_found:
                        logger.warn(f'{line} @ {target} has no image')

    def get_image_list(self):
        meta_file_list = [self.meta.test_file]
        meta_dir_list = [self.meta.img_train_dir, self.meta.img_val_dir]

        image_list = set()
        for target in meta_file_list:
            with open(target) as target_file:
                reader = csv.DictReader(target_file)
                for line in tqdm(reader):
                    for img_dir in meta_dir_list:
                        img_file = img_dir.joinpath(str(line['img_id']).zfill(12)+'.jpg')
                        if img_file.is_file():
                            image_list.add(img_file)

        return list(image_list)


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
