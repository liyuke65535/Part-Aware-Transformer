from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob

from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class RandPerson(object):

    def __init__(self, root, combineall=True):

        self.images_dir = osp.join(root)
        self.img_path = 'your_path/randperson_subset/randperson_subset'
        self.train_path = self.img_path
        self.gallery_path = ''
        self.query_path = ''
        self.train = []
        self.gallery = []
        self.query = []
        self.num_train_ids = 0
        self.has_time_info = True
        # self.show_train()

    def preprocess(self):
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path, '*g')))

        data = []
        all_pids = {}
        camera_offset = [0, 2, 4, 6, 8, 9, 10, 12, 13, 14, 15]
        frame_offset = [0, 160000, 340000,490000, 640000, 1070000, 1330000, 1590000, 1890000, 3190000, 3490000]
        fps = 24

        for fpath in fpaths:
            fname = osp.basename(fpath)  # filename: id6_s2_c2_f6.jpg
            fields = fname.split('_')
            pid = int(fields[0])
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]  # relabel
            camid = camera_offset[int(fields[1][1:])] + int(fields[2][1:])  # make it starting from 0
            time = (frame_offset[int(fields[1][1:])] + int(fields[3][1:7])) / fps
            data.append((fpath, pid, camid, time))
            # print(fname, pid, camid, time)
        return data, int(len(all_pids))

    def show_train(self):
        self.train, self.num_train_ids = self.preprocess()

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  all      | {:5d} | {:8d}\n"
              .format(self.num_train_ids, len(self.train)))
