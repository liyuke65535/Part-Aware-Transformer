import copy
import logging
import os

class Dataset(object):
    """An abstract class representing a Dataset.
    This is the base class for ``ImageDataset`` and ``VideoDataset``.
    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(self, train, query, gallery, transform=None, mode='train',
                 combineall=False, verbose=True, **kwargs):
        self.train = train
        self.query = query
        self.gallery = gallery
        self.query = [tuple(q_tuple)+({'q_or_g': 'query'},) for q_tuple in self.query]
        self.gallery = [tuple(g_tuple)+({'q_or_g': 'gallery'},) for g_tuple in self.gallery]
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        # if self.train != []:
        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

        # if self.verbose:
        #     self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        if len(data[0]) > 3:
            for _, pid, camid, _ in data:
                pids.add(pid)
                cams.add(camid)
        else:
            for _, pid, camid in data:
                pids.add(pid)
                cams.add(camid)
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        def _combine_data(data):
            for img_path, pid, camid, _ in data:
                if pid in self._junk_pids:
                    continue
                pid = self.dataset_name + "_" + str(pid)
                combined.append((img_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
            num_train_pids, len(self.train), num_train_cams,
            num_query_pids, len(self.query), num_query_cams,
            num_gallery_pids, len(self.gallery), num_gallery_cams
        )

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.
    All other image datasets should subclass it.
    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def show_train(self):
        logger = logging.getLogger('PAT')
        num_train_pids, num_train_cams = self.parse_data(self.train)
        logger.info('=> Loaded {}'.format(self.__class__.__name__))
        logger.info('  ----------------------------------------')
        logger.info('  subset   | # ids | # images | # cameras')
        logger.info('  ----------------------------------------')
        logger.info('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, len(self.train), num_train_cams))
        logger.info('  ----------------------------------------')

    def show_test(self):
        logger = logging.getLogger('PAT')
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)
        logger.info('=> Loaded {}'.format(self.__class__.__name__))
        logger.info('  ----------------------------------------')
        logger.info('  subset   | # ids | # images | # cameras')
        logger.info('  ----------------------------------------')
        logger.info('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, len(self.query), num_query_cams))
        logger.info('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, len(self.gallery), num_gallery_cams))
        logger.info('  ----------------------------------------')
