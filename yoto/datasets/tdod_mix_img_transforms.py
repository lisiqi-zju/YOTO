# Copyright (c) Tencent Inc. All rights reserved.
import collections
import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import autocast_box_type
from mmengine.dataset import BaseDataset
from mmengine.dataset.base_dataset import Compose
from numpy import random
from mmyolo.registry import TRANSFORMS


class BaseMultiModalMixImageTransform(BaseTransform, metaclass=ABCMeta):
    """A Base Transform of Multimodal multiple images mixed.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup.

    Cached mosaic transform will random select images from the cache
    and combine them into one output image if use_cached is True.

    Args:
        pre_transform(Sequence[str]): Sequence of transform object or
            config dict to be composed. Defaults to None.
        prob(float): The transformation probability. Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(self,
                 pre_transform: Optional[Sequence[str]] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 40,
                 random_pop: bool = True,
                 max_refetch: int = 15):

        self.max_refetch = max_refetch
        self.prob = prob

        self.use_cached = use_cached
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.results_cache = []

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    @abstractmethod
    def get_indexes(self, dataset: Union[BaseDataset,
                                         list]) -> Union[list, int]:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list or int: indexes.
        """
        pass

    @abstractmethod
    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        pass

    def _update_label_text(self, results: dict) -> dict:
        """Update label text."""
        if 'texts' not in results:
            return results

        mix_texts = sum(
            [results['texts']] +
            [x['texts'] for x in results['mix_results']], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for res in [results] + results['mix_results']:
            for i, label in enumerate(res['gt_bboxes_labels']):
                text = res['texts'][label]
                updated_id = text2id[tuple(text)]
                res['gt_bboxes_labels'][i] = updated_id
            res['texts'] = mix_texts
        return results

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Data augmentation function.

        The transform steps are as follows:
        1. Randomly generate index list of other images.
        2. Before Mosaic or MixUp need to go through the necessary
            pre_transform, such as MixUp' pre_transform pipeline
            include: 'LoadImageFromFile','LoadAnnotations',
            'Mosaic' and 'RandomAffine'.
        3. Use mix_img_transform function to implement specific
            mix operations.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if self.use_cached:
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            dataset = results.pop('dataset', None)
            self.results_cache.append(copy.deepcopy(results))
            if len(self.results_cache) > self.max_cached_images:
                if self.random_pop:
                    index = random.randint(0, len(self.results_cache) - 1)
                else:
                    index = 0
                self.results_cache.pop(index)

            if len(self.results_cache) <= 4:
                return results
        else:
            assert 'dataset' in results
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            dataset = results.pop('dataset', None)

        for _ in range(self.max_refetch):
            # get index of one or three other images
            if self.use_cached:
                indexes = self.get_indexes(self.results_cache)
            else:
                indexes = self.get_indexes(dataset)

            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]

            if self.use_cached:
                mix_results = [
                    copy.deepcopy(self.results_cache[i]) for i in indexes
                ]
            else:
                # get images information will be used for Mosaic or MixUp
                mix_results = [
                    copy.deepcopy(dataset.get_data_info(index))
                    for index in indexes
                ]

            if self.pre_transform is not None:
                for i, data in enumerate(mix_results):
                    # pre_transform may also require dataset
                    data.update({'dataset': dataset})
                    # before Mosaic or MixUp need to go through
                    # the necessary pre_transform
                    _results = self.pre_transform(data)
                    _results.pop('dataset')
                    mix_results[i] = _results

            if None not in mix_results:
                results['mix_results'] = mix_results
                break
            print('Repeated calculation')
        else:
            raise RuntimeError(
                'The loading pipeline of the original dataset'
                ' always return None. Please check the correctness '
                'of the dataset and its pipeline.')

        # print(results['gt_coco_labels'].size==results['gt_bboxes_labels'].size)

        # update labels and texts
        results = self._update_label_text(results)

        # print(results['gt_coco_labels'].size==results['gt_bboxes_labels'].size)
        
        # Mosaic or MixUp
        results = self.mix_img_transform(results)

        if 'mix_results' in results:
            results.pop('mix_results')
        results['dataset'] = dataset

        return results


@TRANSFORMS.register_module()
class TDODMultiModalMosaic(BaseMultiModalMixImageTransform):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |           |
                |      +-----------+    pad    |
                |      |           |           |
                |      |  image1   +-----------+
                |      |           |           |
                |      |           |   image2  |
     center_y   |----+-+-----------+-----------+
                |    |   cropped   |           |
                |pad |   image3    |   image4  |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 40,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
        if use_cached:
            assert max_cached_images >= 4, 'The length of cache must >= 4, ' \
                                           f'but got {max_cached_images}.'

        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)

        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> list:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list: indexes.
        """
        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        # print("1",results['gt_coco_labels'].size==results['gt_bboxes_labels'].size)

        assert 'mix_results' in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_masks = []
        mosaic_coco_labels = []

        with_mask = True if 'gt_masks' in results else False
        # print("with_mask: ", with_mask)
        # self.img_scale is wh format
        img_scale_w, img_scale_h = self.img_scale

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(img_scale_h * 2), int(img_scale_w * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full((int(img_scale_h * 2), int(img_scale_w * 2)),
                                 self.pad_val,
                                 dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * img_scale_w)
        center_y = int(random.uniform(*self.center_ratio_range) * img_scale_h)
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = results
            else:
                results_patch = results['mix_results'][i - 1]

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(img_scale_h / h_i, img_scale_w / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']
            gt_coco_labels_i = results_patch['gt_coco_labels']


            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
            mosaic_coco_labels.append(gt_coco_labels_i)


            if with_mask and results_patch.get('gt_masks', None) is not None:
                gt_masks_i = results_patch['gt_masks']
                gt_masks_i = gt_masks_i.rescale(float(scale_ratio_i))
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padw,
                    direction='horizontal')
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padh,
                    direction='vertical')
                mosaic_masks.append(gt_masks_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)
        mosaic_coco_labels = np.concatenate(mosaic_coco_labels, 0)
        

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * img_scale_h, 2 * img_scale_w])
            if with_mask:
                mosaic_masks = mosaic_masks[0].cat(mosaic_masks)
                results['gt_masks'] = mosaic_masks
        # else:
        #     # remove outside bboxes
        #     inside_inds = mosaic_bboxes.is_inside(
        #         [2 * img_scale_h, 2 * img_scale_w]).numpy()
        #     mosaic_bboxes = mosaic_bboxes[inside_inds]
        #     mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        #     mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        #     if with_mask:
        #         mosaic_masks = mosaic_masks[0].cat(mosaic_masks)[inside_inds]
        #         results['gt_masks'] = mosaic_masks

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        results['gt_coco_labels'] = mosaic_coco_labels

        return results

    def _mosaic_combine(
            self, loc: str, center_position_xy: Sequence[float],
            img_shape_wh: Sequence[int]) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str

@TRANSFORMS.register_module()
class TDODMultiModalMixUp(BaseMultiModalMixImageTransform):
    """MixUp data augmentation for YOLOv5.

    .. code:: text

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset.
        2. Randomly obtain the fusion ratio from the beta distribution,
            then fuse the target
        of the original image and mixup image through this ratio.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        alpha (float): parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        beta (float):  parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        pre_transform (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """

    def __init__(self,
                 alpha: float = 32.0,
                 beta: float = 32.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        if use_cached:
            assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                           f'but got {max_cached_images}.'
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)
        self.alpha = alpha
        self.beta = beta

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            int: indexes.
        """
        return random.randint(0, len(dataset))

    def mix_img_transform(self, results: dict) -> dict:
        """YOLOv5 MixUp transform function.

        Args:
            results (dict): Result dict

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']
        ori_img = results['img']
        assert ori_img.shape == retrieve_img.shape

        # Randomly obtain the fusion ratio from the beta distribution,
        # which is around 0.5
        ratio = np.random.beta(self.alpha, self.beta)
        mixup_img = (ori_img * ratio + retrieve_img * (1 - ratio))

        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']
        retrieve_gt_coco_labels = retrieve_results['gt_coco_labels']



        mixup_gt_bboxes = retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)
        mixup_gt_coco_labels = np.concatenate(
            (results['gt_coco_labels'], retrieve_gt_coco_labels), axis=0)  
        if 'gt_masks' in results:
            assert 'gt_masks' in retrieve_results
            mixup_gt_masks = results['gt_masks'].cat(
                [results['gt_masks'], retrieve_results['gt_masks']])
            results['gt_masks'] = mixup_gt_masks

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags
        results['gt_coco_labels'] = mixup_gt_coco_labels
        return results


