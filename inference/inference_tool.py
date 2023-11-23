import logging
import pdb
import tqdm
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import os
from classname_and_prompt import *
from torchrs.datasets import AID, RESISC45, EuroSATRGB
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from clip_benchmark.datasets.builder import get_dataset_collate_fn
from clip_benchmark.metrics.zeroshot_retrieval import recall_at_k, batchify, dataloader_with_indices
from functools import reduce
import cv2
from scipy.ndimage import maximum_filter
from skimage import measure
import json
from datetime import datetime
from torchvision import transforms


def _convert_to_rgb(image):
    return image.convert('RGB')


def get_preprocess(image_resolution=224, is_train=False, subset_name="clip", aug=None):

    if subset_name == "clip":
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
    elif subset_name == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif subset_name == "rs5m":
        normalize = transforms.Normalize(
            mean=[0.406, 0.423, 0.390], std=[0.188, 0.175, 0.185]
        )

    elif subset_name == "pub11":
        normalize = transforms.Normalize(
            mean=[0.445, 0.469, 0.441], std=[0.208, 0.193, 0.213]
        )

    elif subset_name == "rs3":
        normalize = transforms.Normalize(
            mean=[0.350, 0.356, 0.316], std=[0.158, 0.147, 0.143]
        )

    elif subset_name == "geometa":
        normalize = transforms.Normalize(
            mean=[0.320, 0.322, 0.285], std=[0.179, 0.168, 0.166]
        )

    if is_train:
        preprocess_train = transforms.Compose([
            transforms.RandomResizedCrop(
                image_resolution,
                interpolation=transforms.InterpolationMode.BICUBIC,
                scale=(0.9, 1.0)
            ),
            _convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.ToTensor(),
            normalize,
        ])
        return preprocess_train
    else:
        preprocess_val = transforms.Compose([
            transforms.Resize(
                size=image_resolution,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_resolution),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
        return preprocess_val


def zeroshot_get_dataset(dataset_name, root, split, transform=None):

    if dataset_name == "EuroSAT":
        EuroSAT_root = os.path.join(root, "eurosat-rgb")
        os.makedirs(EuroSAT_root, exist_ok=True)
        dataset = EuroSATRGB(
            root=EuroSAT_root,
            transform=transform
        )
        dataset.classes = dataset.classes
        dataset.templates = RSEuroSAT.templates

    elif dataset_name == "AID":
        AID_root = os.path.join(root, "AID")
        os.makedirs(AID_root, exist_ok=True)
        dataset = AID(
            root=AID_root,
            transform=transform
        )
        dataset.classes = dataset.classes
        dataset.templates = RSAID.templates

    elif dataset_name == "RESISC45":
        RESISC45_root = os.path.join(root, "RESISC45")
        os.makedirs(RESISC45_root, exist_ok=True)
        dataset = RESISC45(
            root=RESISC45_root,
            transform=transform
        )
        dataset.classes = dataset.classes
        dataset.templates = RSRESISC45.templates

    dataset.classes = [dataset.classes[i].replace('_', ' ') for i in range(len(dataset.classes))]
    dataset.classes = [dataset.classes[i].replace('/', ' ') for i in range(len(dataset.classes))]
    dataset.classes = [dataset.classes[i].lower() for i in range(len(dataset.classes))]

    return dataset


def zeroshot_classifier(model, classnames, templates, args):
    tokenizer = open_clip.tokenize
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.replace('{}', classname) for template in templates]
            context_length = 77
            texts = tokenizer(texts, context_length=context_length).to(args.device)

            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embeddings, dim=-1)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding.cpu())
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def zeroshot_evaluation(model, zeroshot_dataset, preprocess, args):

    dataset = zeroshot_get_dataset(dataset_name=zeroshot_dataset, split='test', root=args.test_dataset_dir, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    logging.info(f'Calculating classifier for {zeroshot_dataset}')
    classnames, prompt_templates = dataset.classes, dataset.templates
    import copy
    classnames = copy.deepcopy(classnames)
    classifier = zeroshot_classifier(model, classnames, prompt_templates, args)

    logging.info(f'Calculating image features for {zeroshot_dataset}')
    results = {}
    acc, features, labels = zeroshot_run(model, classifier, dataloader, args)
    logging.info(f'{zeroshot_dataset} zero-shot accuracy: {acc}%')
    results[f'{zeroshot_dataset}-zeroshot-acc'] = acc

    for key, item in results.items():
        results[key] = float(item)

    return results


def zeroshot_accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return float(correct[0].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) * 100 / len(target)


def zeroshot_run(model, classifier, dataloader, args):
    with torch.no_grad():
        all_image_features = []
        all_labels = []
        all_logits = []
        for images, target in tqdm.tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1).detach().cpu()
            logits = 100. * image_features @ classifier
            all_image_features.append(image_features)
            all_labels.append(target)
            all_logits.append(logits)

    all_image_features = torch.cat(all_image_features)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    acc = zeroshot_accuracy(all_logits, all_labels, topk=(1,))
    return round(acc, 2), all_image_features, all_labels


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", nori_dataset=False,
                 images_dir=''):
        logging.debug(f'Loading csv data from {input_filename}.')
        if 'rsicd' in input_filename:
            df = pd.read_csv(input_filename, sep=sep, encoding='gb18030')
        else:
            df = pd.read_csv(input_filename, sep=sep)

        self.nori_dataset = nori_dataset
        self.f = None
        self.images_dir = images_dir

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()

        self.transforms = transforms

        self.duplicate()

        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        texts = self.captions[index]
        image = Image.open(os.path.join(self.images_dir, str(self.images[index])))
        image = self.transforms(image)

        return image, texts

    def duplicate(self):
        unique_images, indexs = np.unique(self.images, return_index=True)
        if len(unique_images) != len(self.images):
            logging.debug(
                f'Amoung all {len(self.images)} images, there are only {len(unique_images)} unique images. Dupication will be performed to enable one-image-to-multiple-text retrieval.')
            self.duplicated_images = []
            self.duplicated_captions = []
            for index in indexs:
                self.duplicated_images.append(self.images[index])
                same_indexs = [i for i, x in enumerate(self.images) if x == self.images[index]]
                captions = []
                for same_index in same_indexs:
                    captions.append(self.captions[same_index])
                self.duplicated_captions.append(captions)

            self.images = self.duplicated_images
            self.captions = self.duplicated_captions


def retrieval_evaluation(model, preprocess, args, recall_k_list=[1, 5, 10], dataset_name=None):
    """
    Modified from https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_retrieval.py
    Evaluate the model on the given dataset

    Parameters
    ----------

    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    device: cpu/cuda
    recall_k_list: list of int
        recall@k k's to use

    Returns
    -------

    dict of retrieval metrics
    """

    if dataset_name == "rsitmd":
        dataset = CsvDataset(
            input_filename=os.path.join(args.test_dataset_dir, "rsitmd", "rsitmd_test.csv"),
            transforms=preprocess,
            img_key="filename",
            caption_key="title",
            sep=",",
            images_dir=os.path.join(args.test_dataset_dir, "rsitmd", "images")
        )
    elif dataset_name == "rsicd":
        dataset = CsvDataset(
            input_filename=os.path.join(args.test_dataset_dir, "rsicd", "rsicd_test.csv"),
            transforms=preprocess,
            img_key="filename",
            caption_key="title",
            sep=",",
            images_dir=os.path.join(args.test_dataset_dir, "rsicd", "RSICD_images")
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=get_dataset_collate_fn('mscoco_captions')
    )
    n_batches = len(dataloader)
    tokenizer = open_clip.tokenize
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)

    for batch_images, batch_texts, inds in tqdm.tqdm(dataloader, total=n_batches):
        batch_images = batch_images.to(args.device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
        # tokenize all texts in the batch
        batch_texts = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(args.device)

        # compute the embedding of images and texts
        with torch.no_grad():
            batch_image_features = model.encode_image(batch_images)
            batch_text_features = model.encode_text(batch_texts)
            batch_images_emb = F.normalize(batch_image_features, dim=-1)
            batch_texts_emb = F.normalize(batch_text_features, dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        '''
        Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        for each image, that number will be greater than 1 for text retrieval.
        However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        it over the dataset.
        '''
        metrics[f"retrieval-image2text-R@{recall_k}-{dataset_name}"] = (batchify(recall_at_k, scores.T,
                                                                                 positive_pairs.T, batch_size,
                                                                                 args.device,
                                                                                 k=recall_k) > 0).float().mean().item() * 100

    for recall_k in recall_k_list:
        metrics[f"retrieval-text2image-R@{recall_k}-{dataset_name}"] = (batchify(recall_at_k, scores, positive_pairs,
                                                                                 batch_size, args.device,
                                                                                 k=recall_k) > 0).float().mean().item() * 100

    metrics[f"retrieval-mean-recall-{dataset_name}"] = np.mean(list(metrics.values()))

    for key, item in metrics.items():
        metrics[key] = round(float(item), 2)
    logging.info(f'{dataset_name} retrieval recall: {metrics}%')

    return metrics


class SLM(object):

    # **
    # * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
    #
    # @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
    #         2022/03/08

    def __init__(self):
        # logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()

        # parameters
        self.rsu_beta = 0.707
        self.rsu_eps = 1e-7

        self.ras_expand_factor = 1.5
        self.ras_filter_times = 5
        self.ras_scala_beta = 3

        self.rda_eta = 0.5

        self.rmi_wsu = 0.4
        self.rmi_was = 0.35
        self.rmi_wda = 0.25

        # visual settings
        self.visual_ras = False
        self.src_addmap_path = None

        # sum indicator
        self.all_metrics = self._format_output_dict()

    def _format_output_dict(self, *params):
        """
        format output dict
        :param params: keys
        :return: format dict
        """
        len_params = len(params)
        if len_params == 0: init_param = [[] for i in range(4)]
        elif len_params == 4: init_param = params
        else: raise NotImplementedError

        return {
            "↑ Rsu [0 ~ 1]": init_param[0],
            "↑ Rda [0 ~ 1]": init_param[1],
            "↓ Ras [0 ~ 1]": init_param[2],
            "↑ Rmi [0 ~ 1]": init_param[3]
        }

    def logging_acc(self, metrics_dict, prob_path = None, ave = False):
        """
        logging the metrics
        :param metrics_dict: dict of metrics
        :param prob_path: path
        :return: 0
        """

        if not ave:
            self.logger.info("Eval {}".format(prob_path))
        else:
            self.logger.info("+++++++++++++++Average++++++++++++++")

        self.logger.info("+++++++ Calc the SLM METRICS +++++++")
        for metric, value in metrics_dict.items():
            self.logger.info("++++     {}:{:.4f}   ++++".format(metric, value))
        self.logger.info("++++++++++++++++++++++++++++++++++++\n")

    def set_visual_options(self, visual_ras, src_addmap_path):
        """
        set visual options
        :param visual_ras: flag
        :param src_addmap_path: set src addmap path
        """
        self.visual_ras = visual_ras
        self.src_addmap_path = src_addmap_path
        return True

    def read_gray_to_prob(self, probmap_path):
        """
        Read the prob maps, and trans to probility
        :param probmap_path: probmap routh
        :return: probability
        """
        gray_image = cv2.imread(probmap_path, cv2.IMREAD_GRAYSCALE)
        prob = gray_image / 255.0
        return prob

    def generate_mask_by_points(self, prob, points_list):
        """
        Generate mask by regions
        :param prob: probability
        :param points_list: regions
        :return: mask
        """
        H, W = prob.shape

        mask = np.zeros((H, W))
        points_list = [np.array(i, np.int32) for i in points_list]
        # fill
        cv2.fillPoly(mask, points_list, 1)
        return mask

    def _get_region_center_radius(self, region_point):
        """
        get the region center and radius
        :param region_point: regions
        :return: mid_x, mid_y, radius
        """
        mid_x = int(reduce(lambda x, y: x+y, np.array(region_point)[:, 0]) / len(region_point))
        mid_y = int(reduce(lambda x, y: x+y, np.array(region_point)[:, 1]) / len(region_point))
        radius = int(np.mean([np.linalg.norm(np.array(point) - np.array([mid_x, mid_y])) for point in region_point]) * self.ras_expand_factor)
        return mid_x, mid_y, radius

    def _get_prob_center_in_gray(self, prob):
        """
        get the top point with the highest probability from the probability map
        :param prob: probability
        :return: centers
        """

        # recover the prob
        gray_img = np.asarray(prob * 255.0, dtype=np.uint8)
        # cv2.imwrite("./gray_img.jpg", gray_img)
        # construct continuous area
        continuous_area = np.asarray(gray_img > 150, np.uint8) * 255
        # cv2.imwrite("./continuous_area_img_0.jpg", continuous_area)
        continuous_area = np.uint8(measure.label(continuous_area, connectivity=2))
        # cv2.imwrite("./continuous_area_img_1.jpg", continuous_area)

        # soften
        for i in range(self.ras_filter_times):
            gray_img = cv2.boxFilter(gray_img, ddepth=-1, ksize=(50, 50))

        # get probability binary map
        mx = maximum_filter(gray_img, size=1000)
        gray_img = np.where(mx == gray_img, gray_img, 0)
        # cv2.imwrite("./local_maxima_before_filter.jpg", gray_img)
        gray_img = np.asarray(gray_img > 0, np.uint8) * 255
        # cv2.imwrite("./local_maxima_after_filter.jpg", gray_img)

        # get probability area information
        labels = measure.label(gray_img, connectivity=2)
        all_region_infos = measure.regionprops(labels)
        centers = [[int(i) for i in prop.centroid][::-1] for prop in all_region_infos]

        # construct v-center list and sort
        v_center = [[c[0], c[1], prob[c[1]][c[0]]] for c in centers]
        v_center.sort(key= lambda x: x[2], reverse=True)
        centers = list(map(lambda x: x[:2], v_center))

        # filter centers
        centers = [i for i in centers if prob[i[1]][i[0]] >= 0.5]

        return centers, continuous_area

    def _get_offset_between_real_and_synthetic(self, real_center_radius, prob_centers, bina_img):
        """
        calculate true center offset from result center
        :param real_center_radius: real_center_radius
        :param prob_centers: prob_centers
        :return: offsets
        """

        # check prob_centers is not None
        if len(prob_centers) == 0 : return [real_center_radius[0][2]]

        offsets = []
        for center_radius in real_center_radius:
            x, y, r = center_radius

            # calc the l2 dis
            dises = list(map(lambda p: np.linalg.norm(np.array([x, y] - np.array(p))), prob_centers))

            # filter the dis in cicle
            dises = list(filter(lambda d: d <= r, dises))

            # if no prob center set it to radius
            offsets.append(np.mean(dises) if len(dises) != 0 else r)

        return offsets

    def _trans_ras_offset_to_scalable_ras(self, offsets, centers_and_radius):
        """
        convert distance offset to ras value
        :param offsets: offsets
        :return: centers_and_radius
        """

        # granular transformation
        granular_offet = np.mean([off/v[2] for off, v in zip(offsets, centers_and_radius)])

        # scala transformation
        granular_offet = (np.exp(self.ras_scala_beta * granular_offet) - 1) / (np.exp(self.ras_scala_beta) - 1)

        return granular_offet

    def ras(self, region_lists, prob, visual=True, src_img=None):
        """
        calc the matric of ras: makes attention center close to annotation center
        :param region_lists: regions
        :param prob: probability
        :return: ras
        """

        # get the annotation center and radius
        centers_and_radius = [self._get_region_center_radius(i) for i in region_lists]

        # get the point with the highest probability from the probability map
        prob_centers, bina_img = self._get_prob_center_in_gray(prob)

        # calculate true center offset from result center
        offsets = self._get_offset_between_real_and_synthetic(centers_and_radius, prob_centers, bina_img)

        # convert distance offset to rcs value
        ras = self._trans_ras_offset_to_scalable_ras(offsets, centers_and_radius)

        # visual
        if visual and (src_img != None):
            src_img = cv2.imread(src_img)

            # logging something
            # print("centers_and_radius: ", centers_and_radius)
            # print("prob_centers: ", prob_centers)
            # print("offsets: ", offsets)

            # backup area
            for c_r in centers_and_radius:
                cv2.circle(src_img, (c_r[0], c_r[1]), c_r[2], 2, 3)

            # candidate points
            for idx, point in enumerate(prob_centers):
                cv2.circle(src_img, tuple(point), 6*(idx+1), 1, 4)
                cv2.putText(src_img, str(idx+1), tuple(point), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 0), 25)

            cv2.imwrite("./img_circle.jpg", src_img)

            # print(prob_centers)

        return ras

    def rsu(self, prob, mask):
        """
        calc the salient area proportion
        :param prob: probability
        :param mask: mask
        :return: rsu
        """

        all_mask_value = np.sum(np.multiply(prob, mask))
        all_value = np.sum(prob)
        H, W = np.shape(mask)
        all_mask = np.sum(mask)

        left_frac = all_mask_value / (all_value - all_mask_value + self.rsu_eps)

        right_frac = (H * W - all_mask) / all_mask

        rsu = -np.exp(-1 * self.rsu_beta * left_frac * right_frac) + 1

        return rsu

    def rda(self, region_lists, prob):
        """
        calc the matric of rda: makes attention center focus on one point
        :param region_lists: regions
        :param prob: probability
        :return: rda
        """

        # get the annotation center and radius
        centers_and_radius = [self._get_region_center_radius(i) for i in region_lists]

        # get the point with the highest probability from the probability map
        prob_centers, bina_img = self._get_prob_center_in_gray(prob)

        # set value
        rda = []
        for c_r in centers_and_radius:
            x, y, r = c_r

            # calc the backup points
            backup_points = list(filter(lambda p: np.linalg.norm(np.array([x, y] - np.array(p))) <= r, prob_centers))

            # margin condition
            len_backup_points = len(backup_points)
            if len_backup_points <= 1 :
                rda.append(float(len_backup_points))
                continue

            # if len_backup_points >= 2, calc the attention discrete
            centers_attention = np.average(backup_points, axis=0)
            dises = list(map(lambda p: np.linalg.norm(np.array(centers_attention - np.array(p))), backup_points))
            meas_dis = np.mean(dises) / r

            rda_single = 0.5 * (1 - meas_dis) + np.exp(- self.rda_eta * (len_backup_points + 2))

            rda.append(rda_single)

        return np.mean(rda)

    def rmi(self, rsu, rda, ras):
        """
        calculate the mean indicator
        :param rsu: rsu
        :param rda: rda
        :param ras: ras
        :return: rmi
        """
        return self.rmi_wsu * rsu + self.rmi_was * (1 - ras) + self.rmi_wda * rda

    def evaluate(self, prob_path, region_list):
        """
        evaluate the slm task
        :param probmap_path: probability map path
        :param region_list: region points
        :return: slm metrics
        """
        # read prob
        prob = self.read_gray_to_prob(prob_path)

        # generate mask
        mask = self.generate_mask_by_points(prob, region_list)
        # import os
        # cv2.imwrite(os.path.join(prob_path.rsplit("/", 1)[0], "maskbypt_0.jpg"), mask*255)
        # rsu
        rsu = self.rsu(prob, mask)

        # ras
        ras = self.ras(region_list, prob, visual=self.visual_ras, src_img=self.src_addmap_path)

        # rda
        rda = self.rda(region_list, prob)

        # mi
        rmi = self.rmi(rsu, rda, ras)

        # sort metrics
        metrics = self._format_output_dict(rsu, rda, ras, rmi)
        # self.logging_acc(metrics, prob_path)

        return metrics

    def append_metric(self, metric):
        """
        append metric to calc ave indicator
        :param metric: sort metrics
        """
        for k in metric.keys():
            self.all_metrics[k].append(metric[k])

    def get_the_mean_metric(self):
        """
        get the mean metric
        """
        mean_metric = {}
        for k in self.all_metrics:
            mean_metric[k] = np.mean(self.all_metrics[k])

        self.logging_acc(mean_metric, ave=True)
        return mean_metric


def semantic_localization_evaluation(model, selo_dataset, preprocess, args):
    assert selo_dataset == 'AIR-SLT'

    def collect_fn_selo(batch):
        assert len(batch) == 1
        source_img, subimages, text, points, subimg_name_list = batch[0]
        return source_img, subimages, text, points, subimg_name_list

    dataset = get_selo_dataset(
        root=args.test_dataset_dir, transform=preprocess, identifier=None
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collect_fn_selo
    )
    tokenizer = open_clip.tokenize
    logger = dataset.logger
    slm_metric = SLM()

    with torch.no_grad():
        for idx, sample in tqdm.tqdm(enumerate(dataloader)):
            source_img, subimages, text, points, subimg_name_list = sample
            subimages = subimages.to(args.device)
            text = tokenizer(text).to(args.device)
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            sim_results = []
            for subimage in subimages:
                subimage = subimage.unsqueeze(0)
                sub_img_feat = model.encode_image(subimage)
                sub_img_feat /= sub_img_feat.norm(dim=-1, keepdim=True)
                similarity = (sub_img_feat * text_features).sum().detach().cpu().numpy()
                sim_results.append(similarity)

            # print("Start generate heatmap ...")
            img_row = np.shape(source_img)[0]
            img_col = np.shape(source_img)[1]

            # mkdir map
            heat_map = np.zeros([img_row, img_col], dtype=float)
            heat_num = np.zeros([img_row, img_col], dtype=float)
            for idx, file in enumerate(subimg_name_list):
                r_start, r_end, c_start, c_end = file.replace(".jpg", "").split("_")
                heat_map[int(r_start):int(r_end), int(c_start):int(c_end)] += sim_results[idx]
                heat_num[int(r_start):int(r_end), int(c_start):int(c_end)] += 1

            for i in range(np.shape(heat_map)[0]):
                for j in range(np.shape(heat_map)[1]):
                    heat_map[i, j] = heat_map[i, j] / heat_num[i, j]

            # logger.info("Generation finished, start operating blur, colormap, etc. ...")
            # filter
            adaptive = np.asarray(heat_map)
            adaptive = adaptive - np.min(adaptive)
            probmap = adaptive / np.max(adaptive)
            # must convert to type unit8
            probmap = np.uint8(255 * probmap)
            probmap = cv2.medianBlur(probmap, 251)
            heatmap = cv2.applyColorMap(probmap, cv2.COLORMAP_JET)
            img_add = cv2.addWeighted(source_img, 0.7, heatmap, 0.3, 0)

            probmap_path = os.path.join(dataset.cache_path, "probmap_{}.jpg".format(idx))
            heatmap_path = os.path.join(dataset.cache_path, "heatmap_{}.jpg".format(idx))
            addmap_path = os.path.join(dataset.cache_path, "addmap_{}.jpg".format(idx))

            # logger.info("Saving heatmap in {} ...".format(heatmap_path))
            # logger.info("Saving probmap in {} ...".format(probmap_path))
            # logger.info("Saving addmap in {} ...".format(addmap_path))

            cv2.imwrite(probmap_path, probmap)
            cv2.imwrite(heatmap_path, heatmap)
            cv2.imwrite(addmap_path, img_add)
            # logger.info("Saved ok.")

            metrics = slm_metric.evaluate(probmap_path, region_list=points)
            slm_metric.append_metric(metrics)

        mean_metric = slm_metric.get_the_mean_metric()

    results = {}
    logging.info(f'{selo_dataset} selo metrics: {mean_metric}')

    for key, item in mean_metric.items():
        results[key] = float(item)

    return results


class AIR_SLT(Dataset):
    # Ref: https://github.com/xiaoyuan1996/SemanticLocalizationMetrics/blob/master/predict/generate_selo.py
    def __init__(self, root, subimage_transform, identifier):
        super().__init__()
        self.json_path = os.path.join(root, "annotations", "anno.json")
        # self.cache_path = os.path.join(root, "selo_cache_{}_{}".format(identifier, str(datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")))
        self.cache_path = os.path.join(root, "selo_cache")
        os.makedirs(self.cache_path, exist_ok=True)
        with open(self.json_path, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)
        self.img_root = os.path.join(root, "imgs")
        self.subimage_transform = subimage_transform
        self.logger = get_logger(os.path.join(self.cache_path, 'log.txt'))
        self.step = "256_512_768"

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        item = self.json_data[index]
        img_name = item['jpg_name']
        text = item['caption']
        points = item['points']
        steps = [int(step) for step in self.step.split("_")]
        img_path = os.path.join(self.img_root, img_name)

        # logging
        # self.logger.info("Processing {}/{}: {}".format(index, len(self.json_data), img_name))
        # self.logger.info("Corresponding text: {}".format(text))

        # processing
        self.split_image(img_path, steps)
        with torch.no_grad():
            subimages_dir = os.path.join(self.cache_path, os.path.basename(img_path).split(".")[0]) + '_subimages'
            subimages = os.listdir(subimages_dir)

            img = cv2.imread(img_path)
            subimg_list = []
            subimg_name_list = []
            for subimage_name in subimages:
                subimage_path = os.path.join(subimages_dir, subimage_name)
                subimg = Image.open(subimage_path)
                subimg = self.subimage_transform(subimg).unsqueeze(0)
                subimg_list.append(subimg)
                subimg_name_list.append(subimage_name)
            subimgs = torch.vstack(subimg_list)
        return img, subimgs, [text], points, subimg_name_list

    def split_image(self, img_path, steps):
        subimage_files_dir = os.path.join(self.cache_path, os.path.basename(img_path).split(".")[0])

        # 裁切图像文件夹
        subimages_dir = subimage_files_dir + '_subimages'
        if os.path.exists(subimages_dir):
            delete_dire(subimages_dir)
        else:
            os.makedirs(subimages_dir)

        # Read Image
        source_img = cv2.imread(img_path)
        img_weight = np.shape(source_img)[0]
        img_height = np.shape(source_img)[1]
        # self.logger.info("img size:{}x{}".format(img_weight, img_height))

        for step in steps:
            # self.logger.info("Start split images with step {}".format(step))
            for gap in [step, 0.5 * step]:
                gap = int(gap)

                # Cut img
                for h in range(0 + (step - gap), img_height, step):
                    h_start, h_end = h, h + step
                    # bound?
                    if h_end >= img_height:
                        h_start, h_end = img_height - step, img_height

                    for w in range(0 + (step - gap), img_weight, step):
                        w_start, w_end = w, w + step
                        # bound?
                        if w_end >= img_weight:
                            w_start, w_end = img_weight - step, img_weight

                        cut_img_name = str(w_start) + "_" + str(w_end) + "_" + str(h_start) + "_" + str(h_end) + ".jpg"
                        cut_img = source_img[w_start:w_end, h_start:h_end]
                        cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                        cv2.imwrite(os.path.join(subimages_dir, cut_img_name), cut_img)

        # self.logger.info("Image {} has been split successfully.".format(img_path))


def delete_dire(dire):
    dir_list = []
    for root, dirs, files in os.walk(dire):
        for afile in files:
            os.remove(os.path.join(root, afile))
        for adir in dirs:
            dir_list.append(os.path.join(root, adir))
    for bdir in dir_list:
        os.rmdir(bdir)


# logger
def get_logger(save_path=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置打印级别
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # 设置屏幕打印的格式
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 设置log保存
    if save_path != None:
        fh = logging.FileHandler(save_path, encoding='utf8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_selo_dataset(root, transform, identifier):

    AIR_SLT_root = os.path.join(root, "AIR-SLT")
    dataset = AIR_SLT(
        root=AIR_SLT_root,
        subimage_transform=transform,
        identifier=identifier
    )

    return dataset