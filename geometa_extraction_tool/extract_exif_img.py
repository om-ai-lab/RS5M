from pprint import pprint

import PIL.Image
from PIL import Image
import piexif
import os
from tqdm import tqdm
import pickle as pkl
PIL.Image.MAX_IMAGE_PIXELS = None

codec = 'ISO-8859-1'  # or latin-1


def exif_to_tag(exif_dict):
    exif_tag_dict = {}
    thumbnail = exif_dict.pop('thumbnail')
    exif_tag_dict['thumbnail'] = thumbnail.decode(codec)

    for ifd in exif_dict:
        exif_tag_dict[ifd] = {}
        for tag in exif_dict[ifd]:
            try:
                element = exif_dict[ifd][tag].decode(codec)

            except AttributeError:
                element = exif_dict[ifd][tag]

            exif_tag_dict[ifd][piexif.TAGS[ifd][tag]["name"]] = element

    return exif_tag_dict


def main_exif():
    img_dir = "/media/zilun/mx500/MillionAID1/test"
    all_imgs = os.listdir(img_dir)
    select_img = all_imgs
    exif_img_list = []
    exif_info_count = 0
    for idx, img_filename in tqdm(enumerate(select_img)):
        filepath = os.path.join(img_dir, img_filename)
        im = Image.open(filepath)
        exif_info = im.info.get('exif')
        if exif_info:
            exif_img_list.append(img_filename)
            try:
                exif_dict = piexif.load(exif_info)
                exif_dict = exif_to_tag(exif_dict)
                pprint(exif_dict['GPS'])
                exif_info_count += 1
            except:
                print(exif_info)
        # else:
        #     print("{}, None exif info: {}".format(idx, filepath))
    print(exif_info_count)
    pkl.dump(exif_img_list, open("exif_img_list.pkl", "wb"))


def main():
    main_exif()


if __name__ == '__main__':
    main()
    print(pkl.load(open("/home/zilun/RS5M_v4/nips_rebuttal/geometa/exif_img_list.pkl", "rb")))