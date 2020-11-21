import pandas as pd
import glob
from p_tqdm import p_map
import urllib.request

dataset_path = '/media/steve/Cache/Documents/Msc Data Science/Thesis/dataset/unsplash/'
save_path = '/media/steve/Data2/unsplash-dataset/images/'
image = urllib.request.URLopener()
documents = ['photos']
datasets = {}
start = 0
stop = 25000


def unsplash_download(i):
    img_name = datasets['photos']['photo_id'][i]
    im = datasets['photos']['photo_image_url'][i]
    image.retrieve(im, save_path + img_name + '.jpg')

def main():

    for doc in documents:
        files = glob.glob(dataset_path + doc + ".tsv*")

        subsets = []
        for filename in files:
            df = pd.read_csv(filename, sep='\t', header=0)
            subsets.append(df)

        datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)

    p_map(unsplash_download, list(range(start, stop)), num_cpus=6)

if __name__ == '__main__':

    main()
