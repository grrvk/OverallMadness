from tqdm import tqdm
import pandas as pd

from generator.settings import setGenSettings
from generator import layout_functions as lf

from data2aug.aug2coco import gen2aug

from aug2coco.settings import setConvSettings
from aug2coco.loader import Loader

AMOUNT = 2
AUG_TIMES = 1

generator_conf = setGenSettings(samples_path='/Users/vika/Desktop/samples_full', dict_output=True)
convertor_conf = setConvSettings(split_type='train/val', split_rate='0.7/0.3', df_input=True, dataset_type='semantic',
                                 upload=True, return_path='df_generated')
df = pd.DataFrame(columns=['image', 'category', 'bbox', 'segmentation', 'area', 'PIL'])

for i in tqdm(range(AMOUNT), desc="images", colour='red'):
    data = lf.generate_brochure(i + 1, generator_conf)
    aug_data = gen2aug(data, AUG_TIMES)

    df = pd.concat([df, pd.DataFrame.from_dict(aug_data)], ignore_index=False)

loader = Loader(convertor_conf)
loader.load(df)
