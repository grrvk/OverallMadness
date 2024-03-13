"""
    Main
"""
from tqdm import tqdm
from generator import layout_functions as lf
from generator.settings import Settings


# -*- coding: utf-8 -*-


def generate(amount: int, settings: Settings):
    print(settings)

    for i in tqdm(range(amount), desc='images created', colour='green'):
        lf.generate_brochure(i + settings.NAMING_INDEX, settings)

    settings.clear_temp_samples()
    print('Finished generating')




