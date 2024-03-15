from imgaug import augmenters as iaa
import numpy as np

# stupid little numpy deprecation
np.bool = np.bool_


def build_heavy_sequence():
    seq = iaa.Sequential(
        [
            iaa.SomeOf((1, 5), [
                # arithmetic pixel transformation
                iaa.OneOf([

                    # noise
                    iaa.OneOf([
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),

                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                        iaa.AdditivePoissonNoise(
                            lam=(0.0, 15), per_channel=0.5
                        )
                    ]),

                    # pixel dropout(simple+large rectangles)
                    iaa.OneOf([
                        iaa.Dropout(
                            per_channel=0.5, p=(0.0, 0.1)
                        ),
                        iaa.CoarseDropout(
                            p=(0.0, 0.05),
                            per_channel=True,
                            size_percent=(0.02, 0.09)
                        )
                    ]),
                ]),

                # DefocusBlur: the picture was taken with defocused camera
                # MotionBlur: the picture was taken in motion
                iaa.OneOf([
                    iaa.imgcorruptlike.DefocusBlur(severity=1),
                    iaa.imgcorruptlike.MotionBlur(severity=1)
                ]),

                # image quality worsening
                iaa.OneOf([
                    iaa.imgcorruptlike.JpegCompression(severity=2),
                    iaa.imgcorruptlike.Pixelate(severity=2)
                ]),

                # contrast
                iaa.OneOf([
                    iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                ]),

                # gradient brightness+color change
                iaa.OneOf([
                    iaa.BlendAlphaHorizontalLinearGradient(
                        iaa.SomeOf((1, 2), [
                            iaa.Multiply((0.5, 1.5), per_channel=False),
                            iaa.ChangeColorTemperature((2000, 20000))
                        ]),
                        min_value=0.2, max_value=0.8),
                ]),

                iaa.OneOf([
                    # image resize
                    iaa.SomeOf((1, 2), [
                        iaa.ScaleX(scale=(0.5, 1.5)),
                        iaa.ScaleY(scale=(0.5, 1.5))
                    ]),

                    iaa.OneOf([
                        iaa.PerspectiveTransform(scale=(0.01, 0.10), keep_size=False),
                        iaa.PerspectiveTransform(scale=(0.01, 0.10), keep_size=True),
                        iaa.Rotate((-50, 50)),
                        iaa.Rot90((1, 3))
                    ]),
                ]),

                iaa.OneOf([
                    iaa.Invert(0.2, per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                ])
            ]),

        ], random_order=True
    )

    return seq


"""Custom sequence(don't forget to change the param of trans_image_boxes call):"""


def build_custom_sequence():
    seq = iaa.Rotate((-15, 15))

    return seq
