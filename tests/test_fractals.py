import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(Path(os.path.dirname(__file__)).parent, 'src', 'pyfractalops'))
from pyfractalops.utils import pil2array, load_image
from pyfractalops.FractalFeatures import features_from_fractals, tversky_feature_similarity
from pyfractalops.Fractals import fractal_encode_cached, CachedFractalEncodeArgs, fractal_decode


def test_fractals_encoding_and_decoding():

    grid_size = 8
    test_img_dir = os.path.join(Path(os.path.dirname(__file__)).parent, 'img')

    # Load the analogy premise images
    imgA = pil2array(load_image(os.path.join(test_img_dir, 'basic_c5_iA.png')))
    imgB = pil2array(load_image(os.path.join(test_img_dir, 'basic_c5_iB.png')))

    # Load the analogy conclusion images (second and third images are possible answer choices, for example)
    imgH = pil2array(load_image(os.path.join(test_img_dir, 'basic_c5_iH.png')))
    img1 = pil2array(load_image(os.path.join(test_img_dir, 'basic_c5_i1.png')))
    img2 = pil2array(load_image(os.path.join(test_img_dir, 'basic_c5_i2.png')))

    ts = time.time()
    codebook_AB = fractal_encode_cached(CachedFractalEncodeArgs([imgA], imgB, grid_size))
    codebook_H1 = fractal_encode_cached(CachedFractalEncodeArgs([imgH], img1, grid_size))
    codebook_H2 = fractal_encode_cached(CachedFractalEncodeArgs([imgH], img2, grid_size))
    print(f'Encoding Wall Clock Time (No Caching): {time.time() - ts:.3f}s')

    iterations_AB = fractal_decode(codebook_AB,
                                   source_images=imgA,
                                   target_image=imgB,
                                   num_iterations=8,
                                   show=True)

    iterations_H1 = fractal_decode(codebook_H1,
                                   source_images=imgH,
                                   target_image=img1,
                                   num_iterations=8,
                                   show=True)

    iterations_H2 = fractal_decode(codebook_H1,
                                   source_images=imgH,
                                   target_image=img2,
                                   num_iterations=8,
                                   show=True)

    T_AB = features_from_fractals(codebook_AB)
    T_H1 = features_from_fractals(codebook_H1)
    T_H2 = features_from_fractals(codebook_H2)

    score1 = tversky_feature_similarity(T_AB, T_H1)
    score2 = tversky_feature_similarity(T_AB, T_H2)

    print(f'Fractal Feature Similarity A:B::H:1 = {score1:.3f}')
    print(f'Fractal Feature Similarity A:B::H:2 = {score2:.3f}')


if __name__ == "__main__":
    test_fractals_encoding_and_decoding()
