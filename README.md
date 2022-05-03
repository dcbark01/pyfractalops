# Overview

This package implements fractal image compression and feature generation for solving Ravens Progressive Matrix problems.
Using fractal encoding we can do cool things like recover images from pure noise using only their fractal codebook.

![fractal-decoding-from-noise](/img/fractal_decode_noise.png)

This code was developed as part the final project deliverable for Georgia Tech's CS7637 Knowledge-based AI course. The
code has been heavily optimized for speed using multiprocessing and caching. If you're interested in the details of 
fractal compression, see the excellent Welstead book mentioned in the references below.


# Installation

To install, simply use ```pip```:

```bash
pip install pyfractalops
```

# Quickstart

A minimal example of encoding/decoding one image in terms of another is shown below.
```python
from pyfractalops.utils import pil2array, load_image
from pyfractalops.Fractals import fractal_encode_cached, CachedFractalEncodeArgs, fractal_decode

grid_size = 8
imgA = pil2array(load_image('<path to image>'))
imgB = pil2array(load_image('<path to image>'))

codebook_AB = fractal_encode_cached(CachedFractalEncodeArgs([imgA], imgB, grid_size))

iterations_AB = fractal_decode(codebook_AB,
                               source_images=imgA,
                               target_image=imgB,
                               num_iterations=8,
                               show=True)

```

Here's an example using images from a Ravens Progressive Matrix problem:

![rpm-example](/img/basic_problem_c5.png)

![fractal-decoding-raven](/img/fractal_decode_raven.png)


See the ```tests``` dir for more examples of creating fractal features and calculating Tversky similarity using these
fractal codebooks.


# References

[1] McGreggor, Keith, Kunda, Maithilee, and Goel, Ashok (Oct. 2014). “Fractals and Ravens”. en. In: Artificial Intelligence 215, pp. 1–23. issn: 0004-3702. doi: 10.1016/j.artint.2014.05.005. url: http://www.sciencedirect.com/science/article/pii/S0004370214000587.

[2] Stephen T. Welstead. 1999. Fractal and Wavelet Image Compression Techniques (1st. ed.). Society of Photo-Optical Instrumentation Engineers (SPIE), USA.