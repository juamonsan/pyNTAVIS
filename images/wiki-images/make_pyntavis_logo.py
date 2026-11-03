"""
Generates the pyNTAVIS logo from the layers of the original pyNAVIS logo (pynavis_logo.psd): the same frame, gradient
and N shape, with the N squeezed vertically and repeated three times, one for each axis. The four gaps (frame - N, N - N,
N - N, N - frame) are equal.

The layer effects of the PSD are rebuilt here, because they cannot be rendered from the file: the outside stroke and the
outer bevel of the N, and the outside stroke and the inner glow of the frame. Their parameters were tuned so that
rebuilding the original logo (one N, not squeezed) matches pynavis_logo.png (mean difference below 3/255).

Outputs:
    images/wiki-images/pyntavis_logo.png      600 x 600, for the README
    docs/source/pyNTAVIS_200.png              200 x 200, logo of the documentation
    docs/source/pyNTAVIS_favicon.png          32 x 32, favicon of the documentation

Requirements (only for this script): numpy, scipy, pillow and psd-tools.

Usage (from the repository root):
    python images/wiki-images/make_pyntavis_logo.py
"""

import os

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PSD_PATH = os.path.join(HERE, 'pynavis_logo.psd')
OUTPUTS = {os.path.join(HERE, 'pyntavis_logo.png'): 600,
           os.path.join(ROOT, 'docs', 'source', 'pyNTAVIS_200.png'): 200,
           os.path.join(ROOT, 'docs', 'source', 'pyNTAVIS_favicon.png'): 32}

SIZE = 600          # size of the PSD
SS = 3              # supersampling factor used while drawing

# Layer effects (sizes in pixels of the 600 x 600 logo)
EFFECTS = dict(bevel_size=10, soften=1.0, depth=1.0, hi_gain=0.3, sh_gain=1.4, n_stroke=3, glow_size=4, glow_soft=1.5,
               f_stroke=2, join=3)
BEVEL_ANGLE, BEVEL_ALTITUDE = 120, 30                  # degrees, from the PSD
BEVEL_HIGHLIGHT = np.array([230, 224, 183]) / 255.     # screen, 50 %; shadow: black, multiply, 75 %

# Layout of the three Ns
FRAME_INNER_TOP, FRAME_INNER_HEIGHT = 30, 540          # inner area of the frame: rows 30-569
N_HEIGHT = 133.3                                       # height of each squeezed N (the original N is 428 px high)


def load_layers():
    psd = PSDImage.open(PSD_PATH)
    layers = {layer.name: layer for layer in psd}

    def full(name):
        layer = layers[name]
        a = np.zeros((SIZE, SIZE, 4))
        im = np.asarray(layer.topil().convert('RGBA')) / 255.
        a[layer.top:layer.top + im.shape[0], layer.left:layer.left + im.shape[1]] = im
        return a

    return dict(background_gradient=full('Layer 8'), background_opacity=layers['Layer 8'].opacity / 255.,
                n=full('Layer 9 copy 3'), frame=full('Layer 1'), gradient=full('Layer 2'))


def up(a):
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((SIZE * SS,) * 2, Image.BICUBIC)) / 255.


def up_rgb(a):
    return np.stack([up(a[..., c]) for c in range(3)], -1)


def dist_out(alpha):
    """Distance from each pixel outside the shape to the shape, in pixels of the 600 x 600 logo."""
    return ndimage.distance_transform_edt(alpha < 0.5) / SS


def dist_in(alpha):
    """Distance from each pixel inside the shape to its edge (the canvas border is also an edge)."""
    padded = np.pad(alpha >= 0.5, 1, constant_values=False)
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1] / SS


def over(dst, color, alpha):
    a = alpha[..., None]
    return dst * (1 - a) + np.asarray(color) * a


def place_n(n_alpha, placements):
    """Squeezes the N vertically and places one copy for each (top, height), in pixels of the 600 x 600 logo."""
    ys, _ = np.where(n_alpha > 0.01)
    crop = n_alpha[ys.min():ys.max() + 1]
    out = np.zeros_like(n_alpha)
    for top, height in placements:
        im = Image.fromarray((crop * 255).astype(np.uint8)).resize((crop.shape[1], int(round(height * SS))), Image.LANCZOS)
        a = np.asarray(im) / 255.
        t = int(round(top * SS))
        out[t:t + a.shape[0]] = np.maximum(out[t:t + a.shape[0]], a)
    return out


def render(layers, placements, P=EFFECTS):
    gradient = up_rgb(layers['gradient'][..., :3])
    background = up_rgb(layers['background_gradient'][..., :3])
    background_alpha = up(layers['background_gradient'][..., 3]) * layers['background_opacity']
    frame_a = up(layers['frame'][..., 3])
    n_a = place_n(up(layers['n'][..., 3]), placements)

    img = over(np.ones((SIZE * SS, SIZE * SS, 3)), background, background_alpha)

    # N layer: outer bevel (outside the shape), outside stroke and gradient fill
    d = dist_out(n_a)
    height = np.where(n_a >= 0.5, 1.0, np.clip(1 - d / P['bevel_size'], 0, 1))
    height = ndimage.gaussian_filter(height, P['soften'] * SS)
    gy, gx = np.gradient(height * P['bevel_size'] * SS * P['depth'])
    ang, alt = np.radians(BEVEL_ANGLE), np.radians(BEVEL_ALTITUDE)
    lx, ly, lz = np.cos(alt) * np.cos(ang), -np.cos(alt) * np.sin(ang), np.sin(alt)
    normal = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    shading = normal[..., 0] * lx + normal[..., 1] * ly + normal[..., 2] * lz - lz
    outside = (n_a < 0.5) * (1 - n_a)
    highlight = np.clip(shading, 0, None) / (1 - lz) * P['hi_gain'] * 0.5 * outside
    shadow = np.clip(-shading, 0, None) / (1 + lz) * P['sh_gain'] * 0.75 * outside
    img = img * (1 - highlight[..., None]) + (1 - (1 - img) * (1 - BEVEL_HIGHLIGHT)) * highlight[..., None]
    img = img * (1 - shadow[..., None])
    img = over(img, (0, 0, 0), np.clip(P['n_stroke'] + 0.5 - d, 0, 1) * (n_a < 0.5))
    img = over(img, gradient, n_a)

    # Frame layer: gradient fill with inner glow, and outside stroke, except where the N tails join the frame
    glow = np.clip(1 - dist_in(frame_a) / P['glow_size'], 0, 1) * (frame_a >= 0.5)
    glow = ndimage.gaussian_filter(glow, P['glow_soft'] * SS) * frame_a * 0.75
    join = ndimage.binary_dilation(n_a >= 0.5, iterations=int(P['join'] * SS))
    frame_stroke = np.clip(P['f_stroke'] + 0.5 - dist_out(frame_a), 0, 1) * (frame_a < 0.5) * (1 - join)
    img = over(img, (0, 0, 0), frame_stroke)
    img = over(img, over(gradient, (0, 0, 0), glow), frame_a)
    tails = n_a * ndimage.binary_dilation(frame_a >= 0.5, iterations=int(P['join'] * SS))
    img = over(img, gradient, tails)

    return Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8))


def main():
    layers = load_layers()
    stroke = EFFECTS['n_stroke']
    gap = (FRAME_INNER_HEIGHT - 3 * (N_HEIGHT + 2 * stroke)) / 4
    placements = [(FRAME_INNER_TOP + gap + stroke + k * (N_HEIGHT + 2 * stroke + gap), N_HEIGHT) for k in range(3)]
    logo = render(layers, placements)
    for path, size in OUTPUTS.items():
        logo.resize((size, size), Image.LANCZOS).save(path)
        print('Saved', path)


if __name__ == '__main__':
    main()
