"""Source-supported shading checks, separate from tracing geometry tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tmp/python-deps'))
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from gradient_fit import fit_visible_colors,visible_color_error,NS

RECT='M 0 0 L 128 0 L 128 96 L 0 96 Z'
def svg(extra=''):
    return f'<svg xmlns="{NS}" width="128" height="96"><path fill="rgb(100,110,120)" d="{RECT}"/>{extra}</svg>'
def noisy_ramp():
    y,x=np.mgrid[:96,:128]
    t=x/127
    colors=np.stack((50+110*t,70+70*t,90+30*t),axis=2)
    noise=np.random.default_rng(741).integers(-14,15,colors.shape)
    return Image.fromarray(np.clip(colors+noise,0,255).astype(np.uint8))

class GradientQualityTests(unittest.TestCase):
    def test_coherent_noisy_ramp_improves_without_geometry_changes(self):
        source=noisy_ramp(); original=svg()
        fitted=fit_visible_colors(original,source)
        self.assertEqual(fitted.count('<linearGradient'),1)
        self.assertLess(visible_color_error(fitted,source),10)
        self.assertLess(visible_color_error(fitted,source),visible_color_error(original,source)*.75)
        self.assertEqual(ET.fromstring(fitted).find('{'+NS+'}path').get('d'),RECT)

    def test_two_flat_colors_stay_a_hard_boundary(self):
        pixels=np.full((96,128,3),(80,100,120),dtype=np.uint8)
        pixels[:,64:]=(120,140,160)
        fitted=fit_visible_colors(svg(),Image.fromarray(pixels))
        self.assertNotIn('<linearGradient',fitted)

    def test_unrelated_texture_does_not_receive_invented_lighting(self):
        pixels=np.random.default_rng(583).integers(30,220,(96,128,3),dtype=np.uint8)
        self.assertNotIn('<linearGradient',fit_visible_colors(svg(),Image.fromarray(pixels)))

    def test_occluding_foreground_is_excluded_from_background_fit(self):
        source=noisy_ramp(); pixels=np.asarray(source).copy()
        # Pixel-aligned right-side occluder leaves a sizeable gradient interior.
        pixels[:,80:]=(240,20,30)
        foreground='<path fill="rgb(240,20,30)" d="M 80 0 L 128 0 L 128 96 L 80 96 Z"/>'
        fitted=fit_visible_colors(svg(foreground),Image.fromarray(pixels))
        paths=ET.fromstring(fitted).findall('{'+NS+'}path')
        self.assertTrue(paths[0].get('fill').startswith('url(#'))
        self.assertEqual(paths[1].get('fill'),'rgb(240,20,30)')
        self.assertLess(visible_color_error(fitted,Image.fromarray(pixels)),7)

    def test_gradient_opt_out_keeps_solid_fills(self):
        self.assertNotIn('<linearGradient',fit_visible_colors(svg(),noisy_ramp(),False))

if __name__=='__main__': unittest.main()
