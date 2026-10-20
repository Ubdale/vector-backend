import unittest
import numpy as np
from PIL import Image
from refinement import prefer_candidate

class RefinementTests(unittest.TestCase):
    def fixture(self):
        x=np.arange(128,dtype=float)/127
        rgb=np.tile(np.column_stack((40+160*x,90+40*x,180-120*x))[None],(64,1,1))
        rgb[25:29,60:64]=0
        source=Image.fromarray(np.rint(rgb).astype(np.uint8))
        start='<svg xmlns="http://www.w3.org/2000/svg" width="128" height="64"><defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0.5" y1="0" x2="127.5" y2="0"><stop offset="0" stop-color="rgb(40,90,180)"/><stop offset="1" stop-color="rgb(200,130,60)"/></linearGradient></defs><path fill="url(#g)" d="M 0 0 L 128 0 L 128 64 L 0 64 Z"/>'
        eye='<path fill="rgb(0,0,0)" d="M 60 25 L 64 25 L 64 29 L 60 29 Z"/>'
        baseline='<svg xmlns="http://www.w3.org/2000/svg"><path fill="rgb(120,110,120)" d="M 0 0 L 128 0 L 128 64 L 0 64 Z"/></svg>'
        return source,baseline,start,eye

    def test_lower_pixel_error_cannot_erase_small_dark_features(self):
        source,baseline,start,eye=self.fixture()
        self.assertFalse(prefer_candidate(baseline,start+'</svg>',source))
        self.assertTrue(prefer_candidate(baseline,start+eye+'</svg>',source))

    def test_same_or_worse_result_is_not_a_quality_improvement(self):
        source,baseline,start,eye=self.fixture()
        exact=start+eye+'</svg>'
        self.assertFalse(prefer_candidate(exact,baseline,source))
        self.assertFalse(prefer_candidate(exact,exact,source))
