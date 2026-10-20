import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
from geometric_logo import classify_geometric
from geometric_refinement import geometry_matches,refine_geometric

class GeometricRefinementTests(unittest.TestCase):
    def fixture(self):
        image=Image.new('RGBA',(128,128),'white')
        ImageDraw.Draw(image).rectangle((32,32,95,95),fill='black')
        svg='<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128"><path fill="white" d="M 0 0 L 128 0 L 128 128 L 0 128 Z"/><path fill="black" d="M 32 32 L 96 32 L 96 96 L 32 96 Z"/></svg>'
        return image,svg,classify_geometric(image)

    def test_unchanged_topology_is_accepted(self):
        image,svg,profile=self.fixture();rgb=np.asarray(image.convert('RGB')).astype(float)
        with patch('geometric_refinement.visible_color_samples',return_value=(rgb,rgb)):
            self.assertTrue(geometry_matches(svg,image,profile))

    def test_small_disconnected_mark_cannot_be_hidden_by_large_shape(self):
        image,svg,_=self.fixture();ImageDraw.Draw(image).rectangle((5,5,8,8),fill='black')
        profile=classify_geometric(image);actual=np.asarray(image.convert('RGB')).astype(float);prediction=actual.copy();prediction[5:9,5:9]=255
        # Losing sixteen pixels barely changes aggregate error; component checks must reject it.
        with patch('geometric_refinement.visible_color_samples',return_value=(prediction,actual)):
            self.assertFalse(geometry_matches(svg,image,profile))

    def test_small_hole_cannot_be_filled(self):
        image,svg,_=self.fixture();ImageDraw.Draw(image).rectangle((50,50,53,53),fill='white')
        profile=classify_geometric(image);actual=np.asarray(image.convert('RGB')).astype(float);prediction=actual.copy();prediction[50:54,50:54]=0
        with patch('geometric_refinement.visible_color_samples',return_value=(prediction,actual)):
            self.assertFalse(geometry_matches(svg,image,profile))

    def test_custom_settings_and_transparency_do_not_call_native(self):
        image,svg,_=self.fixture();baseline=svg.replace('</svg>','<path/>'*24+'</svg>')
        for change in [{'edgeStyle':'faithful'},{'corners':True},{'autoColors':False},{'smooth':2},{'detail':1},{'refine':False}]:
            settings={'edgeStyle':'smooth',**change}
            with patch('inkvec_runner.trace_inkvec') as runner:
                self.assertEqual(refine_geometric(baseline,image.tobytes(),128,128,settings),baseline);runner.assert_not_called()
        image.putpixel((0,0),(255,255,255,0))
        with patch('inkvec_runner.trace_inkvec') as runner:
            self.assertEqual(refine_geometric(baseline,image.tobytes(),128,128,{'edgeStyle':'smooth'}),baseline);runner.assert_not_called()

    def test_missing_optional_engine_keeps_baseline(self):
        image,svg,_=self.fixture();baseline=svg.replace('</svg>','<path/>'*24+'</svg>')
        with patch('inkvec_runner.trace_inkvec',return_value=None):
            self.assertEqual(refine_geometric(baseline,image.tobytes(),128,128,{'edgeStyle':'smooth'}),baseline)
