import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tmp/python-deps'))
import unittest
import numpy as np
from PIL import Image,ImageDraw
from tracer import trace

class TracerTests(unittest.TestCase):
    def test_circle_is_fitted_curve(self):
        image=Image.new('RGBA',(128,128)); ImageDraw.Draw(image).ellipse((16,16,111,111),fill=(30,120,60,255))
        svg=trace(image.tobytes(),128,128,{'edgeStyle':'smooth'})
        self.assertEqual(svg.count('<path'),1)
        self.assertEqual(svg.count(' Q '),24)
    def test_one_pixel_line_and_tiny_accent_survive(self):
        image=Image.new('RGBA',(64,64)); draw=ImageDraw.Draw(image)
        draw.line((5,10,55,10),fill=(10,80,20,255),width=1); draw.point((40,40),fill=(230,10,10,255))
        svg=trace(image.tobytes(),64,64,{'edgeStyle':'smooth'})
        self.assertEqual(svg.count('<path'),2)
        self.assertNotIn(' Q ',svg)
    def test_hole_and_transparency(self):
        image=Image.new('RGBA',(64,64)); draw=ImageDraw.Draw(image)
        draw.rectangle((4,4,59,59),fill=(20,60,90,128)); draw.rectangle((16,16,47,47),fill=(255,255,255,0))
        svg=trace(image.tobytes(),64,64,{'edgeStyle':'smooth'})
        self.assertEqual(svg.count('<path'),1); self.assertEqual(svg.count('M '),2)
        self.assertIn('opacity="0.502"',svg)
    def test_faithful_and_background_removal(self):
        image=Image.new('RGBA',(64,64),'white'); ImageDraw.Draw(image).rectangle((10,10,50,50),fill='black')
        svg=trace(image.tobytes(),64,64,{'edgeStyle':'faithful','background':True})
        self.assertEqual(svg.count('<path'),1); self.assertNotIn(' Q ',svg)
    def test_fully_transparent(self):
        image=np.full((10,10,4),255,dtype=np.uint8); image[:,:,3]=0
        self.assertNotIn('<path',trace(image.tobytes(),10,10,{}))

if __name__=='__main__': unittest.main()
