import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tmp/python-deps'))
import unittest
import numpy as np
from PIL import Image,ImageDraw
from tracer import trace,normalize_svg

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
    def test_complex_opaque_uses_compact_native_cubic_curves(self):
        # Many unique colors exercise native routing rather than low-color
        # fallback. A smooth gradient should form broad, not pixel-size, bands.
        y,x=np.mgrid[:96,:128]
        pixels=np.stack((x*2,y*2,(x+y)%256,np.full_like(x,255)),axis=2).astype(np.uint8)
        svg=trace(pixels.tobytes(),128,96,{'preset':'illustration','edgeStyle':'smooth'})
        self.assertIn(' C ',svg)
        self.assertNotIn('transform=',svg)
        self.assertNotIn('fill="#',svg)
        self.assertLess(svg.count('<path'),200)
        self.assertIn('viewBox="0 0 128 96"',svg)
    def test_native_translation_and_repeated_coordinates_normalized(self):
        svg='<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 C1 2 3 4 5 6 7 8 9 10 11 12 Z" fill="#0A141E" transform="translate(10,20)"/></svg>'
        output=normalize_svg(svg,30,40)
        self.assertIn('M 10 20 C 11 22 13 24 15 26 C 17 28 19 30 21 32 Z',output)
        self.assertIn('fill="rgb(10,20,30)"',output)
    def test_native_rejects_unexpected_transforms(self):
        with self.assertRaises(ValueError):
            normalize_svg('<svg xmlns="http://www.w3.org/2000/svg"><path transform="scale(2)"/></svg>',10,10)
    def test_antialiased_diagram_bypasses_region_cleanup(self):
        pixels=np.full((96,128,4),255,dtype=np.uint8)
        y,x=np.mgrid[:20,:20]
        pixels[40:60,40:60,:3]=np.stack((x*11,y*11,(x+y)*5),axis=2)
        pixels[10,5:120,:3]=0
        svg=trace(pixels.tobytes(),128,96,{'preset':'illustration','edgeStyle':'smooth','autoColors':True})
        self.assertNotIn(' C ',svg)
        self.assertNotIn(' Q ',svg)
        self.assertIn('fill="rgb(0,0,0)"',svg)

if __name__=='__main__': unittest.main()
