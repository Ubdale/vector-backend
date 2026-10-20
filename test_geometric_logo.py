import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tmp/python-deps'))
import unittest
import numpy as np
from PIL import Image,ImageDraw
from geometric_logo import classify_geometric

class GeometricLogoTests(unittest.TestCase):
    def shape(self,kind='circle'):
        image=Image.new('RGB',(256*4,256*4),(242,239,227));draw=ImageDraw.Draw(image)
        if kind=='circle':draw.ellipse((48*4,48*4,208*4,208*4),fill=(102,136,81))
        else:draw.polygon([(40*4,210*4),(128*4,40*4),(216*4,210*4)],fill=(102,136,81))
        return image.resize((256,256),Image.Resampling.BOX)

    def test_circle_and_triangle_return_bounded_palette_and_topology(self):
        for kind in ['circle','triangle']:
            result=classify_geometric(self.shape(kind))
            self.assertIsNotNone(result)
            self.assertEqual(result['palette'].shape,(2,3))
            self.assertEqual(result['labels'].shape,(256,256))
            self.assertEqual(len(result['components']),2)
            self.assertGreaterEqual(result['coverage'],.96)
            self.assertTrue(all(item['area']>=9 for item in result['components']))

    def test_unexplained_one_pixel_and_two_pixel_red_dots_are_protected(self):
        for size in [1,2,3]:
            image=self.shape();ImageDraw.Draw(image).rectangle((20,20,19+size,19+size),fill=(255,0,0))
            self.assertIsNone(classify_geometric(image))

    def test_rare_flat_two_by_two_mixed_ink_is_protected(self):
        image=self.shape();ImageDraw.Draw(image).rectangle((20,20,21,21),fill=(172,188,154))
        self.assertIsNone(classify_geometric(image))

    def test_single_pixel_existing_ink_is_protected(self):
        image=self.shape();image.putpixel((20,20),(102,136,81))
        self.assertIsNone(classify_geometric(image))

    def test_three_ink_local_antialias_junction_is_allowed(self):
        pixels=np.full((128,128,3),(242,239,227),dtype=np.uint8)
        pixels[:,64:]=(102,136,81);pixels[64:]=(207,170,98)
        pixels[63,63]=np.rint(np.array([[242,239,227],[102,136,81],[207,170,98]]).mean(axis=0)).astype(np.uint8)
        self.assertIsNotNone(classify_geometric(Image.fromarray(pixels)))
        # The same novel color in a flat region is unexplained ink.
        pixels[20,20]=pixels[63,63]
        self.assertIsNone(classify_geometric(Image.fromarray(pixels)))

    def test_dense_diagram_and_mandala_remain_on_precision_route(self):
        diagram=Image.new('RGB',(320,320),'white');draw=ImageDraw.Draw(diagram)
        for y in range(10,300,22):
            for x in range(10,300,48):draw.text((x,y),'A1',fill='black')
        self.assertIsNone(classify_geometric(diagram))
        mandala=Image.new('RGB',(320,320),'white');draw=ImageDraw.Draw(mandala)
        for radius in range(10,151,9):draw.ellipse((160-radius,160-radius,160+radius,160+radius),outline='black',width=2)
        self.assertIsNone(classify_geometric(mandala))

    def test_transparency_large_images_and_too_small_images_rejected(self):
        image=self.shape().convert('RGBA');image.putpixel((5,5),(242,239,227,254))
        self.assertIsNone(classify_geometric(image))
        self.assertIsNone(classify_geometric(Image.new('RGB',(769,128))))
        self.assertIsNone(classify_geometric(Image.new('RGB',(31,128))))

if __name__=='__main__':unittest.main()
