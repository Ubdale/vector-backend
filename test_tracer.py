import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tmp/python-deps'))
import unittest
import numpy as np
from PIL import Image,ImageDraw
from tracer import trace,normalize_svg
from gradient_fit import rectangle_gradient,fit_visible_colors

class TracerTests(unittest.TestCase):
    def test_multicolor_texture_does_not_collapse_to_grey(self):
        # Several changing color directions cannot be represented by a single
        # solid fill or two-stop ramp. Regression for native region collapse.
        y,x=np.mgrid[:96,:128]
        rng=np.random.default_rng(41)
        channels=[125+75*np.sin(x/17+y/23),125+75*np.sin(x/29-y/13),125+75*np.cos(x/19+y/31)]
        pixels=np.dstack(channels).astype(float)+rng.normal(0,3,(96,128,3))
        pixels=np.dstack((np.clip(pixels,0,255).astype(np.uint8),np.full((96,128),255,dtype=np.uint8)))
        source=Image.fromarray(pixels)
        svg=trace(pixels.tobytes(),128,96,{'preset':'illustration','edgeStyle':'smooth','autoColors':True})
        from gradient_fit import visible_color_error
        self.assertLess(visible_color_error(svg,source),12)
        self.assertNotIn('<image',svg)

    def test_color_error_accounts_for_occlusion_and_linear_gradient(self):
        from gradient_fit import visible_color_error
        pixels=np.empty((32,256,4),dtype=np.uint8)
        pixels[:,:,0]=np.arange(256,dtype=np.uint8);pixels[:,:,1]=80;pixels[:,:,2]=255-np.arange(256,dtype=np.uint8);pixels[:,:,3]=255
        source=Image.fromarray(pixels)
        self.assertLess(visible_color_error(rectangle_gradient(source),source),1)
        source=Image.new('RGB',(64,64),'blue')
        svg='<svg xmlns="http://www.w3.org/2000/svg"><path fill="rgb(255,0,0)" d="M 0 0 L 64 0 L 64 64 L 0 64 Z"/><path fill="rgb(0,0,255)" d="M 0 0 L 64 0 L 64 64 L 0 64 Z"/></svg>'
        self.assertEqual(visible_color_error(svg,source),0)

    def test_true_circular_lighting_traces_as_one_radial_path(self):
        y,x=np.mgrid[:160,:192]
        distance=np.sqrt((x+.5-88)**2+(y+.5-77)**2)/135
        rgb=30+(np.array([180,100,30])-30)*distance[:,:,None]
        source=Image.fromarray(np.rint(rgb).astype(np.uint8)).convert('RGBA')
        svg=trace(source.tobytes(),192,160,{'preset':'illustration','edgeStyle':'smooth','autoColors':True})
        self.assertIn('data-engine="analytic-radial-gradient"',svg)
        self.assertEqual(svg.count('<path'),1)
        self.assertEqual(svg.count('<radialGradient'),1)
        self.assertNotIn('<image',svg)

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
    def test_full_frame_true_linear_gradient_is_one_vector_path(self):
        x=np.arange(256,dtype=np.uint8)
        pixels=np.empty((48,256,4),dtype=np.uint8)
        pixels[:,:,0]=x; pixels[:,:,1]=80; pixels[:,:,2]=255-x; pixels[:,:,3]=255
        svg=trace(pixels.tobytes(),256,48,{'edgeStyle':'smooth','preset':'illustration'})
        self.assertEqual(svg.count('<linearGradient'),1)
        self.assertEqual(svg.count('<path'),1)
        self.assertIn('data-engine="analytic-gradient"',svg)
        self.assertNotIn('<image',svg)
    def test_single_pixel_accent_rejects_global_gradient(self):
        x=np.arange(256,dtype=np.uint8)
        pixels=np.empty((48,256,4),dtype=np.uint8)
        pixels[:,:,0]=x; pixels[:,:,1]=80; pixels[:,:,2]=255-x; pixels[:,:,3]=255
        pixels[17,37,:3]=0
        self.assertIsNone(rectangle_gradient(Image.fromarray(pixels)))
    def test_alpha_ramp_never_becomes_opaque_global_gradient(self):
        pixels=np.full((16,256,4),100,dtype=np.uint8)
        pixels[:,:,3]=np.arange(256,dtype=np.uint8)
        svg=trace(pixels.tobytes(),256,16,{'edgeStyle':'smooth','preset':'illustration'})
        self.assertNotIn('linearGradient',svg)
        self.assertIn('data-engine="opencv"',svg)
    def test_gradient_fitting_ignores_hidden_foreground_colors(self):
        source=Image.new('RGBA',(64,64),'red')
        ImageDraw.Draw(source).rectangle((16,16,47,47),fill='blue')
        svg='<svg xmlns="http://www.w3.org/2000/svg"><path fill="rgb(240,0,0)" d="M 0 0 L 64 0 L 64 64 L 0 64 Z"/><path fill="rgb(0,0,240)" d="M 16 16 L 48 16 L 48 48 L 16 48 Z"/></svg>'
        output=fit_visible_colors(svg,source)
        self.assertIn('fill="rgb(255,0,0)"',output)
        self.assertIn('fill="rgb(0,0,255)"',output)
        self.assertNotIn('linearGradient',output)
    def test_native_preserves_isolated_one_pixel_highlight(self):
        y,x=np.mgrid[:96,:128]
        pixels=np.stack((x*2,y*2,(x+y)%256,np.full_like(x,255)),axis=2).astype(np.uint8)
        pixels[40:60,40:60,:3]=[30,80,120]
        pixels[50,50,:3]=[255,255,255]
        svg=trace(pixels.tobytes(),128,96,{'preset':'illustration','edgeStyle':'smooth'})
        self.assertIn('data-engine="vtracer-rust"',svg)
        self.assertIn('data-accent="true" fill="rgb(255,255,255)"',svg)
        self.assertIn('M 50 50 L 51 50 L 51 51 L 50 51 Z',svg)
    def test_narrow_noise_uses_precision_fallback(self):
        random=np.random.default_rng(73)
        pixels=random.integers(0,256,(1,512,4),dtype=np.uint8); pixels[:,:,3]=255
        svg=trace(pixels.tobytes(),512,1,{'preset':'illustration','edgeStyle':'smooth'})
        self.assertIn('data-engine="opencv"',svg)

if __name__=='__main__': unittest.main()
