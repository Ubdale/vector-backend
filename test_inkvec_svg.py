"""Contract tests for the opaque Inkvec adapter; no external renderer required."""
import math,re,unittest,xml.etree.ElementTree as ET
from inkvec_svg import normalize_inkvec,transform,point,arcs
from svg.path import parse_path,Arc,CubicBezier

NS={'s':'http://www.w3.org/2000/svg'}
def document(body,viewbox='-.5 -.5 100 100'):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="'+viewbox+'">'+body+'</svg>'

class InkvecAdapterTests(unittest.TestCase):
    def parse(self,body):return ET.fromstring(normalize_inkvec(document(body)))
    def test_inherited_fill_viewbox_translation_and_nested_transform(self):
        root=self.parse('<g fill="#3a5" transform="translate(3 4)"><g transform="scale(2)"><path d="m1 2h3v4z"/></g></g>')
        p=root.find('s:path',NS)
        self.assertEqual(p.get('fill'),'rgb(51,170,85)')
        self.assertEqual(p.get('d'),'M 5.5 8.5 L 11.5 8.5 L 11.5 16.5 Z')
        self.assertEqual(root.get('viewBox'),'0 0 100 100')
        self.assertNotIn('transform',p.attrib)
    def test_primitives_convert_to_absolute_supported_commands(self):
        root=self.parse('<g fill="#4050b0"><rect x="3" y="4" width="20" height="15" rx="3"/><circle cx="45" cy="50" r="10"/><ellipse cx="70" cy="40" rx="13" ry="4" transform="rotate(32 70 40)"/></g>')
        paths=root.findall('s:path',NS);self.assertEqual(len(paths),3)
        for p in paths:
            self.assertNotRegex(p.get('d'),r'[AHVSTahvst]')
            self.assertTrue(p.get('d').endswith('Z'))
            self.assertRegex(p.get('d'),r'\bC ')
    def test_compound_hole_winding_and_evenodd_preserved(self):
        d='M1 1L20 1L20 20L1 20Z M5 5L5 15L15 15L15 5Z'
        p=self.parse('<path fill-rule="evenodd" fill="#123456" d="'+d+'"/>').find('s:path',NS)
        self.assertEqual(p.get('d').count('M '),2);self.assertEqual(p.get('d').count('Z'),2)
        self.assertEqual(p.get('fill-rule'),'evenodd')
        segments=parse_path(p.get('d'));self.assertEqual(segments[0].end,complex(1.5,1.5))
    def test_elliptical_arc_conversion_under_affine_transform(self):
        arc=next(x for x in parse_path('M10 20 A30 12 37 1 1 70 60') if isinstance(x,Arc))
        matrix=transform('translate(4 7) rotate(17) scale(1.3 .6)');curves=arcs(arc,matrix)
        self.assertGreater(len(curves),1);self.assertTrue(all(isinstance(c,CubicBezier) for c in curves))
        self.assertLess(abs(curves[0].start-arc.start),1e-8);self.assertLess(abs(curves[-1].end-arc.end),1e-8)
        for a,b in zip(curves,curves[1:]):self.assertLess(abs(a.end-b.start),1e-8)
        # Every generated control/end point must remain finite after baking.
        for curve in curves:
            for p in [curve.start,curve.control1,curve.control2,curve.end]:
                p=point(matrix,p);self.assertTrue(math.isfinite(p.real) and math.isfinite(p.imag))
    def test_radial_gradient_matrix_and_multistop_contract(self):
        body='<defs><radialGradient id="g" gradientUnits="userSpaceOnUse" cx="20" cy="30" r="15" gradientTransform="translate(20 30) rotate(35) scale(1 .4) translate(-20 -30)"><stop offset="0" stop-color="#123456"/><stop offset=".3" stop-color="#56789a"/><stop offset="1" stop-color="#abcdef"/></radialGradient></defs><g transform="translate(4 7)"><circle cx="20" cy="30" r="15" fill="url(#g)"/></g>'
        root=self.parse(body);g=root.find('s:defs/s:radialGradient',NS);p=root.find('s:path',NS)
        self.assertEqual(len(g),3);self.assertEqual(g.get('cx'),'20');self.assertEqual(g.get('cy'),'30')
        self.assertRegex(g.get('gradientTransform'),r'^matrix\(')
        matrix=transform(g.get('gradientTransform'));center=point(matrix,complex(20,30))
        self.assertAlmostEqual(center.real,24.5,places=5);self.assertAlmostEqual(center.imag,37.5,places=5)
        self.assertRegex(p.get('data-fallback-fill'),r'^rgb\(')
        self.assertEqual(p.get('fill'),'url(#'+g.get('id')+')')
    def test_linear_light_conversion_is_explicit_bounded_and_srgb(self):
        body='<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="90" y2="90" color-interpolation="linearRGB"><stop offset="0" stop-color="#2050ad"/><stop offset=".4" stop-color="#608cdd"/><stop offset="1" stop-color="#87c4f0"/></linearGradient></defs><rect width="100" height="100" fill="url(#g)"/>'
        g=self.parse(body).find('s:defs/s:linearGradient',NS)
        self.assertEqual(g.get('color-interpolation'),'sRGB');self.assertEqual(g.get('data-source-interpolation'),'linearRGB')
        self.assertGreater(len(g),3);self.assertLessEqual(len(g),64)
        offsets=[float(s.get('offset')) for s in g];self.assertEqual(offsets[0],0);self.assertEqual(offsets[-1],1)
        self.assertTrue(all(a<b for a,b in zip(offsets,offsets[1:])))
        self.assertLessEqual(float(g.get('data-approx-channel-bound')),1)
        for stop in g:self.assertRegex(stop.get('stop-color'),r'^rgb\(\d+,\d+,\d+\)$')
    def test_unsupported_rendering_fails_loudly(self):
        cases={
            'stroke':'<path d="M0 0L10 10" stroke="#000"/>',
            'external-paint':'<circle r="5" fill="url(https://example.com/paint.svg#g)"/>',
            'external-image':'<image href="https://example.com/image.png"/>',
            'filter':'<circle r="5" filter="url(#blur)"/>',
            'mask':'<circle r="5" mask="url(#mask)"/>',
            'text':'<text>Unsupported text</text>',
            'group-opacity':'<g opacity=".5"><circle r="5"/></g>',
            'styled-group-opacity':'<g style="opacity:.5"><circle r="5"/></g>',
            'object-bbox-gradient':'<defs><linearGradient id="g"><stop offset="0" stop-color="#000"/><stop offset="1" stop-color="#fff"/></linearGradient></defs><circle r="5" fill="url(#g)"/>',
            'off-center-radial':'<defs><radialGradient id="g" gradientUnits="userSpaceOnUse" cx="4" cy="5" r="6" fx="2"><stop offset="0" stop-color="#000"/><stop offset="1" stop-color="#fff"/></radialGradient></defs><circle r="5" fill="url(#g)"/>',
            'duplicate-stops':'<defs><linearGradient id="g" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#000"/><stop offset="0" stop-color="#fff"/></linearGradient></defs><circle r="5" fill="url(#g)"/>',
            'percent-gradient-units':'<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="10%"><stop offset="0" stop-color="#000"/><stop offset="1" stop-color="#fff"/></linearGradient></defs><circle r="5" fill="url(#g)"/>',
        }
        for name,body in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):normalize_inkvec(document(body))

if __name__=='__main__':unittest.main()
