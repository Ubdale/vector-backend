"""Strict Inkvec SVG adapter. Keeps gradients; bakes shape transforms.

Unsupported rendering constructs raise ValueError rather than silently disappearing.
Arc conversion uses <=45 degree cubics, adaptively subdivided to <0.01px error.
"""
import math,re,copy,xml.etree.ElementTree as ET
from svg.path import parse_path,Move,Line,Close,CubicBezier,QuadraticBezier,Arc

NS='http://www.w3.org/2000/svg'; ET.register_namespace('',NS)
IDENTITY=(1.,0.,0.,1.,0.,0.)
NUMBER=r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'
def num(x):
    if not math.isfinite(x):raise ValueError('Non-finite geometry')
    return f'{x:.7f}'.rstrip('0').rstrip('.') if abs(x)>1e-9 else '0'
def values(text):
    return [float(x) for x in re.findall(NUMBER,text)]
def mul(m,n):
    a,b,c,d,e,f=m;g,h,i,j,k,l=n
    return (a*g+c*h,b*g+d*h,a*i+c*j,b*i+d*j,a*k+c*l+e,b*k+d*l+f)
def point(m,z):
    a,b,c,d,e,f=m;return complex(a*z.real+c*z.imag+e,b*z.real+d*z.imag+f)
def transform(text):
    matrix=IDENTITY;cursor=0
    for match in re.finditer(r'([A-Za-z]+)\s*\(([^)]*)\)',text):
        if text[cursor:match.start()].strip(' ,\t\n\r'):raise ValueError('Invalid transform')
        cursor=match.end();name=match[1];p=values(match[2])
        if name=='matrix' and len(p)==6:m=tuple(p)
        elif name=='translate' and len(p) in (1,2):m=(1,0,0,1,p[0],p[1] if len(p)>1 else 0)
        elif name=='scale' and len(p) in (1,2):m=(p[0],0,0,p[-1],0,0)
        elif name=='rotate' and len(p) in (1,3):
            angle=math.radians(p[0]);m=(math.cos(angle),math.sin(angle),-math.sin(angle),math.cos(angle),0,0)
            if len(p)==3:m=mul(mul((1,0,0,1,p[1],p[2]),m),(1,0,0,1,-p[1],-p[2]))
        elif name in ('skewX','skewY') and len(p)==1:
            t=math.tan(math.radians(p[0]));m=(1,0,t,1,0,0) if name=='skewX' else (1,t,0,1,0,0)
        else:raise ValueError('Unsupported transform: '+match[0])
        matrix=mul(matrix,m)
    if text[cursor:].strip(' ,\t\n\r'):raise ValueError('Invalid transform suffix')
    return matrix
def rgb(text):
    text=text.strip()
    if re.fullmatch(r'#[0-9a-fA-F]{3}',text):text='#'+''.join(c*2 for c in text[1:])
    if re.fullmatch(r'#[0-9a-fA-F]{6}',text):v=[int(text[i:i+2],16) for i in (1,3,5)]
    elif re.fullmatch(r'rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)',text):v=list(map(int,re.findall(r'\d+',text)))
    elif text in {'white','black'}:v=[255]*3 if text=='white' else [0]*3
    else:raise ValueError('Unsupported color: '+text)
    if any(c<0 or c>255 for c in v):raise ValueError('Color out of range')
    return 'rgb('+','.join(map(str,v))+')'
def length(text):
    if not re.fullmatch(NUMBER+r'(?:px)?',str(text)):raise ValueError('Unsupported length: '+str(text))
    return float(str(text).removesuffix('px'))
def offset(text):
    return float(text[:-1])/100 if text.endswith('%') else float(text)
def linear_rgb_stops(stops):
    """Bounded sRGB-stop approximation for exporters that blend in sRGB.

    At most 16 intervals per native segment; reject rather than truncate if
    the 64-stop export contract cannot represent a native gradient.
    """
    def linear(x):
        x=x/255;return x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4
    def encoded(x):return 255*(12.92*x if x<=.0031308 else 1.055*x**(1/2.4)-.055)
    if len(stops)>4:raise ValueError('linearRGB gradient exceeds bounded approximation budget')
    result=[]
    for index,(start,end) in enumerate(zip(stops,stops[1:])):
        a,ca=start;b,cb=end;la=list(map(linear,ca));lb=list(map(linear,cb))
        for step in range(16):
            t=step/16;color=[round(encoded(x+(y-x)*t)) for x,y in zip(la,lb)]
            result.append((a+(b-a)*t,color))
    result.append(stops[-1])
    if len(result)>64:raise ValueError('Too many sampled gradient stops')
    # Check 64 points in each segment against exact linear-light interpolation.
    # Integer channel quantization alone can contribute up to half a level.
    worst=0
    for a,(t0,c0) in enumerate(stops[:-1]):
        t1,c1=stops[a+1];la=list(map(linear,c0));lb=list(map(linear,c1))
        for sample in range(65):
            t=sample/64;p=t*16;i=min(15,int(p));fraction=p-i
            qa=result[a*16+i][1];qb=result[a*16+i+1][1]
            exact=[encoded(x+(y-x)*t) for x,y in zip(la,lb)]
            worst=max(worst,max(abs((x+(y-x)*fraction)-z) for x,y,z in zip(qa,qb,exact)))
    if worst>1:raise ValueError(f'linearRGB approximation channel error exceeds1/255 ({worst})')
    return result,worst
def tag(e):return e.tag.rsplit('}',1)[-1]
def style(e,inherited):
    result=dict(inherited)
    raw={**e.attrib}
    for declaration in e.get('style','').split(';'):
        if declaration.strip():
            key,value=declaration.split(':',1);raw[key.strip()]=value.strip()
    for key in ['fill','fill-rule','fill-opacity','color-interpolation','opacity','stroke']:
        if key in raw:result[key]=raw[key]
    for key in ['filter','clip-path','mask','mix-blend-mode','display','visibility']:
        if raw.get(key) not in (None,'none','visible','inline','normal'):raise ValueError('Unsupported visual property '+key)
    if result.get('stroke','none')!='none':raise ValueError('Strokes are not supported by this adapter')
    return result
def shape_path(e):
    t=tag(e)
    def v(name,default=0):return length(e.get(name,str(default)))
    if t=='path':return parse_path(e.get('d',''))
    if t=='rect':
        x,y,w,h=v('x'),v('y'),v('width'),v('height')
        if w<0 or h<0:raise ValueError('Negative rectangle size')
        rx=min(w/2,v('rx',v('ry')));ry=min(h/2,v('ry',rx))
        if rx and ry:
            d=f'M{x+rx},{y}H{x+w-rx}A{rx},{ry} 0 0 1 {x+w},{y+ry}V{y+h-ry}A{rx},{ry} 0 0 1 {x+w-rx},{y+h}H{x+rx}A{rx},{ry} 0 0 1 {x},{y+h-ry}V{y+ry}A{rx},{ry} 0 0 1 {x+rx},{y}Z'
        else:d=f'M{x},{y}H{x+w}V{y+h}H{x}Z'
        return parse_path(d)
    if t in ('ellipse','circle'):
        x,y=v('cx'),v('cy');rx=v('r') if t=='circle' else v('rx');ry=v('r') if t=='circle' else v('ry')
        if rx<0 or ry<0:raise ValueError('Negative ellipse radius')
        if not rx or not ry:return parse_path('')
        return parse_path(f'M{x+rx},{y}A{rx},{ry} 0 1 1 {x-rx},{y}A{rx},{ry} 0 1 1 {x+rx},{y}Z')
    raise ValueError('Unsupported shape: '+t)
def arcs(segment,matrix):
    if segment.start==segment.end:return []
    if not segment.radius.real or not segment.radius.imag:return [Line(segment.start,segment.end)]
    rx=segment.radius.real*segment.radius_scale;ry=segment.radius.imag*segment.radius_scale
    rotation=math.radians(segment.rotation);cr,sr=math.cos(rotation),math.sin(rotation)
    def tangent(theta):return complex(-rx*math.sin(theta)*cr-ry*math.cos(theta)*sr,-rx*math.sin(theta)*sr+ry*math.cos(theta)*cr)
    # Conservative recursive test uses same-parameter error as well as a <=45deg bound.
    def split(t0,t1,depth=0):
        delta=math.radians(segment.delta)*(t1-t0);theta0=math.radians(segment.theta+segment.delta*t0);theta1=theta0+delta
        p0=segment.start if t0==0 else segment.point(t0)
        p1=segment.end if t1==1 else segment.point(t1)
        k=4/3*math.tan(delta/4)
        cubic=CubicBezier(p0,p0+k*tangent(theta0),p1-k*tangent(theta1),p1)
        # Cubic parameterization differs from angular parameterization. This stricter
        # error bound also bounds geometric deviation under the baked affine matrix.
        error=max(abs(point(matrix,cubic.point(t))-point(matrix,segment.point(t0+(t1-t0)*t))) for t in [.125,.25,.375,.5,.625,.75,.875])
        if (abs(delta)>math.pi/4+1e-12 or error>.0025) and depth<20:
            middle=(t0+t1)/2;return split(t0,middle,depth+1)+split(middle,t1,depth+1)
        if error>.01:raise ValueError('Arc conversion tolerance could not be satisfied')
        return [cubic]
    return split(0,1)
def path_text(path,matrix):
    commands=[]
    def pair(z):
        z=point(matrix,z);return num(z.real)+' '+num(z.imag)
    for segment in path:
        pieces=arcs(segment,matrix) if isinstance(segment,Arc) else [segment]
        for seg in pieces:
            if isinstance(seg,Move):commands.append('M '+pair(seg.end))
            elif isinstance(seg,Close):commands.append('Z')
            elif isinstance(seg,Line):commands.append('L '+pair(seg.end))
            elif isinstance(seg,CubicBezier):commands.append('C '+pair(seg.control1)+' '+pair(seg.control2)+' '+pair(seg.end))
            elif isinstance(seg,QuadraticBezier):commands.append('Q '+pair(seg.control)+' '+pair(seg.end))
            else:raise ValueError('Unsupported path segment '+str(type(seg)))
    return ' '.join(commands)

def normalize_inkvec(svg,width=None,height=None):
    source=ET.fromstring(svg)
    if tag(source)!='svg':raise ValueError('Root must be SVG')
    vb=values(source.get('viewBox',''))
    w=length(source.get('width',str(width))) if source.get('width') else width
    h=length(source.get('height',str(height))) if source.get('height') else height
    if w is None or h is None or w<=0 or h<=0:raise ValueError('Positive dimensions required')
    if not vb:vb=[0,0,w,h]
    if len(vb)!=4 or vb[2]<=0 or vb[3]<=0:raise ValueError('Invalid viewBox')
    if abs(w/vb[2]-h/vb[3])>1e-8:raise ValueError('Nonuniform viewport aspect ratio not supported')
    scale=w/vb[2];root_matrix=(scale,0,0,scale,-vb[0]*scale,-vb[1]*scale)
    output=ET.Element('{'+NS+'}svg',{'width':num(w),'height':num(h),'viewBox':f'0 0 {num(w)} {num(h)}','data-engine':'inkvec-normalized'})
    definitions=ET.SubElement(output,'{'+NS+'}defs');gradient_sources={};cache={}
    for e in source.iter():
        if tag(e) in ('linearGradient','radialGradient'):
            identifier=e.get('id')
            if not identifier or identifier in gradient_sources:raise ValueError('Duplicate/missing gradient id')
            gradient_sources[identifier]=e
    def gradient(identifier,matrix):
        key=(identifier,tuple(matrix))
        if key in cache:return cache[key]
        if identifier not in gradient_sources:raise ValueError('Unknown paint server '+identifier)
        original=gradient_sources[identifier]
        if original.get('gradientUnits','objectBoundingBox')!='userSpaceOnUse':raise ValueError('Only user-space gradients supported')
        if original.get('spreadMethod','pad')!='pad':raise ValueError('Only padded gradients supported')
        if any(k.rsplit('}',1)[-1]=='href' for k in original.attrib):raise ValueError('Gradient inheritance not supported')
        g=copy.deepcopy(original);new_id='inkg'+str(len(cache));g.set('id',new_id)
        for k in list(g.attrib):
            if k not in {'id','gradientUnits','x1','y1','x2','y2','cx','cy','r','fx','fy','fr','gradientTransform','color-interpolation','spreadMethod'}:raise ValueError('Unsupported gradient attribute '+k)
        gmatrix=mul(matrix,transform(original.get('gradientTransform','')))
        g.set('gradientTransform','matrix('+' '.join(num(v) for v in gmatrix)+')')
        for coordinate in ['x1','y1','x2','y2','cx','cy','r','fx','fy','fr']:
            if coordinate in g.attrib:g.set(coordinate,num(length(g.get(coordinate))))
        interpolation=g.get('color-interpolation','sRGB')
        if interpolation not in ('sRGB','linearRGB'):raise ValueError('Unsupported color interpolation')
        colors=[];stops=[]
        for stop in g:
            if tag(stop)!='stop':raise ValueError('Unsupported gradient child')
            st=style(stop,{});color=st.get('stop-color',stop.get('stop-color'))
            if color is None:raise ValueError('Missing stop color')
            stop.set('stop-color',rgb(color));colors.append(list(map(int,re.findall(r'\d+',rgb(color)))))
            if any(k not in {'offset','stop-color','stop-opacity'} for k in stop.attrib):raise ValueError('Unsupported stop attribute')
            if float(stop.get('stop-opacity','1'))!=1:raise ValueError('Non-opaque gradient stops unsupported by export contract')
            position=offset(stop.get('offset','0'))
            if not 0<=position<=1 or (stops and position<=stops[-1][0]):raise ValueError('Gradient stops must increase strictly within0..1')
            stop.set('offset',num(position));stops.append((position,colors[-1]))
        if not colors:raise ValueError('Gradient has no stops')
        if len(colors)<2:raise ValueError('Gradient needs at least2stops')
        if stops[0][0]>0:stops.insert(0,(0,stops[0][1]))
        if stops[-1][0]<1:stops.append((1,stops[-1][1]))
        if interpolation=='linearRGB':
            sampled,worst=linear_rgb_stops(stops)
            for stop in list(g):g.remove(stop)
            for position,color in sampled:ET.SubElement(g,'{'+NS+'}stop',{'offset':num(position),'stop-color':'rgb('+','.join(map(str,color))+')'})
            g.set('color-interpolation','sRGB');g.set('data-source-interpolation','linearRGB');g.set('data-approx-channel-bound',num(worst))
        else:
            for stop in list(g):g.remove(stop)
            for position,color in stops:ET.SubElement(g,'{'+NS+'}stop',{'offset':num(position),'stop-color':'rgb('+','.join(map(str,color))+')'})
        if len(g)>64:raise ValueError('Gradient exceeds64stops')
        if tag(g)=='radialGradient':
            cx,cy=length(g.get('cx','0')),length(g.get('cy','0'))
            if length(g.get('fx',str(cx)))!=cx or length(g.get('fy',str(cy)))!=cy or length(g.get('fr','0'))!=0:raise ValueError('Only centered radial gradients supported')
            if length(g.get('r','0'))<=0:raise ValueError('Radial gradient radius must be positive')
        fallback='rgb('+','.join(str(round(sum(c[i] for c in colors)/len(colors))) for i in range(3))+')'
        definitions.append(g);cache[key]=(new_id,fallback);return cache[key]
    def visit(e,matrix,inherited):
        t=tag(e)
        if t in ('defs','metadata','title','desc'):return
        if t not in ('svg','g','path','rect','ellipse','circle'):raise ValueError('Unsupported SVG element '+t)
        visual=style(e,inherited);local=mul(matrix,transform(e.get('transform','')))
        if t in ('svg','g'):
            if t=='svg' and e is not source:raise ValueError('Nested SVG viewport not supported')
            if float(visual.get('opacity','1'))!=1:raise ValueError('Group opacity must not be flattened')
            for child in e:visit(child,local,visual)
            return
        fill=visual.get('fill','#000000')
        if fill=='none':return
        d=path_text(shape_path(e),local)
        if not d:return
        attrs={'d':d,'opacity':num(float(visual.get('opacity','1'))*float(visual.get('fill-opacity','1'))),'fill-rule':visual.get('fill-rule','nonzero')}
        if attrs['fill-rule'] not in ('nonzero','evenodd'):raise ValueError('Unknown fill rule')
        match=re.fullmatch(r'url\(#([^\s)]+)\)',fill)
        if match:
            identifier,fallback=gradient(match[1],local);attrs['fill']='url(#'+identifier+')';attrs['data-fallback-fill']=fallback
        else:attrs['fill']=rgb(fill)
        if e.get('id'):attrs['id']=e.get('id')
        ET.SubElement(output,'{'+NS+'}path',attrs)
    visit(source,root_matrix,{'fill':'#000000','opacity':'1','fill-opacity':'1','fill-rule':'nonzero','stroke':'none'})
    if not len(definitions):output.remove(definitions)
    return ET.tostring(output,encoding='unicode')

normalize_svg=normalize_inkvec
