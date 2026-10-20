"""Install a checksum-pinned plain CPU Inkvec CLI during the Render build."""
import hashlib,io,os,platform,tarfile,urllib.request,zipfile
from pathlib import Path

VERSION='0.2.6'
ASSETS={
 'Linux':('x86_64-unknown-linux-gnu.tar.gz','541c0250d5735a164efd2619672dad51f09399db0f454800d20bf9370081b962'),
 'Windows':('x86_64-pc-windows-msvc.zip','f41d12526748576e035d9f339981762a882269990921928e7dc1732a2430f325'),
}

def install():
    system=platform.system()
    if system not in ASSETS or platform.machine().lower() not in ('x86_64','amd64'):
        raise RuntimeError('Pinned Inkvec build requires Linux/Windows x86_64')
    asset,digest=ASSETS[system]
    name='inkvec-'+VERSION+'-'+asset
    url='https://github.com/logolabs/inkvec/releases/download/v'+VERSION+'/'+name
    with urllib.request.urlopen(url,timeout=60) as response:
        data=response.read(32*1024*1024+1)
    if len(data)>32*1024*1024 or hashlib.sha256(data).hexdigest()!=digest:
        raise RuntimeError('Inkvec release archive failed checksum/size verification')
    executable='inkvec.exe' if system=='Windows' else 'inkvec'
    if name.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            matches=[entry for entry in archive.infolist() if Path(entry.filename).name==executable and not entry.is_dir()]
            if len(matches)!=1 or matches[0].file_size>32*1024*1024:raise RuntimeError('Unexpected executable archive')
            payload=archive.read(matches[0])
    else:
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            matches=[entry for entry in archive.getmembers() if Path(entry.name).name==executable and entry.isfile()]
            if len(matches)!=1 or matches[0].size>32*1024*1024:raise RuntimeError('Unexpected executable archive')
            payload=archive.extractfile(matches[0]).read()
    directory=Path(__file__).resolve().parent/'bin';directory.mkdir(exist_ok=True)
    target=directory/executable;target.write_bytes(payload)
    if system!='Windows':target.chmod(0o755)
    print('Installed checksum-verified Inkvec '+VERSION)

if __name__=='__main__':install()
