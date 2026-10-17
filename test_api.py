import unittest
import json
from fastapi.testclient import TestClient
from main import app

class ApiTests(unittest.TestCase):
    def setUp(self): self.client=TestClient(app)
    def test_health(self): self.assertEqual(self.client.get('/health').json()['status'],'ok')
    def test_trace_and_size_validation(self):
        headers={'X-Width':'10','X-Height':'10','X-Settings':json.dumps({'edgeStyle':'smooth','colors':8})}
        result=self.client.post('/api/trace',content=bytes([20,80,140,255])*100,headers=headers)
        self.assertEqual(result.status_code,200); self.assertIn('<path',result.json()['svg'])
        self.assertEqual(self.client.post('/api/trace',content=b'x',headers=headers).status_code,400)
    def test_cors(self):
        production=self.client.options('/api/trace',headers={'Origin':'https://vector-gamma-one.vercel.app','Access-Control-Request-Method':'POST'})
        self.assertEqual(production.status_code,200)
        self.assertEqual(production.headers['access-control-allow-origin'],'https://vector-gamma-one.vercel.app')
        result=self.client.options('/api/trace',headers={'Origin':'http://localhost:3000','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'X-Width,X-Height,X-Settings,Content-Type'})
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.headers['access-control-allow-origin'],'http://localhost:3000')
        result=self.client.options('/api/trace',headers={'Origin':'https://unapproved.example','Access-Control-Request-Method':'POST'})
        self.assertEqual(result.status_code,400)
    def test_busy(self):
        from main import capacity
        capacity.acquire()
        try:
            result=self.client.post('/api/trace',content=bytes(400),headers={'X-Width':'10','X-Height':'10'})
            self.assertEqual(result.status_code,503)
        finally: capacity.release()
    def test_malformed_settings_rejected_before_tracing(self):
        for settings in [{'preset':[]},{'edgeStyle':3},{'denoise':'false'},{'autoColors':1},{'smooth':True},{'rounding':float('inf')},{'colors':10**1000}]:
            with self.subTest(settings=settings):
                result=self.client.post('/api/trace',content=bytes(400),headers={'X-Width':'10','X-Height':'10','X-Settings':json.dumps(settings)})
                self.assertEqual(result.status_code,400)

if __name__=='__main__': unittest.main()
