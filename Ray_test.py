import ray
import time
'''
ray.shutdown()
ray.init()

@ray.remote
def A(a):
    print('xxxxxxxxx')
    return a + 5

@ray.remote
def B(a):
    time.sleep(20)
    print('aaaaaaaa')
    return a+10

for i in range(10):
    a_id = A.remote(i)
    b_id = B.remote(ray.get(a_id))
    ray.get(b_id)
ray.shutdown()
'''
import threading

def generate(x):
    return x * 2

def calculate(x):
    return x + 10

for i in range(10):
    t1 = threading.Thread(target=generate(), args=i)
    t2 = threading.Thread(target=generate(), args=i)















