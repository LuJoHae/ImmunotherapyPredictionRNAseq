import immunotherapypredictionrnaseq.utils
import numpy as np

def test_fixseed():
    seed = 42
    n = 10
    immunotherapypredictionrnaseq.utils.fixseed(seed)
    a = np.random.random(n)
    immunotherapypredictionrnaseq.utils.fixseed(seed)
    b = np.random.random(n)
    assert np.all(a==b)