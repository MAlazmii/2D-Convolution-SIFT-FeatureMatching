from pathlib import Path
import ast
import json
import importlib.util
import cv2
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]

def old_convolution():
    cell=json.loads((ROOT/'Assignment-task1COMP338.ipynb').read_text())['cells'][0]
    body=ast.parse(''.join(cell['source'])).body
    funcs=[node for node in body if isinstance(node,ast.FunctionDef) and node.name=='custom_convolution']
    namespace={'np':np}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'original','exec'),namespace)
    return namespace.get('custom_convolution')

def utility(name):
    path=ROOT/'vision_utils.py'
    if not path.exists():
        if name=='custom_convolution':return old_convolution()
        pytest.fail('Independent image utility is missing: '+name)
    spec=importlib.util.spec_from_file_location('vision_utils',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return getattr(m,name)

def test_filter_preserves_signed_response_and_kernel_direction():
    convolution=utility('custom_convolution')
    image=np.zeros((5,5),dtype=np.uint8);image[2,2]=255
    kernel=np.array([[0,0,0],[-1,0,1],[0,0,0]],dtype=float)
    result=convolution(image,kernel)
    assert result[2,1]==-255 and result[2,3]==255
    assert result.dtype.kind=='f'

def test_filter_matches_zero_padded_convolution():
    image=np.arange(20,dtype=float).reshape(4,5)
    kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=float)
    result=utility('custom_convolution')(image,kernel)
    expected=cv2.filter2D(image,-1,np.flip(kernel),borderType=cv2.BORDER_CONSTANT)
    np.testing.assert_allclose(result,expected)

def test_resize_uses_width_and_preserves_aspect():
    a,b=utility('resize_pair')(np.zeros((40,80,3),np.uint8),np.zeros((60,120,3),np.uint8))
    assert a.shape==(40,80,3) and b.shape==(40,80,3)

def test_missing_input_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):utility('read_image')(tmp_path/'absent.png')

def test_orb_uses_binary_distance():
    # L2 picks 3; Hamming picks 128 (one changed bit vs two).
    query=np.array([[0]],np.uint8);train=np.array([[3],[128]],np.uint8)
    matches=utility('ratio_matches')(query,train,binary=True)
    assert len(matches)==1 and matches[0].trainIdx==1

def test_featureless_and_single_neighbor_are_safe():
    f=utility('ratio_matches')
    assert f(None,None,binary=True)==[]
    assert f(np.zeros((1,32),np.uint8),np.zeros((1,32),np.uint8),binary=True)==[]

@pytest.mark.parametrize('filename',['Assignment-task1COMP338.ipynb','Assignment-task2COMP338.ipynb'])
def test_notebook_runs_from_fresh_namespace_with_generated_images(filename,tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.chdir(tmp_path)
    rng=np.random.default_rng(42)
    first=rng.integers(0,256,(48,64,3),dtype=np.uint8)
    second=np.roll(first,3,axis=1)
    cv2.imwrite(str(tmp_path/'victoria1.jpg'),first)
    cv2.imwrite(str(tmp_path/'victoria2.jpg'),second)
    monkeypatch.setattr(plt,'show',lambda:None)
    scope={}
    for cell in json.loads((ROOT/filename).read_text())['cells']:
        if cell['cell_type']=='code':exec(compile(''.join(cell['source']),filename,'exec'),scope)
    assert scope['image'].shape==(48,64) if filename.startswith('Assignment-task1') else scope['image1'].shape==(48,64,3)
    plt.close('all')
