"""Dense runtime must route a one-socket plan to that physical socket."""
from types import SimpleNamespace

import pytest

from sglang.srt.layers.prism.linear.method import _LinearRuntime


@pytest.mark.parametrize('mapping,threads,expected', [('1',7,[7]),('1,0',7,[4,3])])
def test_explicit_physical_numa_map(monkeypatch,mapping,threads,expected):
    from kt_kernel import kt_kernel_ext
    from sglang.srt.layers.prism import numa
    from sglang.srt.layers.prism.linear import cold_backend

    monkeypatch.setenv('SGLANG_PRISM_NUMA_MAP',mapping)
    monkeypatch.setenv('SGLANG_PRISM_CPUINFER_THREADS',str(threads))
    monkeypatch.delenv('SGLANG_PRISM_LINEAR_COLD_THREADS',raising=False)
    monkeypatch.setattr(numa,'numa_node_count',lambda:2)
    monkeypatch.setattr(kt_kernel_ext,'CPUInfer',lambda cfg:cfg)
    monkeypatch.setattr(cold_backend,'KtLinearColdBackend',lambda plan,**kw:kw)
    runtime=_LinearRuntime.__new__(_LinearRuntime)
    runtime.max_tokens=1
    runtime.plan=SimpleNamespace(projs={0:SimpleNamespace(has_tier=lambda tier:True)})
    backend,_=runtime._cold_backend()
    assert backend['num_numa_nodes']==len(expected)
    assert backend['cpuinfer'].subpool_numa_map==list(map(int,mapping.split(',')))
    assert backend['cpuinfer'].subpool_thread_count==expected


@pytest.mark.parametrize('mapping',['1,1','2','-1'])
def test_invalid_physical_numa_map(monkeypatch,mapping):
    from sglang.srt.layers.prism import numa
    monkeypatch.setattr(numa,'numa_node_count',lambda:2)
    monkeypatch.setenv('SGLANG_PRISM_NUMA_MAP',mapping)
    monkeypatch.setenv('SGLANG_PRISM_CPUINFER_THREADS','7')
    runtime=_LinearRuntime.__new__(_LinearRuntime)
    runtime.plan=SimpleNamespace(projs={0:SimpleNamespace(has_tier=lambda tier:True)})
    with pytest.raises(ValueError,match='Invalid NUMA map'):
        runtime._cold_backend()


@pytest.mark.parametrize('fails',[False,True])
def test_load_threads_restored(monkeypatch,fails):
    import torch
    from sglang.srt.layers.prism.linear.method import PrismLinearMethod
    state=[1]
    monkeypatch.setenv('SGLANG_PRISM_LINEAR_LOAD_THREADS','7')
    monkeypatch.setattr(torch,'get_num_threads',lambda:state[0])
    monkeypatch.setattr(torch,'set_num_threads',lambda n:state.__setitem__(0,n))
    def prepare(layer):
        assert state[0]==7
        if fails: raise RuntimeError('preparation failed')
        return 'prepared'
    method=SimpleNamespace(_prepare_weights_after_loading=prepare)
    if fails:
        with pytest.raises(RuntimeError,match='preparation failed'):
            PrismLinearMethod.process_weights_after_loading(method,None)
    else:
        assert PrismLinearMethod.process_weights_after_loading(method,None)=='prepared'
    assert state[0]==1
