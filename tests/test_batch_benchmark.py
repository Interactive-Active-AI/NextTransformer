import json
import pytest
from launch.benchmark_native_batches import telemetry_summary


def test_telemetry_uses_gpu_uuid_and_common_training_window(tmp_path):
    # Visible CUDA device order need not be NVIDIA's physical index order.
    ranks=[{'rank':0,'uuid':'b','measurement_start':10,'measurement_end':20},
           {'rank':1,'uuid':'GPU-a','measurement_start':11,'measurement_end':19}]
    rows=[]
    for index,uuid in ((0,'GPU-a'),(3,'GPU-b')):
        for t in range(9,22):
            inside=11<=t<=19
            rows.append({'time':t,'index':index,'uuid':uuid,'gpu_util':80 if inside else 0,
                         'memory_used':100 if inside else 900,'memory_total':1000,'power':200,'sm_clock':1200})
    path=tmp_path/'samples.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    result=telemetry_summary(path,{'ranks':ranks})
    assert result['interval_start']==11 and result['interval_end']==19
    assert [x['gpu_index'] for x in result['per_gpu']]==[3,0]
    for row in result['per_gpu']:
        assert row['samples']==9 and row['gpu_util_mean_pct']==80
        assert row['memory_used_peak_pct']==10 and row['memory_used_peak_mib']==100
        assert row['memory_used_peak_mib_entire_trial']==900


def test_missing_telemetry_is_not_reported_as_zero(tmp_path):
    path=tmp_path/'samples.jsonl';path.write_text('')
    with pytest.raises(ValueError,match='no telemetry'):
        telemetry_summary(path,{'ranks':[{'rank':0,'uuid':'GPU-a','measurement_start':10,'measurement_end':20}]})


def test_nondividing_batch_is_rejected_before_gpu_launch(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    script=Path(__file__).resolve().parents[1]/'launch/benchmark_native_batches.py'
    output=tmp_path/'invalid'
    result=subprocess.run([sys.executable,str(script),'--output',str(output),'--batches','48'],capture_output=True,text=True)
    assert result.returncode==2 and 'does not divide the fixed global token batch' in result.stderr
    assert not output.exists()
