"""Cost-only calibration. No data, val or CORE scores influence selection.

Writes reproducible configurations and detailed operator records. Candidates
never become completed experiments: cumulative training/evaluation audits are
separate requirements. Use a task-local --cache for intermediate fake traces.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import torch

from nexttransformer.models.config import CHUNK_BACKENDS, CHUNKED_METHODS, V2_CALIBRATION, LAYER_MULTIPLE, METHODS, LINEAR, candidate_config
from nexttransformer.protocol.spec import config_file, Protocol, load_config, write_config
from nexttransformer.protocol.compute import FORMULA_VERSION,audit_graph
from nexttransformer.protocol.fingerprint import VERSION as FINGERPRINT_VERSION,record as source_record,traced
from nexttransformer.models.looped import huginn_depth
from nexttransformer.models.plugins import stages as plugin_stages
from nexttransformer.models.registry import PLUGIN_METHODS


def science_hash():
    """Hash the current implementation, including relocated model dependencies.

    Compatibility aliases are not evidence for the implementation they import.
    Moving or editing scientific code requires a fresh audit; old proofs remain
    immutable and are never relabelled as current evidence.
    """
    root=Path(__file__).resolve().parents[2]
    files=[root/"nexttransformer"/name for name in
           ("protocol/spec.py","training/optimizer.py","protocol/compute.py","protocol/calibrate.py")]
    files+=sorted((root/"nexttransformer"/"models").rglob("*.py"))
    files+=[root/"nanochat"/name for name in
            ("common.py","flash_attention.py","optim.py")]
    digest=hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Auditor:
    def __init__(self,cache):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.source=science_hash();self.executed=set()

    def graph(self,c,stage,depth,t,optimizer=True):
        """Audited graph of one stage and length; records the repository files the audit executed."""
        identity=dict(source=self.source,model=c.fingerprint,stage=stage,depth=depth,t=t,batch=512,
                      optimizer=optimizer,formula=FORMULA_VERSION,torch=torch.__version__,fingerprint=FINGERPRINT_VERSION)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=self.cache/(key+".json")
        if path.exists():
            value=json.loads(path.read_text());self.executed.update(value["source_files"]);return value
        started=time.monotonic()
        with traced() as executed:
            value=audit_graph(c,sequence_len=t,batch=512,stage=stage,depth=depth,optimizer=optimizer)
        if value["unknown"]:raise RuntimeError(f"uncovered floating operators: {value['unknown']}")
        value.update(identity=identity,audit_seconds=time.monotonic()-started,source_files=sorted(executed))
        self.executed.update(executed)
        path.write_text(json.dumps(value,indent=2)+"\n")
        return value

    @staticmethod
    def probe_lengths(c):
        """Scan probes are 16/32/48 tokens; chunk probes are 3/4/5 whole chunks.

        The first chunk has no state read, the last no state update, and the
        chunk-level decay is first applied by the second update (third chunk),
        so the per-chunk structure is affine from three chunks onward.
        """
        chunk=CHUNK_BACKENDS.get(c.backend)
        return (16,32,48) if chunk is None else (3*chunk,4*chunk,5*chunk)

    def affine(self,c,stage,depth):
        """Operator-wise affine extrapolation to the full context, verified at a third length."""
        lengths=self.probe_lengths(c);step=lengths[1]-lengths[0]
        probes=[self.graph(c,stage,depth,t) for t in lengths]
        keys=set().union(*(r["operators"] for r in probes))
        ops={}
        for key in keys:
            item={}
            for field in ("calls","theory","padded_envelope","matrix"):
                a,b,test=[r["operators"].get(key,{}).get(field,0) for r in probes]
                if test!=2*b-a:raise RuntimeError(f"non-affine scan count: {key}/{field}")
                numerator=(b-a)*(c.sequence_len-lengths[0])
                if numerator%step:raise RuntimeError("nonintegral extrapolated arithmetic")
                item[field]=a+numerator//step
            ops[key]=item
        return probes[0],ops,lengths

    def full(self,c,stage="joint",depth=None):
        if c.method not in LINEAR:return self.graph(c,stage,depth,c.sequence_len)
        if c.backend in CHUNK_BACKENDS:
            # A 2048-token graph is only 32 chunks: audit it directly, and require
            # the operator-wise affine extrapolation from 1/2/3 chunks to agree.
            direct=self.graph(c,stage,depth,c.sequence_len)
            _,ops,lengths=self.affine(c,stage,depth)
            if ops!={k:{f:v[f] for f in ("calls","theory","padded_envelope","matrix")}
                     for k,v in direct["operators"].items()}:
                raise RuntimeError("direct chunk audit differs from the affine cross-check")
            out=dict(direct)
            out["extrapolation"]={"kind":"direct full-context audit; operator-wise affine cross-check",
                                  "verified_lengths":list(lengths),"target_length":c.sequence_len}
            return out
        # Every scan iteration is the same fixed-dimensional recurrence, while
        # projections/conv/norm are affine in T. Verify EACH operator's affine
        # count with a third independently executed graph before extrapolating.
        first,ops,lengths=self.affine(c,stage,depth)
        out=dict(first)
        out["operators"]=dict(sorted(ops.items()));out["sequence_len"]=c.sequence_len
        out["phases"]={phase:sum(v["theory"] for k,v in ops.items() if k.startswith(phase+"/"))
                       for phase in ("forward","backward","optimizer")}
        out["total"]=sum(out["phases"].values())
        out["matrix_total"]=sum(v["matrix"] for v in ops.values())
        out["padded_envelope"]=sum(v["padded_envelope"] for v in ops.values())
        out["extrapolation"]={"kind":"operator-wise affine scan","verified_lengths":list(lengths),
                              "target_length":c.sequence_len}
        return out

    def stages(self,c):
        p=Protocol()
        if c.method in PLUGIN_METHODS:return [(name,None,steps) for name,steps in plugin_stages(c)]
        if c.method=="huginn":
            counts=Counter(huginn_depth(c,step) for step in range(p.steps))
            return [("joint",depth,counts[depth]) for depth in sorted(counts)]
        if c.method=="ouro":return [("joint",None,p.steps-c.ouro_gate_steps),("gate_only",None,c.ouro_gate_steps)]
        if c.has_sparse_burnin:
            return [("burnin",None,c.burnin_steps),("joint",None,p.steps-c.burnin_steps)]
        return [("joint",None,p.steps)]

    def aggregate(self,c):
        records=[];weighted=0;self.executed=set()
        for stage,depth,steps in self.stages(c):
            record=self.full(c,stage,depth)
            records.append(dict(stage=stage,depth=depth,optimizer_steps=steps,cost=record))
            weighted+=steps*record["total"]
        return dict(method=c.method,model_hash=c.fingerprint,source_hash=self.source,
                    method_source=source_record(c.method,self.executed,c),
                    protocol_hash=Protocol().fingerprint,formula_version=FORMULA_VERSION,
                    torch=torch.__version__,
                    parameters=records[0]["cost"]["parameters"],
                    total_training_flops=weighted,flops_per_token=weighted/Protocol().total_tokens,
                    stages=records,unknown={})

    def ffn_slope(self,c):
        """FFN-only change: exact finite difference with two sequence lengths.

        The grid never crosses d; Muon matrix orientation and iteration counts
        stay fixed. An independent full audit of the chosen config is required.
        """
        other=replace(c,mlp_width=c.mlp_width+128)
        weighted=0;first,second=self.probe_lengths(c)[:2]
        for stage,depth,steps in self.stages(c):
            delta=[]
            for t in (first,second):
                before=self.graph(c,stage,depth,t);after=self.graph(other,stage,depth,t)
                delta.append(after["total"]-before["total"])
            full=delta[0]+(delta[1]-delta[0])*(c.sequence_len-first)//(second-first)
            weighted+=steps*full
        return weighted/Protocol().total_tokens/128

    def structure(self,c,initial,target):
        """Try adjacent integer depths before changing the FFN expansion.

        Registered grid: ordinary stacks 16..32; shared cores 1..16, holding
        loop count, pre/coda and algorithm settings fixed. Stop at the first
        bracket around the target; never inspect language modelling scores.
        """
        if abs(initial["flops_per_token"]/target-1)<=Protocol().preflight_tolerance:
            return c,initial,[]
        field="core_layers" if c.method in ("ut_act","fixed_recursive","huginn","ouro","deeploop") else "layers"
        lo,hi=(1,16) if field=="core_layers" else (16,32)
        direction=1 if initial["flops_per_token"]<target else -1
        best_c,best=c,initial;trials=[]
        previous=initial["flops_per_token"]-target
        for value in range(getattr(c,field)+direction,hi+1 if direction>0 else lo-1,direction):
            trial_c=replace(c,**{field:value});trial=self.aggregate(trial_c)
            error=trial["flops_per_token"]-target
            trials.append(dict(field=field,value=value,flops_per_token=trial["flops_per_token"]))
            if abs(error)<abs(best["flops_per_token"]-target):best_c,best=trial_c,trial
            if error*previous<=0 or abs(error/target)<=Protocol().preflight_tolerance:break
            previous=error
        return best_c,best,trials

    def v2_grid(self,c,target):
        """PROTOCOL v2 selection for hybrids and their controls, with the repeating unit locked.

        Audit every allowed depth in 16..32 at the starting FFN and take its exact
        FFN slope; estimate each FFN on the [3d, 5d] 128 grid; then fully audit the
        estimated-within-1% candidates in tie-break order (FFN closest to the start,
        smallest |error|, depth closest to the start, shallower) and return the
        first one whose full audit is within tolerance. Publishes the whole table.
        """
        tol=Protocol().preflight_tolerance;depths=[];candidates=[]
        for n in range(16,33):
            if n%LAYER_MULTIPLE.get(c.method,1):continue
            trial=replace(c,layers=n)
            cost=self.aggregate(trial)["flops_per_token"];slope=self.ffn_slope(trial)
            depths.append(dict(layers=n,flops_per_token=cost,ffn_slope=slope))
            for width in range(3*c.width,5*c.width+1,128):
                estimate=(cost+(width-c.mlp_width)*slope)/target-1
                if abs(estimate)<=tol:candidates.append(dict(layers=n,mlp_width=width,estimated_error=estimate))
        candidates.sort(key=lambda t:(abs(t["mlp_width"]-c.mlp_width),abs(t["estimated_error"]),
                                      abs(t["layers"]-c.layers),t["layers"]))
        table=dict(rule="v2 tie-break",start=dict(layers=c.layers,mlp_width=c.mlp_width),
                   depths=depths,candidates=candidates)
        for candidate in candidates:
            chosen=replace(c,layers=candidate["layers"],mlp_width=candidate["mlp_width"])
            selected=self.aggregate(chosen)
            candidate["audited_error"]=selected["flops_per_token"]/target-1
            if abs(candidate["audited_error"])<=tol:return chosen,selected,table
        raise RuntimeError(f"{c.method}: no registered-grid candidate within tolerance; record calibration_pending")


def calibrate_one(args,method,dense,source):
    torch.set_num_threads(1)
    auditor=Auditor(args.cache)
    if auditor.source!=source:raise RuntimeError("source changed during calibration")
    target=dense["flops_per_token"];started=time.monotonic()
    previous=config_file(args.reuse_configs,method) if args.reuse_configs else None
    reuse=previous is not None and previous.exists()
    c=load_config(previous) if reuse else candidate_config(method)
    if method=="dense" and c!=candidate_config("dense"):raise ValueError("Dense d24 anchor is immutable")
    selection_provenance=json.loads(previous.read_text())["calibration"] if reuse else None
    # A backend switch keeps the audited structure as the starting point and only
    # runs the registered grid search if the new training algorithm exceeds 1%.
    switched=bool(args.switch_backend) and method in CHUNKED_METHODS
    if switched:
        if not reuse:raise ValueError("--switch-backend starts from an existing --reuse-configs structure")
        selection_provenance=dict(previous=selection_provenance,backend_switch=dict(
            from_backend=c.backend,to_backend=args.switch_backend,previous_model_hash=c.fingerprint))
        c=replace(c,backend=args.switch_backend)
    fixed=reuse and not switched
    initial=dense if method=="dense" else auditor.aggregate(c)
    if fixed:chosen,selected,structural_trials=c,initial,[]
    elif method in V2_CALIBRATION:chosen,selected,structural_trials=auditor.v2_grid(c,target)
    else:chosen,selected,structural_trials=auditor.structure(c,initial,target)
    if not fixed and method not in V2_CALIBRATION and abs(selected["flops_per_token"]/target-1)>Protocol().preflight_tolerance:
        slope=auditor.ffn_slope(chosen)
        wanted=chosen.mlp_width+(target-selected["flops_per_token"])/slope
        width=128*round(wanted/128)
        if not 3*c.width<=width<=5*c.width:
            raise RuntimeError(f"{method}: registered FFN grid cannot meet budget; revise structural grid explicitly")
        chosen=replace(chosen,mlp_width=width);selected=auditor.aggregate(chosen)
    deviation=selected["flops_per_token"]/target-1
    passed=abs(deviation)<=Protocol().preflight_tolerance and not selected["unknown"]
    status="theoretical_preflight_passed" if passed else "budget_mismatch"
    calibration=dict(status=status,formula_version=FORMULA_VERSION,source_hash=auditor.source,
                     method_source_hash=selected["method_source"]["hash"],
                     torch=torch.__version__,dense_flops_per_token=target,
                     flops_per_token=selected["flops_per_token"],relative_error=deviation,
                     cumulative_run_audit="not_run",
                     cost_record=os.path.relpath(args.results/(method+".json"),config_file(args.output,method).parent))
    selected.update(calibration=calibration,initial_ffn=c.mlp_width,selected_ffn=chosen.mlp_width,
                    physical_layers=chosen.n_layer,core_layers=chosen.core_layers,
                    structural_trials=structural_trials,verification_only=fixed,
                    selection_provenance=selection_provenance)
    if science_hash()!=source:raise RuntimeError("source changed during calibration")
    (args.results/(method+".json")).write_text(json.dumps(selected,indent=2)+"\n")
    write_config(config_file(args.output,method),chosen,calibration)
    if not passed:raise RuntimeError(f"{method}: cost preflight failed; architecture was not silently changed")
    return dict(method=method,status=status,parameters=selected["parameters"],ffn=chosen.mlp_width,
                physical_layers=chosen.n_layer,core_layers=chosen.core_layers,
                flops_per_token=selected["flops_per_token"],relative_error=deviation,
                elapsed_seconds=time.monotonic()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    p.add_argument("--cache",required=True,type=Path)
    p.add_argument("--output",type=Path,default=Path("configs"))
    p.add_argument("--results",type=Path,default=Path("evidence/costs"))
    p.add_argument("--reuse-configs",type=Path,
                   help="re-audit existing structures unchanged; run the registered search for missing methods")
    p.add_argument("--switch-backend",choices=sorted(CHUNK_BACKENDS),
                   help="move KDA/RetNet/ELU+1 to this chunked backend, starting from --reuse-configs")
    p.add_argument("--workers",type=int,default=1,help="independent CPU processes; no GPU use")
    args=p.parse_args()
    if len(set(args.methods))!=len(args.methods) or not 1<=args.workers<=32:raise ValueError("invalid method/worker list")
    torch.set_num_threads(1);auditor=Auditor(args.cache)
    dense=auditor.aggregate(candidate_config("dense"))
    args.results.mkdir(parents=True,exist_ok=True);rows=[]
    if "dense" not in args.methods and not (args.results/"dense.json").exists():
        # Admission reads the cost anchor next to every proof, so a proof set always carries it.
        (args.results/"dense.json").write_text(json.dumps(dense,indent=2)+"\n")
    def record(row):
        rows.append(row);print(json.dumps(row),flush=True)
        (args.results/"summary.json").write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
                source_hash=auditor.source,formula_version=FORMULA_VERSION,torch=torch.__version__,
                rows=sorted(rows,key=lambda r:METHODS.index(r["method"]))),indent=2)+"\n")
    if args.workers==1:
        for method in args.methods:record(calibrate_one(args,method,dense,auditor.source))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            futures=[pool.submit(calibrate_one,args,method,dense,auditor.source) for method in args.methods]
            for future in as_completed(futures):record(future.result())


if __name__=="__main__":main()
