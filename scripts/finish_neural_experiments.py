"""Select on development views, then evaluate the fixed final cameras once."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import subprocess
import argparse
from scenequery.common import ROOT, read_json, atomic_json, sha256


def execute(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study');args=parser.parse_args()
    directory=ROOT/'reports/neural'
    study=read_json(ROOT/'config/neural'/f'study_{args.study}.json') if args.study else None
    study_dir=directory/'studies'/args.study if study else directory
    run_names=list(study['runs']) if study else ['rgbd_s42','rgb_s42','rgbd_s43']
    full_names=study['full_runs'] if study else ['rgbd_s42','rgbd_s43']
    evaluator=study.get('evaluator','evaluate_radiance.py') if study else 'evaluate_radiance.py'
    freeze=read_json(study_dir/'experiment_freeze.json')
    for path,digest in freeze['files'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Frozen experiment input changed: '+path)
    candidates={name:read_json(directory/name/'checkpoint_development.json')['summary']
                for name in full_names}
    valid=[name for name,s in candidates.items() if s['depth_mae_m']['mean']<=.12]
    if not valid:raise RuntimeError('No full run passes the predeclared development-depth condition')
    selected=max(valid,key=lambda name:candidates[name]['psnr_db']['mean'])
    selection={'selected_run':selected,'development_metrics':candidates,
               'rule':'Highest mean development PSNR among full variants with mean development depth MAE <= 0.12 m.',
               'declared_before_final_evaluation':True,
               'protocol_sha256':sha256(ROOT/'config/neural/protocol.json')}
    path=study_dir/'selection.json'
    if path.exists() and read_json(path)!=selection:raise RuntimeError('Cannot change frozen selection')
    atomic_json(path,selection)
    for name in run_names:
        execute(evaluator,'--run',name,'--split','final')
    for checkpoint in ['initial.pt','step_000500.pt','step_002000.pt']:
        execute(evaluator,'--run',selected,'--checkpoint',checkpoint,'--split','final')
    execute('neural_reference.py')
    execute('export_radiance.py','--run',selected)
    execute('analyze_radiance.py',*(['--study',args.study] if args.study else []))


if __name__=='__main__':main()
