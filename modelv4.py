#!/usr/bin/env python3
"""CSE 6242 Team 185 flight risk model

This is an updated version of modelv3.py. The script uses existing code structure and compares models on all flights and on flights w/ valid aircraft rotations. The results can be exported for the dashboard. 

you can use:
    python modelv4.py audit
    python modelv4.py prepare --download
    python modelv4.py run --run corrected-2025

This uses schedule and historical risk data from BTS. The RF model requires scikit-learn, and the 2025 results are exploratory.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from flight_risk_v4.common import CLASS_NAMES, load_config, sha256, utcnow, write_json
from flight_risk_v4.data import connect, files, require_coverage
from flight_risk_v4.models import checked_proba
from flight_risk_v4.team_experiment import EB_FEATURES, evaluate_team, normalized, train_team
from flight_risk_v4.diagnostics import extra_diagnostics, verify_run, dashboard_export

HERE = Path(__file__).resolve().parent


def configuration(path):
    config = load_config(path)
    config['raw_data_dir'] = str((Path(path).resolve().parent / config['raw_data_dir']).resolve())
    history = config['hierarchy_history_period']
    if history[0] > history[1] or history[1] >= config['splits']['initial_fit'][0]:
        raise ValueError('Historical outcome features must predate all model fitting')
    if config['pooling_display_period'][1] >= config['splits']['test'][0]:
        raise ValueError('Display hierarchy cannot use test outcomes')
    return config


def data_audit(config, output):
    coverage = require_coverage(config)
    db = connect(config)
    try:
        years = db.execute("""SELECT year,count(*) flights,
            sum(rotation_status='matched') linked_flights,sum(outcome=2) cancellations,
            sum(outcome=2 AND rotation_status='matched') linked_cancellations
            FROM read_parquet(?,hive_partitioning=false) GROUP BY year ORDER BY year""",
            [files(config)]).fetchdf()
        statuses = db.execute("""SELECT rotation_status,count(*) flights,sum(outcome=2) cancellations
            FROM read_parquet(?,hive_partitioning=false) GROUP BY rotation_status ORDER BY flights DESC""",
            [files(config)]).fetchdf()
    finally:
        db.close()
    path = Path(config['raw_data_dir']) / 'ingestion-manifest.json'
    counts = json.loads(path.read_text()).get('totals') if path.exists() else None
    report = {'created_at': utcnow(), 'coverage': coverage,
        'yearly_linkage': years.to_dict('records'), 'rotation_statuses': statuses.to_dict('records'),
        'ingestion_counts': counts, 'synthetic': False,
        'interpretation': 'All-flight schedule models and linked-rotation comparisons use distinct populations.'}
    write_json(output / 'data-audit.json', report)
    return report


def prepare(config, allow_download):
    from flight_risk_v4.download import download
    from flight_risk_v4.pipeline import ingest
    from flight_risk_v4.rotations import build_rotations
    if allow_download:
        download(config['raw_data_dir'], config['years'])
    ingest(config['raw_data_dir'], config['years'], chunk_rows=config['csv_chunk_rows'])
    source_config = {**config, 'data_dir': config['raw_data_dir']}
    build_rotations(source_config, config['data_dir'], workers=config['threads'])


def predict_profiles(bundle_path, input_path, output_path):
    bundle = joblib.load(bundle_path)
    frame = pd.read_csv(input_path, dtype=str)
    validate_profile_inputs(frame, bundle['config']['require_matched_rotation'])
    if bundle['config']['require_matched_rotation']:
        if 'rotation_status' not in frame or not frame.rotation_status.eq('matched').all():
            raise ValueError('Rotation model requires valid matched-rotation status')
    # schedule models need to work even if the user has no aircraft assignment.
    for key, value in [('rotation_minutes', np.nan), ('rotation_bucket', 'unknown')]:
        if key not in frame:
            frame[key] = value
    frame = normalized(frame)
    historical = bundle['feature_pool'].predict_proba(frame)
    for j, feature in enumerate(EB_FEATURES):
        frame[feature] = historical[:, j]
    output = frame.copy()
    for name, estimator in bundle['models'].items():
        p = checked_proba(estimator, frame)
        for j, c in enumerate(CLASS_NAMES):
            output[f'{name}_{c}'] = p[:, j]
        output[name + '_predicted_class'] = np.asarray(CLASS_NAMES)[p.argmax(axis=1)]
    output.to_csv(output_path, index=False)


def validate_profile_inputs(frame, rotation=False):
    """Reject invalid requests rather than silently encoding them as unknown."""
    required={'origin_id','destination_id','route_id','carrier_id','month','departure_hour',
              'weekday','season','time_bucket','distance','scheduled_duration'}
    if missing:=required-set(frame.columns):
        raise ValueError('Missing profile fields: '+', '.join(sorted(missing)))
    if frame.empty:
        raise ValueError('No profiles supplied')
    ranges={'origin_id':(1,9999999),'destination_id':(1,9999999),'carrier_id':(1,9999999),
            'month':(1,12),'departure_hour':(0,23),'weekday':(1,7)}
    for key,(lo,hi) in ranges.items():
        values=pd.to_numeric(frame[key],errors='coerce')
        if not (values.notna() & values.between(lo,hi) & values.mod(1).eq(0)).all():
            raise ValueError('Invalid '+key)
        frame[key]=values.astype(int).astype(str)
    route=frame.origin_id+'_'+frame.destination_id
    if not frame.route_id.eq(route).all():
        raise ValueError('route_id must agree with origin_id and destination_id')
    month=frame.month.astype(int); hour=frame.departure_hour.astype(int)
    expected_season=month.map({12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',
                               6:'summer',7:'summer',8:'summer',9:'fall',10:'fall',11:'fall'})
    expected_block=np.select([hour.between(5,11),hour.between(12,16),hour.between(17,20)],
                              ['morning','afternoon','evening'],default='night')
    if not frame.season.eq(expected_season).all() or not frame.time_bucket.eq(expected_block).all():
        raise ValueError('Season or departure block conflicts with month/hour')
    if rotation:
        if not {'rotation_minutes','rotation_bucket','rotation_status'}.issubset(frame):
            raise ValueError('Rotation fields are required for this model')
        minutes=pd.to_numeric(frame.rotation_minutes,errors='coerce')
        bucket=np.select([minutes.lt(40),minutes.lt(90),minutes.lt(180)],
                          ['lt40','40_89','90_179'],default='180_1440')
        if not (minutes.notna() & minutes.between(0,1440)).all() or not frame.rotation_bucket.eq(bucket).all():
            raise ValueError('Invalid or inconsistent rotation bucket')


def run(config, run_name):
    started = time.monotonic()
    root = Path(config['artifacts_dir']) / run_name
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'config.resolved.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Use a new run name when the configuration changes')
    write_json(path, config)
    audit = data_audit(config, root)
    results = {}
    for matched, label in [(False, 'all-flights'), (True, 'rotation-linked')]:
        cohort = copy.deepcopy(config)
        cohort['require_matched_rotation'] = matched
        output = root / label
        print('\nRunning population:', label, flush=True)
        train_team(cohort, output)
        # score with the saved model including the frozen 2022 feature history
        bundle = joblib.load(output / 'bundle.joblib')
        report = evaluate_team(bundle, output)
        print('Checking predictions and exporting dashboard data:', label, flush=True)
        diagnostics = extra_diagnostics(bundle, output)
        verification = verify_run(bundle, output)
        export = dashboard_export(bundle, output)
        results[label] = {'models': report['models'], 'population': bundle['population'],
            'paired_comparisons': report['paired_comparisons'],
            'evening_comparison': report['evening_test_comparison'],
            'calibration': bundle['calibration'], 'fit_reports': bundle['fit_reports'],
            'diagnostics': diagnostics, 'verification': verification, 'dashboard_export': export}
    results.update(created_at=utcnow(), seconds=time.monotonic()-started,
        data_audit=audit, test_interpretation=config['test_interpretation'],
        source_provenance=json.loads((HERE / 'provenance.json').read_text()))
    write_json(root / 'findings.json', results)
    print('Completed both populations:', root, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(HERE / 'config.json'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('audit')
    p = sub.add_parser('prepare'); p.add_argument('--download', action='store_true')
    p = sub.add_parser('run'); p.add_argument('--run', default='corrected-2025')
    p = sub.add_parser('predict')
    for name in ['bundle', 'input', 'output']:
        p.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'predict':
        predict_profiles(args.bundle, args.input, args.output)
        return
    config = configuration(args.config)
    if args.command == 'prepare':
        prepare(config, args.download)
    elif args.command == 'audit':
        print(json.dumps(data_audit(config, Path(config['artifacts_dir'])), indent=2))
    else:
        run(config, args.run)


if __name__ == '__main__':
    main()
