"""Report actual own-target pixels, raw frames and pose/time checks; no actor data."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    index=json.loads((a.episode/'visual_diagnostics/index.json').read_text());rows=[]
    for item in index:
        folder=Path(item['path']);m=json.loads((folder/'metadata.json').read_text());row=dict(seed=m['seed'],phase=m['phase'],time_s=m['time_s'],path=str(folder),cameras={})
        for name,c in m['cameras'].items():
            semantic=c['render_outputs'].get('semantic_segmentation',{});mapping=semantic.get('id_to_labels',{});counts=semantic.get('pixel_counts',{})
            own_target=0;self_robot=0;other=0;class_counts={}
            for identifier,labels in mapping.items():
                label=labels.get('class','');pixels=counts.get(str(identifier),0);class_counts[label]=pixels
                if label==f"w8_target_{m['env_index']}":own_target+=pixels
                if label==f"w8_robot_{m['env_index']}":self_robot+=pixels
                if label.startswith(('w8_target_','w8_robot_','w8_table_')) and not label.endswith('_'+str(m['env_index'])):other+=pixels
            row['cameras'][name]=dict(own_target_pixels=own_target,self_robot_pixels=self_robot,other_environment_pixels=other,class_counts=class_counts,
                frame=c['frame'],timestamp_s=c['sensor_timestamp_s'],position=c['live_world_position'],target_optical=c['diagnostic_target_optical_metres'],
                raw_stats=c['render_outputs'].get('rgb',c['render_outputs'].get('distance_to_image_plane')))
        rows.append(row)
    mount=[json.loads(line) for line in (a.episode/'camera_mount_sync.jsonl').open()]
    report=dict(diagnostic_only=True,formal_training=False,source=str(a.episode),frames=rows,
        all_render_clocks_unchanged=all(x.get('render_clock_unchanged',False) for x in mount),
        all_render_physics_states_unchanged=all(x.get('render_state_unchanged',False) for x in mount),
        max_mount_position_error_m=max(row['max_render_position_error_m'] for x in mount for row in x['readbacks']),
        per_seed={str(seed):{name:[dict(phase=x['phase'],time_s=x['time_s'],pixels=x['cameras'][name]['own_target_pixels']) for x in rows if x['seed']==seed] for name in ['head_cam','wrist_cam']} for seed in sorted({x['seed'] for x in rows})},
        visual_correctness_requires_actual_image_review=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['all_render_clocks_unchanged','all_render_physics_states_unchanged','max_mount_position_error_m']}))


if __name__=='__main__':main()
