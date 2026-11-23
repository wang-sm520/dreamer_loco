"""Bounded diagnostic video from already acquired actor camera observations."""
import json,subprocess
import numpy as np
from PIL import Image,ImageDraw


class RecordingVisual:
    def __init__(self,visual,guard):
        self.visual=visual;self.guard=guard;self.saved={};self.frames={}
    def manifest(self):return self.visual.manifest()
    def begin(self,out,driver,seed):
        self.out=out;self.driver=driver;self.seed=seed;self.index=0;self.frames={}
        self.saved.setdefault(driver,set())
        from .final_pose_audit import ActualPoseRecorder
        self.pose_recorder=ActualPoseRecorder(out,seed)
    def observe(self,scene):
        images,rgb,wrist=self.visual.observe(scene)
        if scene is not None:self.pose_recorder.observe(scene,self.index)
        if len(self.saved[self.driver])<2:
            for i in range(min(8,len(rgb))):
                head=Image.fromarray(rgb[i].detach().cpu().numpy().astype(np.uint8)).resize((240,136))
                depth=Image.fromarray((images[i,1,0].detach().cpu().numpy().clip(0,1)*255).astype(np.uint8)).convert('RGB')
                canvas=Image.new('RGB',(368,136));canvas.paste(head);canvas.paste(depth,(240,0))
                draw=ImageDraw.Draw(canvas);draw.rectangle((0,0,367,12),fill='black')
                draw.text((2,1),f'{self.driver} seed={self.seed+i} t={self.index*.16:.2f}s',fill='white')
                self.frames.setdefault(i,[]).append(np.asarray(canvas).copy())
        self.index+=1
        return images,rgb,wrist
    def finish(self,result):
        self.pose_recorder.finish()
        clips=[]
        for i,episode in enumerate(result['episodes']):
            kind='success' if episode['success'] else 'failure'
            if kind in self.saved[self.driver] or i not in self.frames:continue
            self.guard.check(20_000_000)
            path=self.out/f"{kind}_seed_{episode['seed']}.mp4"
            command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-f','rawvideo',
                '-pix_fmt','rgb24','-s','368x136','-r','6.25','-i','pipe:0','-an','-c:v','libx264',
                '-threads','2','-crf','28','-pix_fmt','yuv420p',str(path)]
            job=subprocess.Popen(command,stdin=subprocess.PIPE)
            try:
                for frame in self.frames[i]:job.stdin.write(frame.tobytes())
            finally:job.stdin.close()
            if job.wait():raise RuntimeError('Video evidence encoding failed')
            clips.append(dict(path=str(path),seed=episode['seed'],reason=episode['reason'],
                frames=len(self.frames[i]),fps=6.25,source='actual acquired head RGB + encoded wrist depth',
                first_time_s=0,frame_interval_s=.16,physical_outcome_steps=episode['steps']))
            self.saved[self.driver].add(kind)
        self.frames={}
        (self.out/'video_evidence.json').write_text(json.dumps(clips,indent=2))
