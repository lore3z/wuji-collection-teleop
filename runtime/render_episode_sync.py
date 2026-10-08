#!/usr/bin/env python3
"""Render recorded-time video, local hand projections, wrist path and raw EMG."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import zarr
from imagecodecs.numcodecs import register_codecs
EDGES = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17))
COLORS = ((80,80,80),(255,170,40),(80,210,80),(60,190,255),(190,100,255),(255,100,130))

def _project(points, dims, box, scale):
    x0,y0,w,h=box; xy=points[:,dims].astype(float); xy-=xy[0]; xy[:,1]*=-1
    return np.rint(xy*scale+[x0+w/2,y0+h/2]).astype(np.int32)

def _draw_hand(frame, pixels, title, origin):
    cv2.putText(frame,title,origin,cv2.FONT_HERSHEY_SIMPLEX,.7,(235,235,235),2,cv2.LINE_AA)
    for i,(a,b) in enumerate(EDGES): cv2.line(frame,tuple(pixels[a]),tuple(pixels[b]),COLORS[min(i//4+1,5) if i<20 else 0],4,cv2.LINE_AA)
    for i,p in enumerate(pixels): cv2.circle(frame,tuple(p),7 if i==0 else 5,(245,245,245),-1,cv2.LINE_AA)


def nearest(stamps, targets):
    if not len(stamps) or np.any(np.diff(stamps) <= 0):
        raise ValueError('timestamps must be nonempty and strictly increasing')
    right = np.clip(np.searchsorted(stamps, targets), 0, len(stamps)-1)
    left = np.maximum(right-1, 0)
    return np.where(abs(stamps[left]-targets) <= abs(stamps[right]-targets), left, right)


def render(path, output, fps=30):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    register_codecs()
    root = zarr.open_group(str(path), mode='r')
    data, audit, streams = root['data'], root['audit'], root['streams']
    rgb = streams['camera_main_rgb_raw']
    video_ts = streams['camera_main_rgb_timestamp_ns'][:]
    hand = audit['wuji_hand_skeleton_mediapipe'][:]
    hand_ts = data['timestamps'][:]
    xyz = streams['right_wrist_tracker_pose_raw'][:, :3]
    xyz_ts = streams['right_wrist_tracker_pose_timestamp_ns'][:]
    emg = streams['right_forearm_emg_raw'][:]
    emg_ts = streams['right_forearm_emg_timestamp_ns'][:]
    for values, stamps in ((rgb, video_ts), (hand, hand_ts), (xyz, xyz_ts), (emg, emg_ts)):
        if len(values) != len(stamps):
            raise ValueError('payload and timestamp lengths differ')
        nearest(stamps, stamps[:1])
    for values in (hand, xyz, emg):
        if not np.all(np.isfinite(values)):
            raise ValueError('nonfinite sensor data')
    start = max(int(ts[0]) for ts in (video_ts, hand_ts, xyz_ts, emg_ts))
    end = min(int(ts[-1]) for ts in (video_ts, hand_ts, xyz_ts, emg_ts))
    if end <= start:
        raise ValueError('no shared time interval')
    targets = start + np.rint(np.arange(int((end-start)/1e9*fps)+1)*1e9/fps).astype(np.int64)
    vi, hi, xi = [nearest(ts, targets) for ts in (video_ts, hand_ts, xyz_ts)]
    hand_scale = 220 / max(float(np.max(np.abs(hand-hand[:, :1]))), 0.01)
    center = (xyz.min(axis=0)+xyz.max(axis=0))/2
    scale = 320 / max(float(np.ptp(xyz, axis=0).max()), 0.01)
    colors = [(80,200,255),(100,220,120),(240,170,90),(200,120,220)]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1600,1000))
    if not writer.isOpened():
        raise RuntimeError('cannot open video writer')

    def text(image, label, pos, size=.6):
        cv2.putText(image,label,pos,cv2.FONT_HERSHEY_SIMPLEX,size,(225,230,235),1,cv2.LINE_AA)

    try:
        for f,t in enumerate(targets):
            canvas = np.full((1000,1600,3),24,np.uint8)
            text(canvas,f't={(int(t)-start)/1e9:.3f}s | recorded timestamps | no latency correction',(20,28))
            image = rgb[int(vi[f])][..., ::-1]
            h,w = image.shape[:2]
            ratio = min(780/w,450/h)
            image = cv2.resize(image,(round(w*ratio),round(h*ratio)))
            h,w = image.shape[:2]
            canvas[70:70+h,10+(780-w)//2:10+(780-w)//2+w] = image
            text(canvas,f'Gemini frame {vi[f]}  dt={(int(video_ts[vi[f]])-int(t))/1e6:+.1f}ms',(20,57))
            for dims,box,label in [((0,1),(800,70,380,450),'Hand local XY (m)'),((0,2),(1200,70,380,450),'Hand local XZ (m)')]:
                _draw_hand(canvas,_project(hand[hi[f]],dims,box,hand_scale),label,(box[0]+10,57))
            for dims,origin,label in [((0,1),(200,765),'Wrist tracker XY (m)'),((0,2),(590,765),'Wrist tracker XZ (m)')]:
                points=(xyz-center)[:,dims]*scale
                points[:,1]*=-1
                points=np.rint(points+origin).astype(np.int32)
                cv2.polylines(canvas,[points.reshape(-1,1,2)],False,(65,65,65),1)
                cv2.polylines(canvas,[points[:xi[f]+1].reshape(-1,1,2)],False,(90,205,130),2)
                cv2.circle(canvas,tuple(points[xi[f]]),6,(80,200,255),-1)
                text(canvas,label,(origin[0]-175,565))
            text(canvas,'Native EMG: 8 channels | trailing 2s | signed 24-bit microvolts',(810,550),.52)
            left=np.searchsorted(emg_ts,int(t)-2_000_000_000)
            right=np.searchsorted(emg_ts,t,side='right')
            times=emg_ts[left:right]
            for ch in range(8):
                baseline=590+ch*52
                cv2.line(canvas,(850,baseline),(1580,baseline),(60,60,60),1)
                text(canvas,f'{ch+1}',(815,baseline+5))
                px=850+((times-(int(t)-2_000_000_000))/2e9*730)
                trace=emg[left:right,ch].astype(float)
                scale=max(float(np.percentile(np.abs(trace),95)),1.0)
                py=baseline-trace/scale*22
                pts=np.rint(np.column_stack((px,py))).astype(np.int32)
                if len(pts)>1:
                    cv2.polylines(canvas,[pts.reshape(-1,1,2)],False,colors[ch%4],1,cv2.LINE_AA)
            text(canvas,f'XYZ: {xyz[xi[f],0]:+.3f}, {xyz[xi[f],1]:+.3f}, {xyz[xi[f],2]:+.3f} m',(20,978))
            writer.write(canvas)
            if f%300==0:
                print(f'{f}/{len(targets)}',flush=True)
    finally:
        writer.release()
    print(f'{output}: {len(targets)} frames, {fps} fps',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path,nargs='?',help='episode .zarr directory; defaults to the newest episode in the current directory')
    parser.add_argument('--output',type=Path,help='output MP4; defaults to render_<episode>.mp4')
    parser.add_argument('--fps',type=float,default=30)
    args=parser.parse_args()
    if args.episode is None:
        episodes = sorted(Path.cwd().glob('episode_*.zarr'))
        if not episodes:
            parser.error('no episode_*.zarr found in the current directory')
        args.episode = max(episodes, key=lambda p: (p.stat().st_mtime, p.name))
    if not args.episode.is_dir():
        parser.error(f'episode does not exist or is not a directory: {args.episode}')
    if args.output is None:
        args.output = Path.cwd() / f'render_{args.episode.stem}.mp4'
        index = 1
        while args.output.exists():
            args.output = Path.cwd() / f'render_{args.episode.stem}_{index}.mp4'
            index += 1
    elif args.output.exists():
        parser.error(f'output already exists: {args.output}; choose a new filename with --output')
    print(f'episode: {args.episode}\noutput: {args.output}', flush=True)
    render(args.episode,args.output,args.fps)
