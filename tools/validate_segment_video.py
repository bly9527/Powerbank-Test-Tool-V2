"""Replay real segment ROIs without touching application state or user data.
Outputs raw ROI PNGs (lossless cache), per-frame JSONL and tracker summaries.
Example: python tools/validate_segment_video.py --end 13000 --stop-at-full
"""
import argparse,collections,hashlib,json,os,sys,time,subprocess,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from concurrent.futures import ThreadPoolExecutor
from core.segment_dataset import SegmentMatcher,read_frame_at,crop_digits,read_image,write_image
from core.soc_tracker import SegmentSOCTracker
MATCHER_HASH=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("core/segment_dataset.py").read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,default=13000);p.add_argument('--step',type=float,default=1);p.add_argument('--old',action='store_true');p.add_argument('--stop-at-full',action='store_true');p.add_argument('--name',default='full');p.add_argument('--source',default=r'D:\PB313\5V充电 20261012.mp4');p.add_argument('--library',default=str(Path(os.environ['LOCALAPPDATA'])/'PowerbankTestTool/segment_samples/template_5ceada37d2f0469fb3cdadc3f7f4009d'));p.add_argument('--rois',default='[[949,197,73,47],[1287,252,76,49]]');p.add_argument('--probe',action='store_true');a=p.parse_args()
 out=Path('测试结果/segment-real-regression');out.mkdir(parents=True,exist_ok=True);rois=json.loads(a.rois);identity={'source':str(Path(a.source).resolve()),'size':Path(a.source).stat().st_size,'mtime_ns':Path(a.source).stat().st_mtime_ns,'rois':rois};cache=out/('cache-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]);cache.mkdir(exist_ok=True);(cache/'identity.json').write_text(json.dumps(identity),encoding='utf8')
 root=Path(a.library)
 source=a.source;cap=cv2.VideoCapture(source);cv2.setNumThreads(1)
 models={'new':SegmentMatcher}
 if a.old:
  module=types.ModuleType('core._previous_segment');module.__package__='core';exec(subprocess.check_output(['git','show','HEAD:core/segment_dataset.py']).decode('utf8'),module.__dict__);models['old']=module.SegmentMatcher
 records={};pending={};reasons={};start=time.perf_counter();both_full=None;legacy_index_samples=0
 with ThreadPoolExecutor(10) as pool, (out/(a.name+'.jsonl')).open('w',encoding='utf8') as log:
  matchers={name:cls(root,executor=pool,workers=10) for name,cls in models.items()};trackers={(name,u):SegmentSOCTracker('charge',2,a.step) for name in models for u in (1,2)}
  for key in trackers: records[str(key)]=[];reasons[str(key)]=collections.Counter()
  second=a.start;i=0
  while second<=a.end:
   crops=[];paths=[cache/f'{second:010.3f}-u{u}.png' for u in (1,2)]
   stamp=cache/f'{second:010.3f}.json'
   if all(x.exists() for x in paths) and stamp.exists():
    crops=[read_image(x) for x in paths];timing=json.loads(stamp.read_text());actual=timing['actual'];frame_index=timing['frame_index']
   else:
    ok,frame=read_frame_at(cap,second)
    if not ok:break
    actual=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    frame_index=max(0,int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1)
    stamp.write_text(json.dumps({'requested':second,'actual':actual,'frame_index':frame_index}),encoding='utf8')
    for roi,path in zip(rois,paths):
     crop=crop_digits(frame,roi);write_image(path,crop);crops.append(crop)
   if frame_index is None:legacy_index_samples+=1
   for name,matcher in matchers.items():
    for u,crop in enumerate(crops,1):
     value,score=matcher.extract_soc(crop);key=(name,u);tracker=trackers[key];event=tracker.update(value,actual,lambda: {'crop':str(paths[u-1]),'requested_sec':second,'frame_index':frame_index})
     reason=matcher.last_reason;reasons[str(key)][reason.split('：')[0] if value is None else 'matched']+=1
     row={'requested':second,'actual':actual,'unit':u,'model':name,'value':value,'score':score,'reason':reason,'confirmed':tracker.confirmed,'rejection':tracker.last_rejection}
     if event:
      record={'value':event.value,'second':event.first_sec,'payload':event.payload};records[str(key)].append(record);row['event']=record
     log.write(json.dumps(row,ensure_ascii=False)+'\n')
   if a.stop_at_full and all(trackers['new',u].confirmed==100 for u in (1,2)):
    if both_full is None:both_full=second
    elif second-both_full>=30:break
   if i%100==0:
    log.flush();print(json.dumps({'at':second,'elapsed':round(time.perf_counter()-start,1),'last':{str(k):v.confirmed for k,v in trackers.items()} }),flush=True)
   second+=min(a.step/2,.5) if a.probe and any(trackers['new',u].needs_probe for u in (1,2)) else a.step;i+=1
  for key,tracker in trackers.items():pending[str(key)]=tracker.pending
 targets=list(range(1,11))+list(range(20,91,10))+list(range(91,101))
 coverage={key:{'captured':sorted(set(x['value'] for x in values)&set(targets)),'missing':sorted(set(targets)-set(x['value'] for x in values))} for key,values in records.items()}
 metadata={'legacy_cache_samples_without_frame_index':legacy_index_samples,'coverage':coverage,'source':source,'roi':rois,'video_identity':identity,'probe':a.probe,'step':a.step,'start':a.start,'end_requested':a.end,'end_actual':second,'seconds_elapsed':time.perf_counter()-start,'records':records,'pending':pending,'reasons':reasons,'matcher_sha256':MATCHER_HASH,'limitations':'Stored capture ROIs; no OCR; tracker-only without worker historical correction; no manual ground truth for all frames.'}
 (out/(a.name+'-summary.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8');cap.release();print('DONE '+a.name,flush=True)
if __name__=='__main__':main()
