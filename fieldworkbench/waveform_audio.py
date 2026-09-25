"""Compact authored-source audio monitor (independent of display decimation).

This listens to the source command, before electrical slew/load response. It is
not a pickup-coil simulation or calibrated acoustic output. No PCM is persisted.
"""
from __future__ import annotations

import numpy as np


def source_audio_payload(times, values, *, source=None, protocol=None):
    """Retain the authored DAC clock once, never expand an exposure into PCM."""
    times = np.asarray(times, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if (len(times) < 2 or len(times) != len(values)
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Audio source requires finite increasing time/value samples.")
    duration = float(times[-1] - times[0])
    mean = float(np.dot(np.diff(times) / duration, values[:-1]))
    low, high = float(np.min(values[:-1])), float(np.max(values[:-1]))
    result = {"version": 1, "time_s": times.tolist(), "values": values.tolist(),
              "mean": mean, "peak": (0.0 if low == high else max(abs(low - mean), abs(high - mean))),
              "mode": "finite"}
    if source is not None and protocol is not None:
        mode = str(source.get("playback_mode", "continuous"))
        result.update(mode=mode, phase_s=float(source.get("phase_offset_s", 0.0)),
                      duration_s=float(protocol["duration_s"]))
        if mode == "once_per_step":
            steps = protocol["sequence"]["steps"]
            result["steps"] = [{"duration_s": float(step["duration_s"]),
                                "active": any(str(g.get("source_id")) == str(source["id"])
                                              for g in step.get("drive_groups", []))}
                               for step in steps]
            result["loop"] = bool(protocol["sequence"].get("loop", False))
            # Inactive routing and the remainder of a one-shot command are zero.
            cycle = sum(step["duration_s"] for step in result["steps"])
            duty = sum(min(duration, step["duration_s"]) for step in result["steps"]
                       if step["active"]) / cycle if cycle else 0.0
            result["mean"] = mean * duty
            result["peak"] = max(abs(min(0.0, low) - result["mean"]),
                                 abs(max(0.0, high) - result["mean"]))
    return result


# Embedded in the viewer, so frozen/packaged builds need no new asset loader.
AUDIO_MONITOR_JS = r'''
function audioSourceValue(source, time){
  const ts=source.time_s,vs=source.values,start=ts[0],period=ts[ts.length-1]-start;
  let local=time-start;
  if(source.mode!=='finite'){
    if(time<0||time>=source.duration_s)return 0;
    local=time;
    if(source.mode==='once_per_step'){
      const steps=source.steps||[],cycle=steps.reduce((s,x)=>s+x.duration_s,0);
      if(!(cycle>0))return 0;
      if(source.loop)local=((local%cycle)+cycle)%cycle;
      else if(local>=cycle)return 0;
      let active=false;
      for(const step of steps){if(local<step.duration_s){active=step.active;break;}local-=step.duration_s;}
      if(!active||local>=period)return 0;
    }
    local=((local+(source.phase_s||0))%period+period)%period;
  }else if(local<0||local>=period)return 0;
  const t=start+local;
  let lo=0,hi=ts.length-1;
  while(lo+1<hi){const mid=(lo+hi)>>1;if(ts[mid]<=t)lo=mid;else hi=mid;}
  return Number(vs[lo])||0;
}

// A bounded look-ahead queue uses the audio clock; no full-exposure PCM arrays.
class SourceAudioMonitor{
  constructor(clock,onError){this.clock=clock;this.onError=onError;this.context=null;
    this.source=null;this.nodes=new Set();this.timer=null;this.generation=0;}
  async select(source){this.source=source;await this.restart(true);}
  stop(){
    this.generation++;if(this.timer!==null)clearInterval(this.timer);this.timer=null;
    const ctx=this.context,now=ctx?ctx.currentTime:0;
    if(this.gain){this.gain.gain.cancelScheduledValues(now);this.gain.gain.setTargetAtTime(0,now,.003);}
    for(const node of this.nodes){try{node.stop(now+.015);}catch(_){} }
    this.nodes.clear();
    const oldGain=this.gain,oldFilter=this.filter;
    if(oldGain)setTimeout(()=>{oldGain.disconnect();if(oldFilter)oldFilter.disconnect();},30);
    this.gain=null;this.filter=null;
  }
  async restart(gesture=false){
    this.stop();const token=this.generation;
    if(!this.source)return;
    try{
      if(!this.context){
        if(!gesture)return;
        const AudioCtor=window.AudioContext||window.webkitAudioContext;
        if(!AudioCtor)throw Error('Web Audio is unavailable in this viewer.');
        this.context=new AudioCtor({sampleRate:48000,latencyHint:'interactive'});
      }
      const ctx=this.context;
      if(ctx.state==='suspended'&&gesture)await ctx.resume();
      if(token!==this.generation||!this.source||!this.clock().running)return;
      if(ctx.state!=='running')throw Error('Audio could not start. Try selecting Audio again.');
      this.gain=ctx.createGain();this.filter=ctx.createBiquadFilter();
      this.filter.type='highpass';this.filter.frequency.value=5;this.filter.Q.value=.707;
      this.filter.connect(this.gain);this.gain.connect(ctx.destination);
      this.gain.gain.setValueAtTime(0,ctx.currentTime);
      this.gain.gain.linearRampToValueAtTime(.12,ctx.currentTime+.015);
      this.next=ctx.currentTime+.01;
      const pump=()=>{if(token!==this.generation)return;
        try{this.pump();}catch(error){this.fail(error);}};
      pump();if(token===this.generation)this.timer=setInterval(pump,20);
    }catch(error){if(token===this.generation)this.fail(error);}
  }
  fail(error){this.stop();this.source=null;this.onError(String(error.message||error));}
  pump(){
    const ctx=this.context,state=this.clock(),audioNow=ctx.currentTime;
    if(!state.running){this.stop();return;}
    if(this.next<ctx.currentTime)this.next=ctx.currentTime+.005;
    const source=this.source,rate=ctx.sampleRate,count=Math.round(rate*.025);
    while(this.next<ctx.currentTime+.09){
      const buffer=ctx.createBuffer(1,count,rate),data=buffer.getChannelData(0);
      for(let i=0;i<count;i++){
        // Four sub-samples per output sample soften aliasing at DAC edges.
        let value=0;
        for(let sub=0;sub<4;sub++){
          const ahead=this.next-audioNow+(i+(sub+.5)/4)/rate;
          const t=state.sourceTime(ahead);
          value+=t===null?0:(audioSourceValue(source,t)-source.mean);
        }
        data[i]=source.peak>0?Math.max(-1,Math.min(1,value/(4*source.peak))):0;
      }
      const node=ctx.createBufferSource();node.buffer=buffer;node.connect(this.filter);
      this.nodes.add(node);node.onended=()=>{this.nodes.delete(node);node.disconnect();};
      node.start(this.next);this.next+=count/rate;
    }
  }
  dispose(){this.stop();this.source=null;if(this.context){this.context.close().catch(()=>{});this.context=null;}}
}
'''
