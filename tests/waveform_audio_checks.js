const assert=require('node:assert/strict');
const base={time_s:[0,.25,1],values:[2,-2,999],mean:-1,peak:3,mode:'continuous',duration_s:1800,phase_s:0};
assert.equal(audioSourceValue(base,.249),2);
assert.equal(audioSourceValue(base,.25),-2);
assert.equal(audioSourceValue(base,1000),2);
assert.equal(audioSourceValue({...base,phase_s:.5},0),-2);
const shot={...base,mode:'once_per_step',steps:[{duration_s:2,active:true},{duration_s:1,active:false}],loop:true};
assert.equal(audioSourceValue(shot,0),2);
assert.equal(audioSourceValue(shot,1.2),0);
assert.equal(audioSourceValue(shot,2.1),0);
assert.equal(audioSourceValue(shot,3.1),2);
assert.equal(audioSourceValue({...shot,loop:false},3.1),0);
assert.equal(audioSourceValue({...base,mode:'finite'},1),0);
let timers=new Set(),sequence=0,errors=[];
global.setInterval=()=>{const id=++sequence;timers.add(id);return id;};
global.clearInterval=id=>timers.delete(id);
global.setTimeout=fn=>{fn();return 0;};
class Context{
 constructor(){this.currentTime=0;this.sampleRate=48000;this.state='running';this.buffers=[];this.sources=[];}
 createGain(){return {gain:{cancelScheduledValues(){},setTargetAtTime(){},setValueAtTime(){},linearRampToValueAtTime(){}},connect(){},disconnect(){}};}
 createBiquadFilter(){return {frequency:{value:0},Q:{value:0},connect(){},disconnect(){}};}
 createBuffer(_,n){const data=new Float32Array(n);this.buffers.push(data);return {getChannelData:()=>data};}
 createBufferSource(){const node={connect(){},disconnect(){},start(t){this.when=t;},stop(){this.stopped=true;}};this.sources.push(node);return node;}
 async resume(){this.state='running';}
 async close(){this.state='closed';}
}
global.window={AudioContext:Context};
(async()=>{
 let running=false,speed=1,position=0;
 const clock=()=>({running,sourceTime:ahead=>position+ahead*speed});
 const monitor=new SourceAudioMonitor(clock,e=>errors.push(e));
 assert.equal(monitor.source,null);assert.equal(monitor.context,null);
 await monitor.select(base);assert.equal(timers.size,0);assert.equal(monitor.nodes.size,0);
 running=true;await monitor.restart(true);
 assert.equal(timers.size,1);assert(monitor.nodes.size<=5);
 assert(monitor.context.buffers.every(b=>b.length===1200));
 const old=[...monitor.nodes];await monitor.select({...base,phase_s:.5});
 assert(old.every(n=>n.stopped));assert.equal(timers.size,1);
 monitor.stop();assert.equal(timers.size,0);assert.equal(monitor.nodes.size,0);
 position=.4;speed=.1;await monitor.restart(true);assert.equal(timers.size,1);
 running=false;monitor.pump();assert.equal(timers.size,0);
 running=true;await monitor.select({...base,mean:2,peak:0,values:[2,2,2]});
 assert(monitor.context.buffers.slice(-4).every(b=>b.every(v=>v===0)));
 // Verify physical pitch and slow-motion pitch from generated PCM, not plotted points.
 const sine={time_s:[],values:[],mean:0,peak:1,mode:'continuous',duration_s:1800};
 for(let i=0;i<=200;i++){sine.time_s.push(i/200000);sine.values.push(Math.sin(2*Math.PI*i/200));}
 const crossings=()=>{const data=monitor.context.buffers.slice(-4).flatMap(b=>Array.from(b));let n=0;
   for(let i=1;i<data.length;i++)if(data[i-1]<0&&data[i]>=0)n++;return n;};
 position=0;speed=1;await monitor.select(sine);assert(Math.abs(crossings()-100)<=1);
 speed=.1;await monitor.restart(true);assert(Math.abs(crossings()-10)<=1);
 const ctx=monitor.context;monitor.dispose();assert.equal(ctx.state,'closed');assert.equal(timers.size,0);
 assert.deepEqual(errors,[]);
 // A rejected browser activation never leaves a checked or running monitor.
 class Broken extends Context{constructor(){super();this.state='suspended';}async resume(){throw Error('blocked');}}
 window.AudioContext=Broken;const failed=new SourceAudioMonitor(clock,e=>errors.push(e));
 await failed.select(base);assert.equal(failed.source,null);assert.equal(timers.size,0);assert.equal(errors.length,1);
 // A pending activation cannot resurrect a deselected source.
 let resume;class Delayed extends Context{constructor(){super();this.state='suspended';}resume(){return new Promise(r=>resume=()=>{this.state='running';r();});}}
 window.AudioContext=Delayed;const delayed=new SourceAudioMonitor(clock,e=>errors.push(e));
 const pending=delayed.select(base);await delayed.select(null);resume();await pending;
 assert.equal(delayed.nodes.size,0);assert.equal(timers.size,0);delayed.dispose();
 console.log('Audio source and lifecycle checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
