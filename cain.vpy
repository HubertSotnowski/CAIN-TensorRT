import vapoursynth as vs
import sys
import os
import os.path
core = vs.core
core.num_threads = 8
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")
clip =  core.lsmas.LWLibavSource(r'testsrc.mp4')
clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip_pos1 = clip[1:]
clip_pos2 = clip.std.Trim(first=1)
print(clip_pos2.frames)
clipstack =  [clip_pos1,clip_pos2]
output = core.trt.Model(clipstack, engine_path=f"2.engine",num_streams=8)
output=core.std.Interleave([output,clip])
output = vs.core.resize.Bicubic(output, format=vs.YUV420P8, matrix_s="709")
output.set_output()