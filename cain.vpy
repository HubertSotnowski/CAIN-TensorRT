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
clip_pos2 = clip
clipstack = core.std.StackHorizontal([clip_pos1, clip_pos2])
clipstack2=clipstack
output = core.trt.Model([clipstack], engine_path=f"test.engine",num_streams=8)
output=core.std.Interleave([output,clipstack])
output=core.std.Crop(output,right=clip.width)
output = vs.core.resize.Bicubic(output, format=vs.YUV420P8, matrix_s="709")
output.set_output()