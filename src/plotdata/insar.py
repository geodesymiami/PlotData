import os
import glob

def generate_view_ifgram_cmd(work_dir, date12, inps):
    ifgram_file = work_dir + '/mintpy/geo/geo_ifgramStack.h5'
    geom_file = work_dir + '/mintpy/geo/geo_geometryRadar.h5'
    mask_file = work_dir + '/mintpy/geo/geo_maskTempCoh.h5'   # generated with generate_mask.py geo_geometryRadar.h5 height -m 3.5 -o waterMask.h5 option
    
    ## Configuration for InSAR background: check view.py -h for more plotting options.
    cmd = 'view.py {} unwrapPhase-{} -m {} -d {} '.format(ifgram_file, date12, mask_file, geom_file)
    if inps.plot_box:
        cmd += f"--sub-lat {inps.plot_box[0]} {inps.plot_box[1]} --sub-lon {inps.plot_box[2]} {inps.plot_box[3]} "
    cmd += '--notitle -u cm -c jet_r --nocbar --noverbose '
    #print(cmd)
    return cmd

def generate_view_velocity_cmd(vel_file,  inps):
    cmd = 'view.py {} velocity '.format(vel_file)
    if inps.plot_box:
        cmd += f" --sub-lat {inps.plot_box[0]} {inps.plot_box[1]} --sub-lon {inps.plot_box[2]} {inps.plot_box[3]} "
    cmd += f"--notitle -u {inps.unit} --fontsize {inps.font_size} -c jet --noverbose" 
    if inps.vlim:
        cmd += f" --vlim {inps.vlim[0]} {inps.vlim[1]}"
        
    # print(cmd)
    return cmd
    
    