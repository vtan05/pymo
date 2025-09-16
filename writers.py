import numpy as np
import pandas as pd

class BVHWriter():
    def __init__(self):
        pass
    
    def write(self, X, ofile, framerate=-1, start=0, stop=-1):
        
        # Writing the skeleton info
        ofile.write('HIERARCHY\n')
        
        self.motions_ = []
        self._printJoint(X, X.root_name, 0, ofile)

        if stop > 0:
            nframes = stop-start
        else:
            nframes = X.values.shape[0]
            stop = X.values.shape[0]

        # Writing the motion header
        ofile.write('MOTION\n')
        ofile.write('Frames: %d\n'%nframes)
        
        if framerate > 0:
            ofile.write('Frame Time: %f\n'%float(1.0/framerate))
        else:
            ofile.write('Frame Time: %f\n'%X.framerate)

        # Writing the data
        self.motions_ = np.asarray(self.motions_).T
        lines = [" ".join(item) for item in self.motions_[start:stop].astype(str)]
        ofile.write("".join("%s\n"%l for l in lines))

    def _printJoint(self, X, joint, tab, ofile):
        
        if X.skeleton[joint]['parent'] == None:
            ofile.write('ROOT %s\n'%joint)
        elif len(X.skeleton[joint]['children']) > 0:
            ofile.write('%sJOINT %s\n'%('\t'*(tab), joint))
        else:
            ofile.write('%sEnd site\n'%('\t'*(tab)))

        ofile.write('%s{\n'%('\t'*(tab)))
        
        ofile.write('%sOFFSET %3.5f %3.5f %3.5f\n'%('\t'*(tab+1),
                                                X.skeleton[joint]['offsets'][0],
                                                X.skeleton[joint]['offsets'][1],
                                                X.skeleton[joint]['offsets'][2]))
        rot_order = X.skeleton[joint]['order']
        
        #print("rot_order = " + rot_order)
        channels = X.skeleton[joint]['channels']
        rot = [c for c in channels if ('rotation' in c)]
        # pos = [c for c in channels if ('position' in c)]
        # Only allow position channels on the root
        pos = [c for c in channels if 'position' in c] if joint == X.root_name else []
        
        n_channels = len(rot) +len(pos)
        ch_str = ''
        if n_channels > 0:
            for ci in range(len(pos)):
                cn = pos[ci]
                self.motions_.append(np.asarray(X.values['%s_%s'%(joint,cn)].values))
                ch_str = ch_str + ' ' + cn 
            for ci in range(len(rot)):
                cn = '%srotation'%(rot_order[ci])
                self.motions_.append(np.asarray(X.values['%s_%s'%(joint,cn)].values))
                ch_str = ch_str + ' ' + cn 
        if len(X.skeleton[joint]['children']) > 0:
            #ch_str = ''.join(' %s'*n_channels%tuple(channels))
            ofile.write('%sCHANNELS %d%s\n' %('\t'*(tab+1), n_channels, ch_str)) 

            for c in X.skeleton[joint]['children']:
                self._printJoint(X, c, tab+1, ofile)

        ofile.write('%s}\n'%('\t'*(tab)))

    # def _printJoint(self, X, joint, tab, ofile):
    #     # Node header
    #     if X.skeleton[joint]['parent'] is None:
    #         ofile.write(f'ROOT {joint}\n')
    #     elif len(X.skeleton[joint]['children']) > 0:
    #         ofile.write(f'{"\t"*tab}JOINT {joint}\n')
    #     else:
    #         ofile.write(f'{"\t"*tab}End Site\n')

    #     ofile.write(f'{"\t"*tab}' + '{\n')
    #     ox, oy, oz = X.skeleton[joint]['offsets']
    #     ofile.write(f'{"\t"*(tab+1)}OFFSET {ox:.5f} {oy:.5f} {oz:.5f}\n')

    #     rot_order = X.skeleton[joint]['order']          # e.g., "XYZ"
    #     channels   = X.skeleton[joint]['channels']      # e.g., ['Xposition','Yposition','Zposition','Zrotation','Xrotation','Yrotation']

    #     # Only allow position channels on the root
    #     pos = [c for c in channels if 'position' in c] if joint == X.root_name else []
    #     # Rotation channels always allowed
    #     rot = [c for c in channels if 'rotation' in c]

    #     ch_names_written = []

    #     # Add position channels that actually exist in X.values
    #     for cn in pos:
    #         col = f'{joint}_{cn}'
    #         if col in X.values.columns:
    #             self.motions_.append(np.asarray(X.values[col].values))
    #             ch_names_written.append(cn)

    #     # Add rotation channels following rot_order, but only if they exist
    #     for axis in rot_order:
    #         cn = f'{axis}rotation'
    #         col = f'{joint}_{cn}'
    #         if cn in rot and col in X.values.columns:
    #             self.motions_.append(np.asarray(X.values[col].values))
    #             ch_names_written.append(cn)

    #     # Write CHANNELS only for non-End Site nodes
    #     if len(X.skeleton[joint]['children']) > 0 or X.skeleton[joint]['parent'] is None:
    #         ofile.write(f'{"\t"*(tab+1)}CHANNELS {len(ch_names_written)}' + ''.join(f' {c}' for c in ch_names_written) + '\n')
    #         # Recurse into children
    #         for c in X.skeleton[joint]['children']:
    #             self._printJoint(X, c, tab+1, ofile)

    #     ofile.write(f'{"\t"*tab}' + '}\n')
