import numpy as np
import pandas as pd

class BVHWriter:
    def __init__(self, strict=False, warn_missing=True):
        """
        strict: if True, raise if a required column is missing.
        warn_missing: if True, print a summary of zero-filled columns after writing.
        """
        self.strict = strict
        self.warn_missing = warn_missing

    def write(self, X, ofile, framerate=-1, start=0, stop=-1):
        """X: MocapData-like with .values (DataFrame), .skeleton, .root_name, .framerate"""
        self._missing_notes = []
        self.motions_ = []

        # --- Hierarchy ---
        ofile.write('HIERARCHY\n')
        self._printJoint(X, X.root_name, 0, ofile)

        # --- Motion header ---
        if stop > 0:
            nframes = stop - start
        else:
            nframes = X.values.shape[0]
            stop = X.values.shape[0]

        ofile.write('MOTION\n')
        ofile.write('Frames: %d\n' % nframes)
        if framerate > 0:
            ofile.write('Frame Time: %0.6f\n' % (1.0 / float(framerate)))
        else:
            ofile.write('Frame Time: %0.6f\n' % float(X.framerate))

        # --- Data ---
        data = np.asarray(self.motions_).T  # (T, nchannels)
        lines = [" ".join(item) for item in data[start:stop].astype(str)]
        ofile.write("".join("%s\n" % l for l in lines))

        if self.warn_missing and self._missing_notes:
            print("[BVHWriter] Missing columns were filled with zeros:")
            for note in self._missing_notes:
                print("  -", note)

    def _printJoint(self, X, joint, tab, ofile):
        sk = X.skeleton
        is_root = (sk[joint]['parent'] is None)
        is_end  = (len(sk[joint]['children']) == 0)

        indent = '\t' * tab

        # Node header
        if is_root:
            ofile.write('ROOT %s\n' % joint)
        elif is_end:
            ofile.write('%sEnd Site\n' % indent)
        else:
            ofile.write('%sJOINT %s\n' % (indent, joint))

        ofile.write('%s{\n' % indent)

        # OFFSET
        off = sk[joint]['offsets']
        ofile.write('%sOFFSET %0.5f %0.5f %0.5f\n' % ('\t'*(tab+1), off[0], off[1], off[2]))

        # End Site: close and return (no channels)
        if is_end:
            ofile.write('%s}\n' % indent)
            return

        # CHANNELS: root has XYZ position + 3 rotations; others have 3 rotations only
        order = sk[joint].get('order', 'XYZ')
        rot_names = ['%srotation' % order[i] for i in range(3)]
        pos_names = ['Xposition', 'Yposition', 'Zposition'] if is_root else []
        channels = pos_names + rot_names

        ch_str = ''.join(' %s' % c for c in channels)
        ofile.write('%sCHANNELS %d%s\n' % ('\t'*(tab+1), len(channels), ch_str))

        # Append motion arrays, zero-filling if missing
        T = len(X.values.index)
        for cname in pos_names:
            self._append_channel(X, joint, '%s_%s' % (joint, cname), T)
        for cname in rot_names:
            self._append_channel(X, joint, '%s_%s' % (joint, cname), T)

        # Recurse
        for child in sk[joint]['children']:
            self._printJoint(X, child, tab+1, ofile)

        ofile.write('%s}\n' % indent)

    def _append_channel(self, X, joint, col, T):
        if col in X.values.columns:
            self.motions_.append(np.asarray(X.values[col].values))
        else:
            if self.strict:
                raise KeyError("Required BVH column missing: %s" % col)
            self.motions_.append(np.zeros(T))
            self._missing_notes.append("%s: %s" % (joint, col))
