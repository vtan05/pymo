'''
Preprocessing Tranformers Based on sci-kit's API

By Omid Alemi
Created on June 12, 2017
'''
import copy
import pandas as pd
import numpy as np
import transforms3d as t3d
import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R

from scipy import signal, interpolate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from pymo.rotation_tools import Rotation, euler2expmap, euler2expmap2, expmap2euler, euler_reorder, unroll, euler2vectors, vectors2euler
from pymo.Quaternions import Quaternions
from pymo.Pivots import Pivots

class MocapParameterizer(BaseEstimator, TransformerMixin):
    def __init__(self, param_type = 'euler', ref_pose=None):
        '''
        param_type = {'euler', 'quat', 'expmap', 'position', 'expmap2pos'}
        '''
        self.param_type = param_type
        if (ref_pose is not None):
            self.ref_pose = self._to_quat(ref_pose)[0]
        else:
            self.ref_pose = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            if self.ref_pose is None:
                return self._to_expmap(X)
            else:
                return self._to_expmap2(X)
        elif self.param_type == 'vectors':
            return self._euler_to_vectors(X)
        elif self.param_type == 'quat':
            return self._to_quat(X)
        elif self.param_type == 'position':
            return self._to_pos(X)
        elif self.param_type == 'expmap2pos':
            return self._expmap_to_pos(X)
        else:
            raise 'param types: euler, quat, expmap, position, expmap2pos'
    
    def inverse_transform(self, X, copy=None): 
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            if self.ref_pose is None:
                return self._expmap_to_euler(X)
            else:
                return self._expmap_to_euler2(X)                
        elif self.param_type == 'vectors':
            return self._vectors_to_euler(X)
        elif self.param_type == 'quat':
            return self._quat_to_euler(X)
        elif self.param_type == 'position':
            print('positions 2 eulers is not supported')
            return X
        else:
            raise 'param types: euler, quat, expmap, position'

    def _to_quat(self, X):
        '''Converts joints rotations in quaternions'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            quat_df = euler_df.copy()

            # List the columns that contain rotation channels
            rot_cols = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                rot_order = track.skeleton[joint]['order']

                # Get the rotation columns that belong to this joint
                rc = euler_df[[c for c in rot_cols if joint in c]]

                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                # Make sure the columns are organized in xyz order
                if rc.shape[1] < 3:
                    euler_values = np.zeros((euler_df.shape[0], 3))
                    rot_order = "XYZ"
                else:
                    euler_values = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                
                quat_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                quats = Quaternions.from_euler(np.asarray(euler_values), order=rot_order.lower(), world=False)

                # Create the corresponding columns in the new DataFrame
                quat_df['%s_qWrotation'%joint] = pd.Series(data=[e[0] for e in quats], index=quat_df.index)
                quat_df['%s_qXrotation'%joint] = pd.Series(data=[e[1] for e in quats], index=quat_df.index)
                quat_df['%s_qYrotation'%joint] = pd.Series(data=[e[2] for e in quats], index=quat_df.index)
                quat_df['%s_qZrotation'%joint] = pd.Series(data=[e[3] for e in quats], index=quat_df.index)

            new_track = track.clone()
            new_track.values = quat_df
            Q.append(new_track)
        return Q
    
    def _quat_to_euler(self, X):
        Q = []
        for track in X:
            channels = []
            titles = []
            quat_df = track.values

            # Create a new DataFrame to store the exponential map rep
            #euler_df = pd.DataFrame(index=exp_df.index)
            euler_df = quat_df.copy()

            # List the columns that contain rotation channels
            quat_params = [c for c in quat_df.columns if ( any(p in c for p in ['qWrotation','qXrotation','qYrotation','qZrotation']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = quat_df[[c for c in quat_params if joint in c]] # Get the columns that belong to this joint
                
                euler_df.drop(['%s_qWrotation'%joint, '%s_qXrotation'%joint, '%s_qYrotation'%joint, '%s_qZrotation'%joint], axis=1, inplace=True)
                quat = [[f[1]['%s_qWrotation'%joint], f[1]['%s_qXrotation'%joint], f[1]['%s_qYrotation'%joint], f[1]['%s_qZrotation'%joint]] for f in r.iterrows()] # Make sure the columsn are organized in xyz order
                quats=Quaternions(np.asarray(quat))
                euler_rots = 180/np.pi*quats.euler()
                track.skeleton[joint]['order'] = 'ZYX'
                rot_order = track.skeleton[joint]['order']
                #euler_rots = [Rotation(f, 'expmap').to_euler(True, rot_order) for f in expmap] # Convert the exp maps to eulers
                #euler_rots = [expmap2euler(f, rot_order, True) for f in expmap] # Convert the exp maps to eulers
                                                  
                # Create the corresponding columns in the new DataFrame
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q
    
    def _to_pos(self, X):
        '''Converts joints rotations in Euler angles to joint positions'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            pos_df = pd.DataFrame(index=euler_df.index)

            # List the columns that contain rotation channels
            rot_cols = [c for c in euler_df.columns if ('rotation' in c)]

            # List the columns that contain position channels
            pos_cols = [c for c in euler_df.columns if ('position' in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton)
            
            tree_data = {}

            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']
                rot_order = track.skeleton[joint]['order']
                #print("rot_order:" + joint + " :" + rot_order)

                # Get the rotation columns that belong to this joint
                rc = euler_df[[c for c in rot_cols if joint in c]]

                # Get the position columns that belong to this joint
                pc = euler_df[[c for c in pos_cols if joint in c]]

                # Make sure the columns are organized in xyz order
                if rc.shape[1] < 3:
                    euler_values = np.zeros((euler_df.shape[0], 3))
                    rot_order = "XYZ"
                else:
                    euler_values = np.pi/180.0*np.transpose(np.array([track.values['%s_%srotation'%(joint, rot_order[0])], track.values['%s_%srotation'%(joint, rot_order[1])], track.values['%s_%srotation'%(joint, rot_order[2])]]))

                if pc.shape[1] < 3:
                    pos_values = np.asarray([[0,0,0] for f in pc.iterrows()])
                else:
                    pos_values =np.asarray([[f[1]['%s_Xposition'%joint], 
                                  f[1]['%s_Yposition'%joint], 
                                  f[1]['%s_Zposition'%joint]] for f in pc.iterrows()])
                
                quats = Quaternions.from_euler(np.asarray(euler_values), order=rot_order.lower(), world=False)
                
                tree_data[joint]=[
                                    [], # to store the rotation matrix
                                    []  # to store the calculated position
                                 ] 
                if track.root_name == joint:
                    tree_data[joint][0] = quats#rotmats
                    # tree_data[joint][1] = np.add(pos_values, track.skeleton[joint]['offsets'])
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent
                    tree_data[joint][0] = tree_data[parent][0]*quats# np.matmul(rotmats, tree_data[parent][0])

                    # add the position channel to the offset and store it in k, for every frame i
                    k = pos_values + np.asarray(track.skeleton[joint]['offsets'])

                    # multiply k to the rotmat of the parent for every frame i
                    q = tree_data[parent][0]*k #np.matmul(k.reshape(k.shape[0],1,3), tree_data[parent][0])

                    # add q to the position of the parent, for every frame i
                    tree_data[joint][1] = tree_data[parent][1] + q #q.reshape(k.shape[0],3) + tree_data[parent][1]

                # Create the corresponding columns in the new DataFrame
                df = pd.DataFrame(data=tree_data[joint][1], 
                                  index=pos_df.index, 
                                  columns=['%s_Xposition'%joint, '%s_Yposition'%joint, '%s_Zposition'%joint])
                pos_df = pd.concat((pos_df, df), axis=1)

            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)
        return Q

    def _expmap2rot(self, expmap):

        theta = np.linalg.norm(expmap, axis=1, keepdims=True)
        nz = np.nonzero(theta)[0]

        expmap[nz,:] = expmap[nz,:]/theta[nz]

        nrows=expmap.shape[0]
        x = expmap[:,0]
        y = expmap[:,1]
        z = expmap[:,2]

        s = np.sin(theta*0.5).reshape(nrows)
        c = np.cos(theta*0.5).reshape(nrows)

        rotmats = np.zeros((nrows, 3, 3))

        rotmats[:,0,0] = 2*(x*x-1)*s*s+1
        rotmats[:,0,1] = 2*x*y*s*s-2*z*c*s
        rotmats[:,0,2] = 2*x*z*s*s+2*y*c*s
        rotmats[:,1,0] = 2*x*y*s*s+2*z*c*s
        rotmats[:,1,1] = 2*(y*y-1)*s*s+1
        rotmats[:,1,2] = 2*y*z*s*s-2*x*c*s
        rotmats[:,2,0] = 2*x*z*s*s-2*y*c*s
        rotmats[:,2,1] =  2*y*z*s*s+2*x*c*s
        rotmats[:,2,2] =  2*(z*z-1)*s*s+1

        return rotmats

    def _expmap_to_pos(self, X):
        '''Converts joints rotations in expmap notation to joint positions'''

        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            pos_df = pd.DataFrame(index=exp_df.index)

            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton)

            tree_data = {}
                        
            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']
                
                if 'Nub' not in joint:
                    r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                    expmap = r.values
                    #expmap = [[f[1]['%s_alpha'%joint], f[1]['%s_beta'%joint], f[1]['%s_gamma'%joint]] for f in r.iterrows()]
                else:
                    expmap = np.zeros((exp_df.shape[0], 3))

                # Convert the eulers to rotation matrices
                #rotmats = np.asarray([Rotation(f, 'expmap').rotmat for f in expmap])
                #angs = np.linalg.norm(expmap,axis=1, keepdims=True)
                rotmats = self._expmap2rot(expmap)
                
                tree_data[joint]=[
                                    [], # to store the rotation matrix
                                    []  # to store the calculated position
                                 ] 
                pos_values = np.zeros((exp_df.shape[0], 3))

                if track.root_name == joint:
                    tree_data[joint][0] = rotmats
                    # tree_data[joint][1] = np.add(pos_values, track.skeleton[joint]['offsets'])
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent
                    tree_data[joint][0] = np.matmul(rotmats, tree_data[parent][0])

                    # add the position channel to the offset and store it in k, for every frame i
                    k = pos_values + track.skeleton[joint]['offsets']

                    # multiply k to the rotmat of the parent for every frame i
                    q = np.matmul(k.reshape(k.shape[0],1,3), tree_data[parent][0])

                    # add q to the position of the parent, for every frame i
                    tree_data[joint][1] = q.reshape(k.shape[0],3) + tree_data[parent][1]


                # Create the corresponding columns in the new DataFrame
                df = pd.DataFrame(data=tree_data[joint][1], 
                                  index=pos_df.index, 
                                  columns=['%s_Xposition'%joint, '%s_Yposition'%joint, '%s_Zposition'%joint])
                pos_df = pd.concat((pos_df, df), axis=1)

            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)
        return Q

    def _to_expmap(self, X):
        '''Converts Euler angles to Exponential Maps'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            exp_df = euler_df.copy()# pd.DataFrame(index=euler_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                #print(joint)
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                
                exp_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                euler = np.transpose(np.array([r[r1_col], r[r2_col], r[r3_col]]))
                #exps = [Rotation(f, 'euler', from_deg=True, order=rot_order).to_expmap() for f in euler] # Convert the eulers to exp maps
                exps = unroll(np.array([euler2expmap(f, rot_order, True) for f in euler])) # Convert the exp maps to eulers
                #exps = euler2expmap2(euler, rot_order, True) # Convert the eulers to exp maps

                # Create the corresponding columns in the new DataFrame    
                exp_df.insert(loc=0, column='%s_gamma'%joint, value=pd.Series(data=[e[2] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_beta'%joint, value=pd.Series(data=[e[1] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_alpha'%joint, value=pd.Series(data=[e[0] for e in exps], index=exp_df.index))

            #print(exp_df.columns)
            new_track = track.clone()
            new_track.values = exp_df
            Q.append(new_track)

        return Q
                
    def _expmap_to_euler(self, X):
        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            euler_df = exp_df.copy()
            
            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                jt_alpha = '%s_alpha'%joint
                jt_beta = '%s_beta'%joint
                jt_gamma = '%s_gamma'%joint
                
                euler_df.drop([jt_alpha, jt_beta, jt_gamma], axis=1, inplace=True)
                expmap = np.transpose(np.array([track.values[jt_alpha], track.values[jt_beta], track.values[jt_gamma]]))
                rot_order = track.skeleton[joint]['order']
                euler_rots = np.array(R.from_rotvec(expmap).as_euler(rot_order, degrees=True))                
                #euler_rots = [expmap2euler(f, rot_order, True) for f in expmap] # Convert the exp maps to eulers
                
                # Create the corresponding columns in the new DataFrame    
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q

    def _to_expmap2(self, X):
        '''Converts Euler angles to Exponential Maps'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            exp_df = euler_df.copy()# pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            #rxp = '%s_Xposition'%track.root_name
            #ryp = '%s_Yposition'%track.root_name
            #rzp = '%s_Zposition'%track.root_name
            #exp_df[rxp] = pd.Series(data=euler_df[rxp], index=exp_df.index)
            #exp_df[ryp] = pd.Series(data=euler_df[ryp], index=exp_df.index)
            #exp_df[rzp] = pd.Series(data=euler_df[rzp], index=exp_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']

                # Get the rotation columns that belong to this joint
                rc = euler_df[[c for c in rots if joint in c]]

                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                # Make sure the columns are organized in xyz order
                #print("joint:" + str(joint) + "  rot_order:" + str(rot_order))
                if rc.shape[1] < 3:
                    euler_values = np.zeros((euler_df.shape[0], 3))
                    rot_order = "XYZ"
                else:
                    euler_values = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                
                quats = Quaternions.from_euler(np.asarray(euler_values), order=rot_order.lower(), world=False)
                #exps = [Rotation(f, 'euler', from_deg=True, order=rot_order).to_expmap() for f in euler] # Convert the eulers to exp maps
                #exps = unroll(np.array([euler2expmap(f, rot_order, True) for f in euler])) # Convert the exp maps to eulers
                #exps = euler2expmap2(euler, rot_order, True) # Convert the eulers to exp maps
                # Create the corresponding columns in the new DataFrame
                if (self.ref_pose is not None):
                    q1_col = '%s_qWrotation'%(joint)
                    q2_col = '%s_qXrotation'%(joint)
                    q3_col = '%s_qYrotation'%(joint)
                    q4_col = '%s_qZrotation'%(joint)
                    ref_q = Quaternions(np.asarray([[f[1][q1_col], f[1][q2_col], f[1][q3_col], f[1][q4_col]] for f in self.ref_pose.values.iterrows()]))
                    #print("ref_q:" + str(ref_q.shape))
                    ref_q = ref_q[0,:]
                    quats=(-ref_q)*quats
    
                angles, axis = quats.angle_axis()
                aa = np.where(angles>np.pi)
                angles[aa] = angles[aa]-2*np.pi                
                #exps = unroll(angles[:,None]*axis)
                exps = angles[:,None]*axis
                #print(f"{joint}: {str(exps[0,:])}")

                #exps = np.array([quat2expmap(f) for f in quats])
                exp_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                exp_df.insert(loc=0, column='%s_gamma'%joint, value=pd.Series(data=[e[2] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_beta'%joint, value=pd.Series(data=[e[1] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_alpha'%joint, value=pd.Series(data=[e[0] for e in exps], index=exp_df.index))

            #print(exp_df.columns)
            new_track = track.clone()
            new_track.values = exp_df
            Q.append(new_track)

        return Q
        
    def _expmap_to_euler2(self, X):
        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            #euler_df = pd.DataFrame(index=exp_df.index)
            euler_df = exp_df.copy()

            # Copy the root positions into the new DataFrame
            #rxp = '%s_Xposition'%track.root_name
            #ryp = '%s_Yposition'%track.root_name
            #rzp = '%s_Zposition'%track.root_name
            #euler_df[rxp] = pd.Series(data=exp_df[rxp], index=euler_df.index)
            #euler_df[ryp] = pd.Series(data=exp_df[ryp], index=euler_df.index)
            #euler_df[rzp] = pd.Series(data=exp_df[rzp], index=euler_df.index)
            
            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                
                euler_df.drop(['%s_alpha'%joint, '%s_beta'%joint, '%s_gamma'%joint], axis=1, inplace=True)
                expmap = [[f[1]['%s_alpha'%joint], f[1]['%s_beta'%joint], f[1]['%s_gamma'%joint]] for f in r.iterrows()] # Make sure the columsn are organized in xyz order
                angs = np.linalg.norm(expmap, axis=1)
                quats=Quaternions.from_angle_axis(angs, expmap/(np.tile(angs[:, None]+1e-10, (1,3))))
                if (self.ref_pose is not None):
                    q1_col = '%s_qWrotation'%(joint)
                    q2_col = '%s_qXrotation'%(joint)
                    q3_col = '%s_qYrotation'%(joint)
                    q4_col = '%s_qZrotation'%(joint)
                    ref_q = Quaternions(np.asarray([[f[1][q1_col], f[1][q2_col], f[1][q3_col], f[1][q4_col]] for f in self.ref_pose.values.iterrows()]))
                    #print("ref_q:" + str(ref_q.shape))
                    ref_q = ref_q[0,:]
                    quats=ref_q*quats
                    
                euler_rots = 180/np.pi*quats.euler()
                track.skeleton[joint]['order'] = 'ZYX'
                rot_order = track.skeleton[joint]['order']
                #euler_rots = [Rotation(f, 'expmap').to_euler(True, rot_order) for f in expmap] # Convert the exp maps to eulers
                #euler_rots = [expmap2euler(f, rot_order, True) for f in expmap] # Convert the exp maps to eulers
                                                  
                # Create the corresponding columns in the new DataFrame
    
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q
        
    def _euler_to_vectors(self, X):
        '''Converts Euler angles to Up and Fwd vectors'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            vec_df = euler_df.copy()# pd.DataFrame(index=euler_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                #print(joint)
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                
                vec_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                euler = [[f[1][r1_col], f[1][r2_col], f[1][r3_col]] for f in r.iterrows()]
                vectors = np.array([euler2vectors(f, rot_order, True) for f in euler])
    
                vec_df.insert(loc=0, column='%s_xUp'%joint, value=pd.Series(data=[e[0] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_yUp'%joint, value=pd.Series(data=[e[1] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_zUp'%joint, value=pd.Series(data=[e[2] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_xFwd'%joint, value=pd.Series(data=[e[3] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_yFwd'%joint, value=pd.Series(data=[e[4] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_zFwd'%joint, value=pd.Series(data=[e[5] for e in vectors], index=vec_df.index))

            #print(exp_df.columns)
            new_track = track.clone()
            new_track.values = vec_df
            Q.append(new_track)

        return Q
            
    def _vectors_to_euler(self, X):
        '''Converts Up and Fwd vectors to Euler angles'''
        Q = []
        for track in X:
            channels = []
            titles = []
            vec_df = track.values

            # Create a new DataFrame to store the exponential map rep
            #euler_df = pd.DataFrame(index=exp_df.index)
            euler_df = vec_df.copy()

            # List the columns that contain rotation channels
            vec_params = [c for c in vec_df.columns if ( any(p in c for p in ['xUp', 'yUp','zUp','xFwd', 'yFwd','zFwd']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = vec_df[[c for c in vec_params if joint in c]] # Get the columns that belong to this joint
                
                euler_df.drop(['%s_xUp'%joint, '%s_yUp'%joint, '%s_zUp'%joint, '%s_xFwd'%joint, '%s_yFwd'%joint, '%s_zFwd'%joint], axis=1, inplace=True)
                vectors = [[f[1]['%s_xUp'%joint], f[1]['%s_yUp'%joint], f[1]['%s_zUp'%joint], f[1]['%s_xFwd'%joint], f[1]['%s_yFwd'%joint], f[1]['%s_zFwd'%joint]] for f in r.iterrows()] # Make sure the columsn are organized in xyz order
                rot_order = track.skeleton[joint]['order']
                euler_rots = [vectors2euler(f, rot_order, True) for f in vectors]
                
                # Create the corresponding columns in the new DataFrame
    
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q

#################### Motorica Mirror ###################################
class Mirror(BaseEstimator, TransformerMixin):
    def __init__(self, axis="X", append=True):
        """
        Mirrors the data 
        """
        self.axis = axis
        self.append = append
        
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print("Mirror: " + self.axis)
        Q = []
        
        if self.append:
            for track in X:
                Q.append(track)
            
        for track in X:
            channels = []
            titles = []
            
            if self.axis == "X":
                signs = np.array([1,-1,-1])
            if self.axis == "Y":
                signs = np.array([-1,1,-1])
            if self.axis == "Z":
                signs = np.array([-1,-1,1])

            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            new_df = pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            new_df[rxp] = pd.Series(data=-signs[0]*euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=-signs[1]*euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=-signs[2]*euler_df[rzp], index=new_df.index)
            
            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]
            #lft_rots = [c for c in euler_df.columns if ('Left' in c and 'rotation' in c and 'Nub' not in c)]
            #rgt_rots = [c for c in euler_df.columns if ('Right' in c and 'rotation' in c and 'Nub' not in c)]
            lft_joints = (joint for joint in track.skeleton if 'Left' in joint and 'Nub' not in joint)
            rgt_joints = (joint for joint in track.skeleton if 'Right' in joint and 'Nub' not in joint)
                        
            new_track = track.clone()

            # Efficiently mirror left ↔ right joint rotations with sign adjustments
            columns_to_add = {}
            for lft_joint in lft_joints:
                rgt_joint = lft_joint.replace('Left', 'Right')

                # Mirror right → left
                columns_to_add[f'{lft_joint}_Xrotation'] = signs[0] * track.values[f'{rgt_joint}_Xrotation']
                columns_to_add[f'{lft_joint}_Yrotation'] = signs[1] * track.values[f'{rgt_joint}_Yrotation']
                columns_to_add[f'{lft_joint}_Zrotation'] = signs[2] * track.values[f'{rgt_joint}_Zrotation']

                # Mirror left → right
                columns_to_add[f'{rgt_joint}_Xrotation'] = signs[0] * track.values[f'{lft_joint}_Xrotation']
                columns_to_add[f'{rgt_joint}_Yrotation'] = signs[1] * track.values[f'{lft_joint}_Yrotation']
                columns_to_add[f'{rgt_joint}_Zrotation'] = signs[2] * track.values[f'{lft_joint}_Zrotation']

            # Add all mirrored columns at once to avoid fragmentation
            new_df = pd.concat([new_df, pd.DataFrame(columns_to_add, index=new_df.index)], axis=1)
    
            # List the joints that are not left or right, i.e. are on the trunk
            joints = (joint for joint in track.skeleton if 'Nub' not in joint and 'Left' not in joint and 'Right' not in joint)

            # Create the corresponding columns for all joints at once to avoid fragmentation
            columns_to_add = {}
            for joint in joints:
                columns_to_add[f'{joint}_Xrotation'] = signs[0] * track.values[f'{joint}_Xrotation']
                columns_to_add[f'{joint}_Yrotation'] = signs[1] * track.values[f'{joint}_Yrotation']
                columns_to_add[f'{joint}_Zrotation'] = signs[2] * track.values[f'{joint}_Zrotation']

            # Concatenate all new columns at once
            new_df = pd.concat([new_df, pd.DataFrame(columns_to_add, index=new_df.index)], axis=1)

            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        return X


#################### Finedance Mirror ###################################
class MirrorYbot(BaseEstimator, TransformerMixin):
    def __init__(self, axis="X", append=True):
        self.axis = axis
        self.append = append

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Q = []

        if self.append:
            Q.extend(X)

        for track in X:
            new_df = track.values.copy()  # Copy motion data

            # ✅ Define mirroring signs based on the axis
            axis_signs = {
                "X": np.array([1, -1, -1]),  # Flip Y and Z
                "Y": np.array([-1, 1, -1]),  # Flip X and Z
                "Z": np.array([-1, -1, 1])   # Flip X and Y
            }

            if self.axis not in axis_signs:
                raise ValueError("Axis must be 'X', 'Y', or 'Z'.")

            signs = axis_signs[self.axis]

            # ✅ Mirror root position carefully
            root_pos_cols = [f"{track.root_name}_Xposition",
                             f"{track.root_name}_Yposition",
                             f"{track.root_name}_Zposition"]

            for i, col in enumerate(root_pos_cols):
                if col in new_df.columns:
                    if i == 1:  
                        if self.axis == "Y":  
                            new_df[col] = -track.values[col]  # Flip Y if mirroring along Y-axis
                        else:
                            new_df[col] = track.values[col]  # Keep height unchanged for X/Z mirroring
                    else:
                        new_df[col] = signs[i] * track.values[col]

            # ✅ Mirror Legs Correctly
            leg_joints = ["UpLeg", "Leg", "Foot", "ToeBase"]
            rotation_axes = ["Xrotation", "Yrotation", "Zrotation"]

            for suffix in leg_joints:
                left_limb = f"Left{suffix}"
                right_limb = f"Right{suffix}"

                if left_limb in track.skeleton and right_limb in track.skeleton:
                    for rot_axis in rotation_axes:
                        left_col = f"{left_limb}_{rot_axis}"
                        right_col = f"{right_limb}_{rot_axis}"

                        if left_col in new_df.columns and right_col in new_df.columns:
                            new_df[left_col], new_df[right_col] = (
                                signs[0] * new_df[right_col],  
                                signs[0] * new_df[left_col]
                            )

            # ✅ Mirror Chest and Spine
            trunk_joints = ["Spine", "Chest"]
            for trunk_joint in trunk_joints:
                for rot_axis in rotation_axes:
                    col = f"{trunk_joint}_{rot_axis}"
                    if col in new_df.columns:
                        new_df[col] = signs[0] * track.values[col]

            # # ✅ Swap Shoulders Without Changing Their Rotation
            # shoulder_joints = ["Shoulder"]
            # for suffix in shoulder_joints:
            #     left_shoulder = f"Left{suffix}"
            #     right_shoulder = f"Right{suffix}"

            #     if left_shoulder in track.skeleton and right_shoulder in track.skeleton:
            #         for rot_axis in rotation_axes:
            #             left_col = f"{left_shoulder}_{rot_axis}"
            #             right_col = f"{right_shoulder}_{rot_axis}"

            #             if left_col in new_df.columns and right_col in new_df.columns:
            #                 # 🚀 **Swap shoulders but do NOT flip their rotations**
            #                 new_df[left_col], new_df[right_col] = new_df[right_col], new_df[left_col]

            # # ✅ Swap Arms Left-Right Without Changing Rotation
            # arm_joints = ["Arm", "Hand"]

            # for suffix in arm_joints:
            #     left_arm = f"Left{suffix}"
            #     right_arm = f"Right{suffix}"

            #     if left_arm in track.skeleton and right_arm in track.skeleton:
            #         for rot_axis in rotation_axes:
            #             left_col = f"{left_arm}_{rot_axis}"
            #             right_col = f"{right_arm}_{rot_axis}"

            #             if left_col in new_df.columns and right_col in new_df.columns:
            #                 # 🚀 **Only swap arms, DO NOT apply sign flipping**
            #                 new_df[left_col], new_df[right_col] = (
            #                     new_df[right_col],  # Keep original values
            #                     new_df[left_col]
            #                 )

            # # ✅ Fix Forearm Rotation to Prevent Opposite Bending
            # forearm_joints = ["ForeArm"]

            # for suffix in forearm_joints:
            #     left_forearm = f"Left{suffix}"
            #     right_forearm = f"Right{suffix}"

            #     if left_forearm in track.skeleton and right_forearm in track.skeleton:
            #         for rot_axis in rotation_axes:
            #             left_col = f"{left_forearm}_{rot_axis}"
            #             right_col = f"{right_forearm}_{rot_axis}"

            #             if left_col in new_df.columns and right_col in new_df.columns:
            #                 if rot_axis == "Xrotation":
            #                     # 🚀 **Flip X rotation to fix elbow bending**
            #                     new_df[left_col], new_df[right_col] = (
            #                         -new_df[right_col],  
            #                         -new_df[left_col]
            #                     )
            #                 elif rot_axis == "Yrotation":
            #                     # 🚀 **Flip Y rotation to fix wrist twist**
            #                     new_df[left_col], new_df[right_col] = (
            #                         -new_df[right_col],  
            #                         -new_df[left_col]
            #                     )
            #                 elif rot_axis == "Zrotation":
            #                     # 🚀 **Keep Z rotation as is to maintain natural movement**
            #                     new_df[left_col], new_df[right_col] = (
            #                         new_df[right_col],  
            #                         new_df[left_col]
            #                     )

            # ✅ Clone track and update values
            new_track = track.clone()
            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        return X


class RootAugmentation(BaseEstimator, TransformerMixin):
    def __init__(self, axis="X", append=True):
        self.axis = axis
        self.append = append

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Q = []

        if self.append:
            Q.extend(X)

        for track in X:
            new_df = track.values.copy()  # Copy motion data

            # ✅ Define mirroring signs based on the axis
            axis_signs = {
                "X": np.array([1, -1, -1]),  # Flip Y and Z
                "Y": np.array([-1, 1, -1]),  # Flip X and Z
                "Z": np.array([-1, -1, 1])   # Flip X and Y
            }

            if self.axis not in axis_signs:
                raise ValueError("Axis must be 'X', 'Y', or 'Z'.")

            signs = axis_signs[self.axis]

            # ✅ Mirror root position carefully
            root_pos_cols = [f"{track.root_name}_Xposition",
                             f"{track.root_name}_Yposition",
                             f"{track.root_name}_Zposition"]

            for i, col in enumerate(root_pos_cols):
                if col in new_df.columns:
                    if i == 1:  
                        if self.axis == "Y":  
                            new_df[col] = -track.values[col]  # Flip Y if mirroring along Y-axis
                        else:
                            new_df[col] = track.values[col]  # Keep height unchanged for X/Z mirroring
                    else:
                        new_df[col] = signs[i] * track.values[col]

            # ✅ Clone track and update values
            new_track = track.clone()
            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        return X
#########################################################################


class EulerReorder(BaseEstimator, TransformerMixin):
    def __init__(self, new_order):
        """
        Add a 
        """
        self.new_order = new_order
        
    
    def fit(self, X, y=None):
        self.orig_skeleton = copy.deepcopy(X[0].skeleton)
        return self
    
    def transform(self, X, y=None):
        #print("EulerReorder")
        Q = []

        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            #new_df = pd.DataFrame(index=euler_df.index)
            new_df = euler_df.copy()

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            new_df[rxp] = pd.Series(data=euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=euler_df[rzp], index=new_df.index)
            
            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            new_track = track.clone()
            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                
                #euler = [[f[1][r1_col], f[1][r2_col], f[1][r3_col]] for f in r.iterrows()]
                euler = np.transpose(np.array([r[r1_col], r[r2_col], r[r3_col]]))                
                
                #euler = [[f[1]['%s_Xrotation'%(joint)], f[1]['%s_Yrotation'%(joint)], f[1]['%s_Zrotation'%(joint)]] for f in r.iterrows()]
                new_euler = [euler_reorder(f, rot_order, self.new_order, True) for f in euler]
                #new_euler = euler_reorder2(np.array(euler), rot_order, self.new_order, True)
                
                # Create the corresponding columns in the new DataFrame
                new_df['%s_%srotation'%(joint, self.new_order[0])] = pd.Series(data=[e[0] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, self.new_order[1])] = pd.Series(data=[e[1] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, self.new_order[2])] = pd.Series(data=[e[2] for e in new_euler], index=new_df.index)
    
                new_track.skeleton[joint]['order'] = self.new_order

            new_track.values = new_df
            Q.append(new_track)

        return Q
        
    def inverse_transform(self, X, copy=None, start_pos=None):
        return X

class JointSelector(BaseEstimator, TransformerMixin):
    '''
    Allows for filtering the mocap data to include only the selected joints
    '''
    def __init__(self, joints, include_root=False):
        self.joints = joints
        self.include_root = include_root

    def fit(self, X, y=None):
        selected_joints = []
        selected_channels = []

        if self.include_root:
            selected_joints.append(X[0].root_name)
        
        selected_joints.extend(self.joints)

        for joint_name in selected_joints:
            if joint_name.endswith("_Nub"):
                selected_channels.extend([o for o in X[0].values.columns if (joint_name + "_") in o])
            else:
                selected_channels.extend([o for o in X[0].values.columns if (joint_name + "_") in o and 'Nub' not in o])
        
        self.selected_joints = selected_joints
        self.selected_channels = selected_channels
        self.not_selected = X[0].values.columns.difference(selected_channels)
        self.not_selected_values = {c:X[0].values[c].values[0] for c in self.not_selected}

        self.orig_skeleton = X[0].skeleton
        return self

    def transform(self, X, y=None):
        Q = []
        for track in X:
            t2 = track.clone()
            for key in track.skeleton.keys():
                if key not in self.selected_joints:
                    t2.skeleton.pop(key)
            t2.values = track.values[self.selected_channels]

            for key in t2.skeleton.keys():
                to_remove = list(set(t2.skeleton[key]['children']) - set(self.selected_joints))
                [t2.skeleton[key]['children'].remove(c) for c in to_remove]

            Q.append(t2)
      
        return Q
    
    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            t2 = track.clone()
            # Ensure skeleton is a copy to prevent modifying self.orig_skeleton
            skeleton = self.orig_skeleton.copy()

            # Restore skeleton rotation orders
            for key, value in track.skeleton.items():
                skeleton[key]['order'] = value['order']

            t2.skeleton = skeleton

            # Restore ignored values efficiently using pd.concat()
            restore_values = {d: self.not_selected_values.get(d, np.nan) for d in self.not_selected}
            t2.values = pd.concat([t2.values, pd.DataFrame(restore_values, index=t2.values.index)], axis=1)

            Q.append(t2)

        return Q


# class Numpyfier(BaseEstimator, TransformerMixin):
#     """
#     Converts the values in a MocapData object into a NumPy array.
#     Useful for the final stage of a pipeline before training.

#     Also provides an inverse_transform method to convert the NumPy array 
#     back to a MocapData object with its original structure.
#     """

#     def __init__(self, indices=None):
#         """
#         Initializes the Numpyfier with optional indices to filter specific columns.

#         Parameters:
#         ----------
#         indices : list, optional
#             List of column names to retain during transformation. 
#             If None, all columns are used.
#         """
#         self.indices = indices

#     def fit(self, X, y=None):
#         """
#         Fit method for the transformer. Stores a copy of the first MocapData object 
#         with values cleared for use in inverse transformation.

#         Parameters:
#         ----------
#         X : list of MocapData
#             The input data to fit on.
        
#         Returns:
#         -------
#         self : Numpyfier
#             The fitted transformer.
#         """
#         if len(X) == 0:
#             raise ValueError("Input data X must contain at least one MocapData object.")
        
#         # Clone and clear the first MocapData object
#         self.org_mocap_ = X[0].clone()
#         self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)
        
#         # Store the column indices
#         if self.indices is not None:
#             self.indices_ = [col for col in self.indices if col in self.org_mocap_.values.columns]
#         else:
#             self.indices_ = self.org_mocap_.values.columns
        
#         return self

#     def transform(self, X, y=None):
#         """
#         Transforms the input MocapData objects into a NumPy array, retaining only the specified columns.

#         Parameters:
#         ----------
#         X : list of MocapData
#             The input data to transform.
        
#         Returns:
#         -------
#         numpy.ndarray
#             The transformed data as a NumPy array.
#         """
#         Q = []
        
#         for track in X:
#             # Filter the columns based on self.indices_
#             filtered_values = track.values[self.indices_]
#             Q.append(filtered_values.values)
#             # print("Numpyfier:", filtered_values.columns)  # Debugging: check the order of the data
            
#         return np.array(Q)

#     def inverse_transform(self, X, copy=None):
#         """
#         Inverse transforms the NumPy array back into MocapData objects.

#         Parameters:
#         ----------
#         X : numpy.ndarray
#             The input data to inverse transform.
#         copy : None or bool, optional
#             If True, makes a copy of the input array. Not used here.
        
#         Returns:
#         -------
#         list of MocapData
#             The inverse transformed data as a list of MocapData objects.
#         """
#         Q = []

#         for track in X:
#             new_mocap = self.org_mocap_.clone()
#             time_index = pd.to_timedelta(range(track.shape[0]), unit='s') * new_mocap.framerate
            
#             # Reconstruct the DataFrame with original columns
#             new_df = pd.DataFrame(data=track, index=time_index, columns=self.indices_)
            
#             new_mocap.values = new_df
#             Q.append(new_mocap)

#         return Q
    

# class Numpyfier(BaseEstimator, TransformerMixin):
#     '''
#     Just converts the values in a MocapData object into a numpy array
#     Useful for the final stage of a pipeline before training
#     '''
#     def __init__(self):
#         pass

#     def fit(self, X, y=None):
#         self.org_mocap_ = X[0].clone()
#         self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

#         return self

#     def transform(self, X, y=None):
#         print("Numpyfier")
#         Q = [track.values.to_numpy() for track in X]  # Convert DataFrame to NumPy array

#         # Find the maximum shape across all tracks
#         max_rows = max(track.shape[0] for track in Q)
#         max_cols = max(track.shape[1] for track in Q)

#         # Pad each track to have the same shape
#         padded_Q = [np.pad(track, ((0, max_rows - track.shape[0]), (0, max_cols - track.shape[1])), 
#                             mode='constant', constant_values=0) for track in Q]

#         return np.stack(padded_Q)  # Now all arrays have the same shape

#     def inverse_transform(self, X, copy=None):
#         Q = []

#         for track in X:
            
#             new_mocap = self.org_mocap_.clone()
#             time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

#             new_df =  pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)
            
#             new_mocap.values = new_df
            

#             Q.append(new_mocap)

#         return Q


class Numpyfier(BaseEstimator, TransformerMixin):
    """
    Converts MocapData objects into NumPy arrays for model training.
    Ensures selected features match post-ConstantsRemover output.
    """

    def __init__(self, indices=None):
        """
        :param indices: List of column names OR indices to keep. If None, keeps all columns AFTER preprocessing.
        """
        self.indices = indices
        self.selected_indices_ = None
        self.selected_columns_ = None
        self.org_mocap_ = None

    def fit(self, X, y=None):
        """
        Determines which columns to keep based on `ConstantsRemover` output.
        """
        if not X:
            raise ValueError("Numpyfier received an empty dataset!")

        # Clone the first MocapData instance as a reference
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values = self.org_mocap_.values.iloc[:0]  # Keep structure but remove rows

        all_columns = self.org_mocap_.values.columns.tolist()

        # ✅ **Get Features AFTER `ConstantsRemover`**
        if hasattr(self, 'post_constant_columns_'):
            all_columns = self.post_constant_columns_

        # Ensure indices match actual columns
        if self.indices is not None:
            if all(isinstance(i, str) for i in self.indices):
                self.selected_columns_ = [col for col in self.indices if col in all_columns]
                self.selected_indices_ = [all_columns.index(col) for col in self.selected_columns_]
            elif all(isinstance(i, int) for i in self.indices):
                self.selected_indices_ = [i for i in self.indices if i < len(all_columns)]
                self.selected_columns_ = [all_columns[i] for i in self.selected_indices_]
            else:
                raise ValueError("indices must be a list of column names (str) or indices (int).")
        else:
            self.selected_columns_ = all_columns
            self.selected_indices_ = list(range(len(all_columns)))

        print(f"\n📌 **Numpyfier Fit Complete (After ConstantsRemover)**")
        print(f"Expected Feature Count: {len(self.selected_columns_)}")
        print(f"Feature Names: {self.selected_columns_[:10]} ... (truncated)")

        return self

    def transform(self, X, y=None):
        """
        Converts MocapData tracks into NumPy arrays with uniform shape.
        """
        print("\n📢 **Numpyfier: Converting MocapData to NumPy array (After ConstantsRemover)**")

        if not X:
            raise ValueError("Numpyfier received an empty dataset during transform!")

        # Convert each track's DataFrame into a NumPy array, filtering by selected indices
        Q = []
        for i, track in enumerate(X):
            try:
                filtered_data = track.values.iloc[:, self.selected_indices_].to_numpy()
                Q.append(filtered_data)
            except Exception as e:
                print(f"⚠️ Error processing track {i}: {e}")

        # Determine max sequence length and feature count
        max_rows = max(track.shape[0] for track in Q)
        max_cols = max(track.shape[1] for track in Q)

        print(f"🛠 Padding sequences to uniform shape: (max {max_rows} frames, {max_cols} features)")

        # Pad tracks to the same shape with zeros
        padded_Q = [np.pad(track, ((0, max_rows - track.shape[0]), (0, max_cols - track.shape[1])) ,
                            mode='constant', constant_values=0) for track in Q]

        stacked_array = np.stack(padded_Q)

        print(f"✅ Final Data Shape: {stacked_array.shape} (samples, time, features)")
        return stacked_array

    def inverse_transform(self, X, copy=None):
        """
        Converts NumPy arrays back into MocapData objects with original column names.
        """
        print("\n🔄 **Numpyfier: Performing inverse transformation**")

        Q = []
        actual_feature_count = X.shape[2]

        # ✅ **Ensure Feature Count Matches**
        if actual_feature_count != len(self.selected_columns_):
            print(f"⚠️ Feature count mismatch in inverse_transform: Expected {len(self.selected_columns_)}, Got {actual_feature_count}")
            print(f"📌 Adjusting column list to match actual data...")

            # Trim feature names if too many, extend with placeholders if too few
            self.selected_columns_ = self.selected_columns_[:actual_feature_count]

        for i, track in enumerate(X):
            # Clone the original structure
            new_mocap = self.org_mocap_.clone()

            if track.shape[1] != len(self.selected_columns_):
                print(f"⚠️ Shape mismatch in track {i}: Expected {len(self.selected_columns_)}, Got {track.shape[1]}")
                continue  # Skip this track to avoid crashes

            # Create time index (assuming uniform time step)
            time_index = pd.RangeIndex(start=0, stop=track.shape[0], step=1)

            # Recreate DataFrame using selected columns
            new_df = pd.DataFrame(data=track, index=time_index, columns=self.selected_columns_)

            # Assign back to MocapData object
            new_mocap.values = new_df
            Q.append(new_mocap)

        print(f"✅ Successfully converted {len(Q)} tracks back to MocapData format.")
        return Q


class Slicer(BaseEstimator, TransformerMixin):
    '''
    Slice the data into intervals of equal size 
    '''
    def __init__(self, window_size, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        #print("Slicer")
        Q = []
        
        for track in X:
            vals = track.values.values
            nframes = vals.shape[0]
            overlap_frames = (int)(self.overlap*self.window_size)
            
            n_sequences = (nframes-overlap_frames)//(self.window_size-overlap_frames)
            
            if n_sequences>0:
                y = np.zeros((n_sequences, self.window_size, vals.shape[1]))

                # extract sequences from the input data
                for i in range(0,n_sequences):
                    frameIdx = (self.window_size-overlap_frames) * i
                    Q.append(vals[frameIdx:frameIdx+self.window_size,:])

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

            new_df =  pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)
            
            new_mocap.values = new_df

            Q.append(new_mocap)

        return Q

class RootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method, hips_axis_order="XYZ", position_smoothing=0, rotation_smoothing=0, separate_root=True, adaptive_smoothing=True):
        """
        Accepted methods:
            abdolute_translation_deltas
            pos_rot_deltas
        """
        self.method = method
        self.position_smoothing = position_smoothing
        self.rotation_smoothing = rotation_smoothing
        self.separate_root = separate_root
        self.hips_axis_order = hips_axis_order
        self.adaptive_smoothing = adaptive_smoothing
        
        # relative rotation from the hips awis the the x-side, y-up, z-forward convention
        rot_mat = np.zeros((3,3))
        for i in range(3):
            ax_i = ord(hips_axis_order[i])-ord("X")    
            rot_mat[i,ax_i]=1
        self.root_rotation_offset = Quaternions.from_transforms(rot_mat[np.newaxis, :, :])
        self.hips_side_axis = -rot_mat[0,:]
    
    def fit(self, X, y=None):
        return self
    

    def adaptive_trajectory_smoothing(self, positions, min_smoothing=1, max_smoothing=10):
        """
        Applies adaptive or fixed Gaussian smoothing to a trajectory.

        Parameters:
            positions (np.ndarray): N x 3 array of positions (X, Y, Z).
            position_smoothing (float): Fixed smoothing value if adaptive_smoothing=False.
            adaptive_smoothing (bool): Whether to apply adaptive smoothing.
            min_smoothing (float): Minimum adaptive smoothing.
            max_smoothing (float): Maximum adaptive smoothing.

        Returns:
            np.ndarray: Smoothed trajectory.
        """

        position_smoothing = self.position_smoothing 
        adaptive_smoothing = self.adaptive_smoothing

        if position_smoothing == 0:
            return positions  # No smoothing applied

        if adaptive_smoothing:
            # Compute velocity magnitude (how much position changes over time)
            velocity = np.linalg.norm(np.diff(positions, axis=0, prepend=positions[:1]), axis=1)

            # Normalize velocity between 0 and 1
            vel_min, vel_max = np.min(velocity), np.max(velocity)
            norm_velocity = (velocity - vel_min) / (vel_max - vel_min) if vel_max > vel_min else np.zeros_like(velocity)

            # Compute adaptive smoothing: less smoothing for fast motion, more for slow
            adaptive_smoothing_values = max_smoothing - (norm_velocity * (max_smoothing - min_smoothing))

            # Apply adaptive Gaussian filter
            smoothed_positions = np.zeros_like(positions)
            for i in range(positions.shape[1]):  # Iterate over X, Y, Z
                for t in range(len(positions)):  # Apply smoothing per timestep
                    smoothed_positions[t, i] = filters.gaussian_filter1d(
                        positions[:, i], sigma=adaptive_smoothing_values[t], mode='nearest'
                    )[t]  # Extract the smoothed value for this timestep

        else:
            # Fixed Gaussian smoothing
            smoothed_positions = filters.gaussian_filter1d(positions, sigma=position_smoothing, axis=0, mode='nearest')

        return smoothed_positions


    def adaptive_rotation_smoothing(self, forward, min_smoothing=1, max_smoothing=10):
        """
        Applies adaptive or fixed Gaussian smoothing to forward direction.

        Parameters:
            forward (np.ndarray): N x 3 array of forward vectors.
            rotation_smoothing (float): Fixed smoothing value if adaptive_smoothing=False.
            adaptive_smoothing (bool): Whether to apply adaptive smoothing.
            min_smoothing (float): Minimum adaptive smoothing.
            max_smoothing (float): Maximum adaptive smoothing.

        Returns:
            np.ndarray: Smoothed forward direction (unit vectors).
        """

        rotation_smoothing = self.rotation_smoothing 
        adaptive_smoothing = self.adaptive_smoothing

        if rotation_smoothing == 0:
            return forward  # No smoothing applied

        if adaptive_smoothing:
            # Compute angular velocity (change in forward direction)
            angular_velocity = np.linalg.norm(np.diff(forward, axis=0, prepend=forward[:1]), axis=1)

            # Normalize angular velocity between 0 and 1
            ang_min, ang_max = np.min(angular_velocity), np.max(angular_velocity)
            norm_ang_velocity = (angular_velocity - ang_min) / (ang_max - ang_min) if ang_max > ang_min else np.zeros_like(angular_velocity)

            # Compute adaptive smoothing: less smoothing for fast rotation, more for slow
            adaptive_smoothing_values = max_smoothing - (norm_ang_velocity * (max_smoothing - min_smoothing))

            # Apply adaptive Gaussian filter
            smoothed_forward = np.zeros_like(forward)
            for i in range(forward.shape[1]):  # Iterate over X, Y, Z
                for t in range(len(forward)):  # Apply smoothing per timestep
                    smoothed_forward[t, i] = filters.gaussian_filter1d(
                        forward[:, i], sigma=adaptive_smoothing_values[t], mode='nearest'
                    )[t]  # Extract the smoothed value for this timestep

        else:
            # Fixed Gaussian smoothing
            smoothed_forward = filters.gaussian_filter1d(forward, sigma=rotation_smoothing, axis=0, mode='nearest')

        # Normalize to keep it as a unit vector
        return smoothed_forward / np.linalg.norm(smoothed_forward, axis=-1, keepdims=True)

        
    def transform(self, X, y=None):
        #print("RootTransformer")
        Q = []

        for track in X:
            if self.method == 'abdolute_translation_deltas':
                new_df = track.values.copy()
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name
                
                x=track.values[xpcol].copy()
                z=track.values[zpcol].copy()
                
                if self.position_smoothing>0:
                    x_sm = filters.gaussian_filter1d(x, self.position_smoothing, axis=0, mode='nearest')    
                    z_sm = filters.gaussian_filter1d(z, self.position_smoothing, axis=0, mode='nearest')                    
                    dx = pd.Series(data=x_sm, index=new_df.index).diff()
                    dz = pd.Series(data=z_sm, index=new_df.index).diff()
                    new_df[xpcol] = x-x_sm
                    new_df[zpcol] = z-z_sm
                else:
                    dx = x.diff()
                    dz = z.diff()
                    new_df.drop([xpcol, zpcol], axis=1, inplace=True)
                    
                dx[0] = dx[1]
                dz[0] = dz[1]
                
                new_df[dxpcol] = dx
                new_df[dzpcol] = dz
                
                new_track = track.clone()
                new_track.values = new_df
            # end of abdolute_translation_deltas
            
            elif self.method == 'pos_rot_deltas':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name
                
                #rot_order = track.skeleton[track.root_name]['order']
                #%(joint, rot_order[0])

                rot_order = track.skeleton[track.root_name]['order']
                r1_col = '%s_%srotation'%(track.root_name, rot_order[0])
                r2_col = '%s_%srotation'%(track.root_name, rot_order[1])
                r3_col = '%s_%srotation'%(track.root_name, rot_order[2])

                # Delta columns
                # dxp_col = '%s_dXposition'%track.root_name
                # dzp_col = '%s_dZposition'%track.root_name

                # dxr_col = '%s_dXrotation'%track.root_name
                # dyr_col = '%s_dYrotation'%track.root_name
                # dzr_col = '%s_dZrotation'%track.root_name
                dxp_col = 'reference_dXposition'
                dzp_col = 'reference_dZposition'
                dxr_col = 'reference_dXrotation'
                dyr_col = 'reference_dYrotation'
                dzr_col = 'reference_dZrotation'

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                
                """ Get Trajectory and smooth it"""                
                reference = positions.copy()*np.array([1,0,1])
                reference = self.adaptive_trajectory_smoothing(reference)
                
                """ Get Root Velocity """
                velocity = np.diff(reference, axis=0)                
                velocity = np.vstack((velocity[0,:], velocity))

                """ Remove Root Translation """
                positions = positions-reference

                """ Get Forward Direction along the x-z plane, assuming character is facig z-forward """
                #forward = [Rotation(f, 'euler', from_deg=True, order=rot_order).rotmat[:,2] for f in rotations] # get the z-axis of the rotation matrix, assuming character is facig z-forward
                #print("order:" + rot_order.lower())
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)
                #forward = quats*np.array([[0,0,1]])
                #forward[:,1] = 0
                side_dirs = quats*self.hips_side_axis
                forward = np.cross(np.array([[0,1,0]]), side_dirs)

                # """ Smooth Forward Direction """                
                # direction_filterwidth = self.rotation_smoothing
                # if direction_filterwidth>0:
                #     forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    

                # forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

                """ Smooth Forward Direction with Adaptive Smoothing """
                forward = self.adaptive_rotation_smoothing(forward)

                """ Remove Y Rotation """
                target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
                rotation = Quaternions.between(target, forward)[:,np.newaxis]    
                positions = (-rotation[:,0]) * positions
                #new_rotations = (-rotation[:,0]) * quats
                new_rotations = (-self.root_rotation_offset) * (-rotation[:,0]) * quats

                """ Get Root Rotation """
                #print(rotation[:,0])
                velocity = (-rotation[:,0]) * velocity
                rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
                rvelocity = np.vstack((rvelocity[0], rvelocity))

                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in new_rotations])*180.0/np.pi
                
                new_df = track.values.copy()

                root_pos_x = pd.Series(data=positions[:,0], index=new_df.index)
                root_pos_y = pd.Series(data=positions[:,1], index=new_df.index)
                root_pos_z = pd.Series(data=positions[:,2], index=new_df.index)
                root_pos_x_diff = pd.Series(data=velocity[:,0], index=new_df.index)
                root_pos_z_diff = pd.Series(data=velocity[:,2], index=new_df.index)

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                root_rot_y_diff = pd.Series(data=rvelocity[:,0], index=new_df.index)
                
                #new_df.drop([xr_col, yr_col, zr_col, xp_col, zp_col], axis=1, inplace=True)

                new_df[xp_col] = root_pos_x
                new_df[yp_col] = root_pos_y
                new_df[zp_col] = root_pos_z
                new_df[dxp_col] = root_pos_x_diff
                new_df[dzp_col] = root_pos_z_diff

                new_df[r1_col] = root_rot_1
                new_df[r2_col] = root_rot_2
                new_df[r3_col] = root_rot_3
                #new_df[dxr_col] = root_rot_x_diff
                new_df[dyr_col] = root_rot_y_diff
                #new_df[dzr_col] = root_rot_z_diff

                new_track.values = new_df
            elif self.method == 'pos_xyz_rot_deltas':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name
                
                #rot_order = track.skeleton[track.root_name]['order']
                #%(joint, rot_order[0])

                rot_order = track.skeleton[track.root_name]['order']
                r1_col = '%s_%srotation'%(track.root_name, rot_order[0])
                r2_col = '%s_%srotation'%(track.root_name, rot_order[1])
                r3_col = '%s_%srotation'%(track.root_name, rot_order[2])

                # Delta columns
                # dxp_col = '%s_dXposition'%track.root_name
                # dzp_col = '%s_dZposition'%track.root_name

                # dxr_col = '%s_dXrotation'%track.root_name
                # dyr_col = '%s_dYrotation'%track.root_name
                # dzr_col = '%s_dZrotation'%track.root_name
                dxp_col = 'reference_dXposition'
                dyp_col = 'reference_dYposition'
                dzp_col = 'reference_dZposition'
                dxr_col = 'reference_dXrotation'
                dyr_col = 'reference_dYrotation'
                dzr_col = 'reference_dZrotation'

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                
                """ Get Trajectory and smooth it"""                
                reference = positions.copy()*np.array([1,0,1])
                reference = self.adaptive_trajectory_smoothing(reference)
                
                """ Get Root Velocity """
                velocity = np.diff(reference, axis=0)                
                velocity = np.vstack((velocity[0,:], velocity))

                """ Remove Root Translation """
                positions = positions-reference

                """ Get Forward Direction along the x-z plane, assuming character is facig z-forward """
                #forward = [Rotation(f, 'euler', from_deg=True, order=rot_order).rotmat[:,2] for f in rotations] # get the z-axis of the rotation matrix, assuming character is facig z-forward
                #print("order:" + rot_order.lower())
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)

                #calculate the hips forward directions given in global cordinates 
                #side_ax = np.zeros((1,3))
                #side_ax[0,self.hips_side_axis]=1
                #side_dirs = quats*side_ax
                side_dirs = quats*self.hips_side_axis
                forward = np.cross(np.array([[0,1,0]]), side_dirs)

                # """ Smooth Forward Direction """                
                # direction_filterwidth = self.rotation_smoothing
                # if direction_filterwidth>0:
                #     forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    

                """ Smooth Forward Direction with Adaptive Smoothing """
                forward = self.adaptive_rotation_smoothing(forward)

                """ Remove Y Rotation """
                target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
                rotation = Quaternions.between(target, forward)[:,np.newaxis]    
                positions = (-rotation[:,0]) * positions
                new_rotations = (-self.root_rotation_offset) * (-rotation[:,0]) * quats

                """ Get Root Rotation """
                #print(rotation[:,0])
                velocity = (-rotation[:,0]) * velocity
                rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
                rvelocity = np.vstack((rvelocity[0], rvelocity))

                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in new_rotations])*180.0/np.pi
                
                new_df = track.values.copy()

                root_pos_x = pd.Series(data=positions[:,0], index=new_df.index)
                root_pos_y = pd.Series(data=positions[:,1], index=new_df.index)
                root_pos_z = pd.Series(data=positions[:,2], index=new_df.index)
                root_pos_x_diff = pd.Series(data=velocity[:,0], index=new_df.index)
                root_pos_y_diff = pd.Series(data=velocity[:,1], index=new_df.index)
                root_pos_z_diff = pd.Series(data=velocity[:,2], index=new_df.index)

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                root_rot_y_diff = pd.Series(data=rvelocity[:,0], index=new_df.index)
                
                #new_df.drop([xr_col, yr_col, zr_col, xp_col, zp_col], axis=1, inplace=True)

                new_df[xp_col] = root_pos_x
                new_df[yp_col] = root_pos_y
                new_df[zp_col] = root_pos_z
                new_df[dxp_col] = root_pos_x_diff
                new_df[dyp_col] = root_pos_y_diff
                new_df[dzp_col] = root_pos_z_diff

                new_df[r1_col] = root_rot_1
                new_df[r2_col] = root_rot_2
                new_df[r3_col] = root_rot_3
                #new_df[dxr_col] = root_rot_x_diff
                new_df[dyr_col] = root_rot_y_diff
                #new_df[dzr_col] = root_rot_z_diff

                new_track.values = new_df


            elif self.method == 'hip_centric':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name
                
                new_df = track.values.copy()

                all_zeros = np.zeros(track.values[xp_col].values.shape)

                new_df[xp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[yp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zp_col] = pd.Series(data=all_zeros, index=new_df.index)

                new_df[xr_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[yr_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zr_col] = pd.Series(data=all_zeros, index=new_df.index)

                new_track.values = new_df

            #print(new_track.values.columns)
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        Q = []

        #TODO: simplify this implementation

        startx = 0
        startz = 0

        if start_pos is not None:
            startx, startz = start_pos

        for track in X:
            new_track = track.clone()
            if self.method == 'abdolute_translation_deltas':
                new_df = new_track.values
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name

                dx = track.values[dxpcol].values
                dz = track.values[dzpcol].values

                recx = [startx]
                recz = [startz]

                for i in range(dx.shape[0]-1):
                    recx.append(recx[i]+dx[i+1])
                    recz.append(recz[i]+dz[i+1])

                # recx = [recx[i]+dx[i+1] for i in range(dx.shape[0]-1)]
                # recz = [recz[i]+dz[i+1] for i in range(dz.shape[0]-1)]
                # recx = dx[:-1] + dx[1:]
                # recz = dz[:-1] + dz[1:]
                if self.position_smoothing > 0:                    
                    new_df[xpcol] = pd.Series(data=new_df[xpcol]+recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=new_df[zpcol]+recz, index=new_df.index)
                else:
                    new_df[xpcol] = pd.Series(data=recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=recz, index=new_df.index)

                new_df.drop([dxpcol, dzpcol], axis=1, inplace=True)
                
                new_track.values = new_df
            # end of abdolute_translation_deltas
            
            elif self.method == 'pos_rot_deltas':
                # Absolute columns
                rot_order = track.skeleton[track.root_name]['order']
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name
                r1_col = '%s_%srotation'%(track.root_name, rot_order[0])
                r2_col = '%s_%srotation'%(track.root_name, rot_order[1])
                r3_col = '%s_%srotation'%(track.root_name, rot_order[2])

                # Delta columns
                # dxp_col = '%s_dXposition'%track.root_name
                # dzp_col = '%s_dZposition'%track.root_name
                # dyr_col = '%s_dYrotation'%track.root_name
                dxp_col = 'reference_dXposition'
                dzp_col = 'reference_dZposition'
                dyr_col = 'reference_dYrotation'

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)

                new_df = track.values.copy()

                dx = track.values[dxp_col].values
                dz = track.values[dzp_col].values

                dry = track.values[dyr_col].values

                #rec_p = np.array([startx, 0, startz])+positions[0,:]
                rec_ry = Quaternions.id(quats.shape[0])
                rec_xp = [0]
                rec_zp = [0]

                #rec_r = Quaternions.id(quats.shape[0])

                for i in range(dx.shape[0]-1):
                    #print(dry[i])
                    q_y = Quaternions.from_angle_axis(np.array(dry[i+1]), np.array([0,1,0]))
                    rec_ry[i+1] = q_y*rec_ry[i]
                    #print("dx: + " + str(dx[i+1]))
                    dp = rec_ry[i+1]*np.array([dx[i+1], 0, dz[i+1]])
                    rec_xp.append(rec_xp[i]+dp[0,0])
                    rec_zp.append(rec_zp[i]+dp[0,2])
                    
                if self.separate_root:
                    qq = quats
                    xx = positions[:,0]
                    zz = positions[:,2]
                else:
                    qq = rec_ry*self.root_rotation_offset*quats
                    pp = rec_ry*positions
                    xx = rec_xp + pp[:,0]
                    zz = rec_zp + pp[:,2]
                
                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in qq])*180.0/np.pi
                
                new_df = track.values.copy()

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                
                new_df[xp_col] = pd.Series(data=xx, index=new_df.index)
                new_df[zp_col] = pd.Series(data=zz, index=new_df.index)

                new_df[r1_col] = pd.Series(data=root_rot_1, index=new_df.index)
                new_df[r2_col] = pd.Series(data=root_rot_2, index=new_df.index)
                new_df[r3_col] = pd.Series(data=root_rot_3, index=new_df.index)

                if self.separate_root:
                    ref_rot_order="ZXY"
                    new_df["reference_Xposition"] = pd.Series(data=rec_xp, index=new_df.index)
                    new_df["reference_Zposition"] = pd.Series(data=rec_zp, index=new_df.index)                    
                    eulers_ry = np.array([t3d.euler.quat2euler(q, axes=('s'+ref_rot_order.lower()[::-1]))[::-1] for q in rec_ry])*180.0/np.pi
                    new_df["reference_Yrotation"] = pd.Series(data=eulers_ry[:,ref_rot_order.find('Y')], index=new_df.index)
                    

                new_df.drop([dyr_col, dxp_col, dzp_col], axis=1, inplace=True)


                new_track.values = new_df
                
            elif self.method == 'pos_xyz_rot_deltas':
                # Absolute columns
                rot_order = track.skeleton[track.root_name]['order']
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name
                r1_col = '%s_%srotation'%(track.root_name, rot_order[0])
                r2_col = '%s_%srotation'%(track.root_name, rot_order[1])
                r3_col = '%s_%srotation'%(track.root_name, rot_order[2])

                # Delta columns
                # dxp_col = '%s_dXposition'%track.root_name
                # dzp_col = '%s_dZposition'%track.root_name
                # dyr_col = '%s_dYrotation'%track.root_name
                dxp_col = 'reference_dXposition'
                dyp_col = 'reference_dYposition'
                dzp_col = 'reference_dZposition'
                dyr_col = 'reference_dYrotation'

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)

                new_df = track.values.copy()

                dx = track.values[dxp_col].values
                dy = track.values[dyp_col].values
                dz = track.values[dzp_col].values

                dry = track.values[dyr_col].values

                #rec_p = np.array([startx, 0, startz])+positions[0,:]
                rec_ry = Quaternions.id(quats.shape[0])
                rec_xp = [0]
                rec_yp = [0]
                rec_zp = [0]

                #rec_r = Quaternions.id(quats.shape[0])

                for i in range(dx.shape[0]-1):
                    #print(dry[i])
                    q_y = Quaternions.from_angle_axis(np.array(dry[i+1]), np.array([0,1,0]))
                    rec_ry[i+1] = q_y*rec_ry[i]
                    #print("dx: + " + str(dx[i+1]))
                    dp = rec_ry[i+1]*np.array([dx[i+1], dy[i+1], dz[i+1]])
                    rec_xp.append(rec_xp[i]+dp[0,0])
                    rec_yp.append(rec_yp[i]+dp[0,1])
                    rec_zp.append(rec_zp[i]+dp[0,2])
                    
                if self.separate_root:
                    qq = quats
                    xx = positions[:,0]
                    yy = positions[:,1]
                    zz = positions[:,2]
                else:
                    qq = rec_ry*self.root_rotation_offset*quats
                    pp = rec_ry*positions
                    xx = rec_xp + pp[:,0]
                    yy = rec_yp + pp[:,1]
                    zz = rec_zp + pp[:,2]
                
                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in qq])*180.0/np.pi
                
                new_df = track.values.copy()

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                
                new_df[xp_col] = pd.Series(data=xx, index=new_df.index)
                new_df[yp_col] = pd.Series(data=yy, index=new_df.index)
                new_df[zp_col] = pd.Series(data=zz, index=new_df.index)

                new_df[r1_col] = pd.Series(data=root_rot_1, index=new_df.index)
                new_df[r2_col] = pd.Series(data=root_rot_2, index=new_df.index)
                new_df[r3_col] = pd.Series(data=root_rot_3, index=new_df.index)

                if self.separate_root:
                    new_df["reference_Xposition"] = pd.Series(data=rec_xp, index=new_df.index)
                    new_df["reference_Yposition"] = pd.Series(data=rec_yp, index=new_df.index)
                    new_df["reference_Zposition"] = pd.Series(data=rec_zp, index=new_df.index)                    
                    eulers_ry = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in rec_ry])*180.0/np.pi
                    new_df["reference_Yrotation"] = pd.Series(data=eulers_ry[:,rot_order.find('Y')], index=new_df.index)
                    

                new_df.drop([dyr_col, dxp_col, dyp_col, dzp_col], axis=1, inplace=True)


                new_track.values = new_df
                
            #print(new_track.values.columns)
            Q.append(new_track)

        return Q



class RootCentricPositionNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            all_but_root = [joint for joint in track.skeleton if track.root_name not in joint]
            # all_but_root = [joint for joint in track.skeleton]
            for joint in all_but_root:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]-projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]-projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]-projected_root_pos[rzp], index=new_df.index)
            

            # keep the root as it is now
            new_df[rxp] = track.values[rxp]
            new_df[ryp] = track.values[ryp]
            new_df[rzp] = track.values[rzp]

            new_track.values = new_df

            Q.append(new_track)
        
        return Q

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            for joint in track.skeleton:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]+projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]+projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]+projected_root_pos[rzp], index=new_df.index)
                

            new_track.values = new_df

            Q.append(new_track)
        
        return Q

class Flattener(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.concatenate(X, axis=0)

class ConstantsRemover(BaseEstimator, TransformerMixin):
    '''
    Dynamically removes features with zero variance across all tracks.
    '''

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.const_dims_ = []
        self.const_values_ = {}

    def fit(self, X, y=None):
        """
        Detects constant features across ALL tracks.
        """
        # Stack all data to compute global variance
        all_values = np.vstack([track.values for track in X])  
        stds = all_values.std(axis=0)  # Compute standard deviation for each feature
        cols = X[0].values.columns.values  # Feature names

        # Identify zero-variance columns
        self.const_dims_ = [cols[i] for i in range(len(cols)) if stds[i] < self.eps]
        self.const_values_ = {c: X[0].values[c].values[0] for c in self.const_dims_}  # Store constant values

        print(f"Removing constant features: {self.const_dims_}")
        return self

    def transform(self, X, y=None):
        """
        Removes detected constant features from each track.
        """
        Q = []
        for track in X:
            t2 = track.clone()
            t2.values.drop(self.const_dims_, axis=1, inplace=True)
            Q.append(t2)
        return Q

    def inverse_transform(self, X, copy=None):
        """
        Restores removed features with their original values.
        """
        Q = []
        for track in X:
            t2 = track.clone()
            for d in self.const_dims_:
                t2.values[d] = self.const_values_[d]  # Restore original constant value
            Q.append(t2)
        return Q


class ListStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame
    
    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_mean_ = np.mean(X_train_flat, axis=0)
        self.data_std_ = np.std(X_train_flat, axis=0)

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_mean_) / self.data_std_
            else:
                normalized_track = (track - self.data_mean_) / self.data_std_

            Q.append(normalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            
            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * self.data_std_) + self.data_mean_
            else:
                unnormalized_track = (track * self.data_std_) + self.data_mean_

            Q.append(unnormalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

class ListMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame
    
    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_max_ = np.max(X_train_flat, axis=0)
        self.data_min_ = np.min(X_train_flat, axis=0)

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_min_) / (self.data_max_ - self.data_min_) 
            else:
                normalized_track = (track - self.data_min_) / (self.data_max_ - self.data_min_)

            Q.append(normalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            
            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * (self.data_max_ - self.data_min_)) + self.data_min_
            else:
                unnormalized_track = (track * (self.data_max_ - self.data_min_)) + self.data_min_

            Q.append(unnormalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)
        
class Resampler(BaseEstimator, TransformerMixin):
    def __init__(self, fps, method='cubic'):
        '''
        Method to resample a pandas dataframe to a different framerate.
        NOTE: Pandas resampling is quit unintuitive when resampling to odd framerates using interpolation.
        Thus we do it in this complex way.
        '''
        self.tgt_frametime = 1.0/fps
        self.method = method
    
    def fit(self, X, y=None):
        #print("Resampling to tgt_frametime: " + str(self.tgt_frametime))
        self.orig_frametime=X[0].framerate
        return self
    
    def resample_dataframe(self, df, frametime, method='cubic'):
        #Create a time index for the resampled data
        rate = str(round(1.0e9*frametime))+'N'
        time_index = df.resample(rate).indices
        
        #reindex the old data. This will turn all non-matching indices to NAN
        tmp = df.reindex(time_index)
        
        #merge with the old data and sort
        tmp = pd.concat([df, tmp]).sort_index()
        
        #remove duplicate time indices. Then fill the NAN values using interpolation
        tmp=tmp[~tmp.index.duplicated(keep='first')].interpolate(method=method)

        #return the values using the resampled indices
        return tmp.loc[list(time_index)]
        
    def resample_df(self, df, new_frametime, old_frametime, mode='cubic'):

        #Create a time index for the resampled data
        data = df.values

        nframes = data.shape[0]
        nframes_new = round(nframes*old_frametime/new_frametime)
        x = np.arange(0, nframes)/(nframes-1)
        xnew = np.arange(0, nframes_new)/(nframes_new-1)

        data_out = np.zeros((nframes_new, data.shape[1]))
        for jj in range(data.shape[1]):
            y = data[:,jj]
            f = interpolate.interp1d(x, y, bounds_error=False, kind=mode, fill_value='extrapolate')
            data_out[:,jj] = f(xnew)

        time_index = pd.to_timedelta([f for f in range(xnew.shape[0])], unit='s')*new_frametime
        out = pd.DataFrame(data=data_out, index=time_index, columns=df.columns)
        
        #Scale root deltas to match new frame-rate
        sc = nframes/nframes_new
        rootdelta_cols = [c for c in df.columns if ('reference_d' in c)]    
        out[rootdelta_cols]*=sc

        return out 

    # def resample_poly_df(self, df, new_frametime, old_frametime):
        # old_fps = round(1/old_frametime)
        # new_fps = round(1/new_frametime)
        # lcm = np.lcm(old_fps, new_fps)
        # up = lcm//old_fps
        # down = lcm//new_fps
        # new_vals = signal.resample_poly(df.values, up, down, padtype='line')
        # time_index = pd.to_timedelta([f for f in range(new_vals.shape[0])], unit='s')*new_frametime
        # new_df =  pd.DataFrame(data=new_vals, index=time_index, columns=df.columns)
        # return new_df
        
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            new_track = track.clone()
            # if self.method=="resample_poly":
                # new_track.values = self.resample_poly_df(track.values, self.tgt_frametime, track.framerate)
            # else:
            new_track.values = self.resample_df(track.values, self.tgt_frametime, track.framerate, self.method)
            #new_track.values = self.resample_dataframe(track.values, self.tgt_frametime, method=self.method)
            new_track.framerate = self.tgt_frametime
            Q.append(new_track)
        
        return Q
        
    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            new_track = track.clone()
            #new_track.values = self.resample_dataframe(track.values, self.orig_frametime, method=self.method)
            if self.method=="resample_poly":
                new_track.values = self.resample_poly_df(track.values, self.orig_frametime, track.framerate)
            else:
                new_track.values = self.resample_df(track.values, self.orig_frametime, track.framerate, self.method)
            new_track.framerate = self.orig_frametime
            Q.append(new_track)
        
        return Q
    
class DownSampler(BaseEstimator, TransformerMixin):
    def __init__(self, tgt_fps, keep_all=False):
        self.tgt_fps = tgt_fps
        self.keep_all = keep_all
        
    
    def fit(self, X, y=None):    

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            orig_fps=round(1.0/track.framerate)
            rate = orig_fps//self.tgt_fps
            if orig_fps%self.tgt_fps!=0:
                print("error orig_fps (" + str(orig_fps) + ") is not dividable with tgt_fps (" + str(self.tgt_fps) + ")")
            else:
                print("downsampling with rate: " + str(rate))
                
            #print(track.values.size)
            for ii in range(0,rate):
                new_track = track.clone()
                if self.keep_all:
                    new_track.take_name = new_track.take_name + "_" + str(ii).zfill(2)
                new_track.values = track.values[ii::rate].copy()            
                #print(new_track.values.size)
                #new_track = track[0:-1:self.rate]
                new_track.framerate = 1.0/self.tgt_fps
                Q.append(new_track)
                if not self.keep_all:
                    break
        
        return Q
        
    def inverse_transform(self, X, copy=None):
      return X

class ReverseTime(BaseEstimator, TransformerMixin):
    def __init__(self, append=True):
        self.append = append
        
    
    def fit(self, X, y=None):    

        return self
    
    def transform(self, X, y=None):
        #print("ReverseTime")
        Q = []
        if self.append:
            for track in X:
                Q.append(track)
        for track in X:
            new_track = track.clone()                            
            new_track.values = track.values[-1::-1]
            new_track.values.index=new_track.values.index[0]-new_track.values.index
            Q.append(new_track)
        
        return Q
        
    def inverse_transform(self, X, copy=None):
      return X

class ListFeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self, processors):
        self.processors = processors
    
    def fit(self, X, y=None):
        assert(y is None)
        for proc in self.processors:
            if isinstance(proc, Pipeline):
                #Loop steps and run fit on each. This is necessary since
                #running fit on a Pipeline runs fit_transform on all steps
                #and not only fit.
                for step in proc.steps:
                    step[1].fit(X)
            else:
                proc.fit(X)
        return self
    
    def transform(self, X, y=None):

        assert(y is None)
        #print("ListFeatureUnion")

        Q = []
        
        idx=0
        for proc in self.processors:
            Z = proc.transform(X)
            if idx==0:
                Q = Z
            else:
                assert(len(Q)==len(Z))
                for idx2,track in enumerate(Z):
                    Q[idx2].values = pd.concat([Q[idx2].values,Z[idx2].values], axis=1)
            idx += 1
                    
        return Q

    def inverse_transform(self, X, y=None):
        return X

class RollingStatsCalculator(BaseEstimator, TransformerMixin):
    '''
    Creates a causal mean and std filter with a rolling window of length win (based on using prev and current values)
    '''
    def __init__(self, win):
        self.win = win
            
    def fit(self, X, y=None):    

        return self
    
    def transform(self, X, y=None):
        #print("RollingStatsCalculator: " + str(self.win))

        Q = []
        for track in X:
            new_track = track.clone()
            mean_df = track.values.rolling(window=self.win).mean()            
            std_df = track.values.rolling(window=self.win).std()
            # rolling.mean results in Nans in start seq. Here we fill these
            win = min(self.win, new_track.values.shape[0])
            for i in range(1,win):
                mm=track.values[:i].rolling(window=i).mean()
                ss=track.values[:i].rolling(window=i).std()
                mean_df.iloc[i-1] = mm.iloc[i-1]
                std_df.iloc[i-1] = ss.iloc[i-1]
              
            std_df.iloc[0] = std_df.iloc[1]
            # Append to
            new_track.values=pd.concat([mean_df.add_suffix('_mean'), std_df.add_suffix('_std')], axis=1)
            Q.append(new_track)        
        return Q
        
    def inverse_transform(self, X, copy=None):
        return X

class FeatureCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.n_features = None  # Store number of features dynamically

    def fit(self, X, y=None):
        """
        Count the number of features AFTER other transformations.
        """
        if isinstance(X, list) and len(X) > 0:
            self.n_features = len(X[0].values.columns)  # Count features dynamically
        else:
            raise ValueError("FeatureCounter received an empty dataset!")

        print(f"\n📊 **Final Feature Count (After ConstantsRemover): {self.n_features}**")
        print(f"Features: {X[0].values.columns.tolist()}")
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, copy=None):
        """
        Ensure the transformed data has the correct feature dimensions before inversion.
        """
        if self.n_features is None:
            raise ValueError("FeatureCounter was not fitted properly. n_features is None.")

        for track in X:
            if len(track.values.columns) != self.n_features:
                raise ValueError(
                    f"Feature mismatch! Expected {self.n_features}, but got {len(track.values.columns)}."
                )

        return X

#TODO: JointsSelector (x)
#TODO: SegmentMaker
#TODO: DynamicFeaturesAdder
#TODO: ShapeFeaturesAdder
#TODO: DataFrameNumpier (x)

class TemplateTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, '__iter__') and not isinstance(X, (dict, str)):
            return [self._drop_columns(df) for df in X]
        else:
            return self._drop_columns(X)

    def inverse_transform(self, X):
        # Re-add missing columns as 0s
        if hasattr(X, '__iter__') and not isinstance(X, (dict, str)):
            return [self._restore_columns(df) for df in X]
        else:
            return self._restore_columns(X)

    def _drop_columns(self, df):
        if hasattr(df, 'values'):
            for col in self.columns_to_drop:
                if col in df.values.columns:
                    df.values.drop(columns=col, inplace=True)
        else:
            df.drop(columns=[col for col in self.columns_to_drop if col in df.columns], inplace=True)
        return df

    def _restore_columns(self, df):
        if hasattr(df, 'values'):
            for col in self.columns_to_drop:
                if col not in df.values.columns:
                    df.values[col] = 0.0
        else:
            for col in self.columns_to_drop:
                if col not in df.columns:
                    df[col] = 0.0
        return df


