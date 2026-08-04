import pandas as pd
import numpy as np
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
#import IPython
import os

def save_fig(fig_id, tight_layout=True):
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_id + '.png', format='png', dpi=300)
    
    
def draw_stickfigure(mocap_track, frame, data=None, joints=None, draw_names=False, ax=None, figsize=(8,8)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints
    
    if data is None:
        df = mocap_track.values
    else:
        df = data
        
    for joint in joints_to_draw:
        ax.scatter(x=df['%s_Xposition'%joint][frame], 
                   y=df['%s_Yposition'%joint][frame],  
                   alpha=0.6, c='b', marker='o')

        parent_x = df['%s_Xposition'%joint][frame]
        parent_y = df['%s_Yposition'%joint][frame]
        
        children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]
        
        for c in children_to_draw:
            child_x = df['%s_Xposition'%c][frame]
            child_y = df['%s_Yposition'%c][frame]
            ax.plot([parent_x, child_x], [parent_y, child_y], 'k-', lw=2)
            
        if draw_names:
            ax.annotate(joint, 
                    (df['%s_Xposition'%joint][frame] + 0.1, 
                     df['%s_Yposition'%joint][frame] + 0.1))

    return ax

def draw_stickfigure3d(mocap_track, frame, data=None, joints=None, draw_names=False, ax=None, figsize=(8,8)):
    from mpl_toolkits.mplot3d import Axes3D
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d') 
    
    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints
    
    if data is None:
        df = mocap_track.values
    else:
        df = data
        
    for joint in joints_to_draw:
        parent_x = df['%s_Xposition'%joint][frame]
        parent_y = df['%s_Zposition'%joint][frame]
        parent_z = df['%s_Yposition'%joint][frame]
        # ^ In mocaps, Y is the up-right axis 

        ax.scatter(xs=parent_x, 
                   ys=parent_y,  
                   zs=parent_z,  
                   alpha=0.6, c='b', marker='o')

        
        children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]
        
        for c in children_to_draw:
            child_x = df['%s_Xposition'%c][frame]
            child_y = df['%s_Zposition'%c][frame]
            child_z = df['%s_Yposition'%c][frame]
            # ^ In mocaps, Y is the up-right axis

            ax.plot([parent_x, child_x], [parent_y, child_y], [parent_z, child_z], 'k-', lw=2, c='black')
            
        if draw_names:
            ax.text(x=parent_x + 0.1, 
                    y=parent_y + 0.1,
                    z=parent_z + 0.1,
                    s=joint,
                    color='rgba(0,0,0,0.9')

    return ax


def sketch_move(mocap_track, data=None, ax=None, figsize=(16,8)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    if data is None:
        data = mocap_track.values

    for frame in range(0, data.shape[0], 4):
#         draw_stickfigure(mocap_track, f, data=data, ax=ax)
        
        for joint in mocap_track.skeleton.keys():
            children_to_draw = [c for c in mocap_track.skeleton[joint]['children']]
            
            parent_x = data['%s_Xposition'%joint][frame]
            parent_y = data['%s_Yposition'%joint][frame]
            
            frame_alpha = frame/data.shape[0]
            
            for c in children_to_draw:
                child_x = data['%s_Xposition'%c][frame]
                child_y = data['%s_Yposition'%c][frame]
                
                ax.plot([parent_x, child_x], [parent_y, child_y], '-', lw=1, color='gray', alpha=frame_alpha)

def render_mp4(mocap_track, filename, data=None, ax=None, axis_scale=50, elev=45, azim=45,
               dpi=50, figsize=(10,10), ground_res=11, outline=True,
               extra_args=('-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p')):
    """Render a skeleton animation to an mp4.

    Parameters
    ----------
    mocap_track : MocapData
        Track holding joint positions (run MocapParameterizer('position') first).
    filename : str or None
        Output mp4 path. Nothing is written when None.
    data : pandas.DataFrame, optional
        Position values to draw. Defaults to mocap_track.values.
    ax : matplotlib 3d axes, optional
        Draw into an existing axes instead of building a new figure.
    axis_scale : float
        Half-extent of the plotted volume.
    elev, azim : float
        Camera elevation and azimuth.
    dpi : int
        Output dots-per-inch. With figsize=(10,10) the default gives 500x500.
    figsize : tuple of float
        Figure size in inches.
    ground_res : int
        Number of grid lines per side of the ground plane. It is re-projected
        on every frame, so a dense grid is expensive.
    outline : bool
        Draw a black stroke around each limb. Costs ~20% of the render time.
    extra_args : sequence of str
        Passed to ffmpeg. The default trades file size for encoding speed.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-axis_scale, axis_scale)
        ax.set_zlim3d( 0, axis_scale)
        ax.set_ylim3d(-axis_scale, axis_scale)
        ax.grid(True)
        ax.set_axis_off()

        ax.view_init(elev=elev, azim=azim)

        xs = np.linspace(-200, 200, ground_res)
        ys = np.linspace(-200, 200, ground_res)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros(X.shape)

        wframe = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='grey',lw=0.2)
    else:
        fig = ax.get_figure()

    if data is None:
        data = mocap_track.values

    fps=int(np.round(1/mocap_track.framerate))
    joints = list(mocap_track.skeleton.keys())
    effects = [pe.Stroke(linewidth=3, foreground='black'), pe.Normal()] if outline else None
    lines = [ax.plot([0,0], [0,0], [0,0], color='red',
        lw=2, path_effects=effects)[0] for _ in joints]

    # Pull the joint positions into one dense array up front. Indexing the
    # DataFrame per joint per frame inside animate() dominated the render.
    xyz = np.empty((data.shape[0], len(joints), 3))
    for j, joint in enumerate(joints):
        xyz[:, j, 0] = data['%s_Xposition'%joint].to_numpy()
        xyz[:, j, 1] = data['%s_Yposition'%joint].to_numpy()
        xyz[:, j, 2] = data['%s_Zposition'%joint].to_numpy()
    children = [[joints.index(c) for c in mocap_track.skeleton[joint]['children']]
                for joint in joints]

    def animate(frame):
        pts = xyz[frame]
        for j, children_to_draw in enumerate(children):
            for c in children_to_draw:
                lines[j].set_data(np.array([[pts[c,0], pts[j,0]],[-pts[c,2],-pts[j,2]]]))
                lines[j].set_3d_properties(np.array([pts[c,1], pts[j,1]]))

    plt.tight_layout()

    if filename != None:
        # Drive the writer directly. FuncAnimation redraws the canvas once for
        # the animation and again when saving each frame, doubling the work.
        writer = animation.FFMpegWriter(fps=fps, codec='h264',
                                        extra_args=list(extra_args))
        with writer.saving(fig, filename, dpi):
            for frame in range(data.shape[0]):
                animate(frame)
                writer.grab_frame()
        plt.close(fig)


def viz_cnn_filter(feature_to_viz, mocap_track, data, gap=25):
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot2grid((1,8),(0,0))
    ax.imshow(feature_to_viz.T, aspect='auto', interpolation='nearest')
    
    ax = plt.subplot2grid((1,8),(0,1), colspan=7)
    for frame in range(feature_to_viz.shape[0]):
        frame_alpha = 0.2#frame/data.shape[0] * 2 + 0.2

        for joint_i, joint in enumerate(mocap_track.skeleton.keys()):
            children_to_draw = [c for c in mocap_track.skeleton[joint]['children']]

            parent_x = data['%s_Xposition'%joint][frame] + frame * gap
            parent_y = data['%s_Yposition'%joint][frame] 

            ax.scatter(x=parent_x, 
                       y=parent_y,  
                       alpha=0.6,
                       cmap='RdBu',
                       c=feature_to_viz[frame][joint_i] * 10000,
                       marker='o',
                       s = abs(feature_to_viz[frame][joint_i] * 10000))
            plt.axis('off')
            for c in children_to_draw:
                child_x = data['%s_Xposition'%c][frame] + frame * gap
                child_y = data['%s_Yposition'%c][frame] 

                ax.plot([parent_x, child_x], [parent_y, child_y], '-', lw=1, color='gray', alpha=frame_alpha)

                   
def print_skel(X):
    stack = [X.root_name]
    tab=0
    while stack:
        joint = stack.pop()
        tab = len(stack)
        print('%s- %s (%s)'%('| '*tab, joint, X.skeleton[joint]['parent']))
        for c in X.skeleton[joint]['children']:
            stack.append(c)


# def nb_play_mocap_fromurl(mocap, mf, frame_time=1/30, scale=1, base_url='http://titan:8385'):
    # if mf == 'bvh':
        # bw = BVHWriter()
        # with open('test.bvh', 'w') as ofile:
            # bw.write(mocap, ofile)
        
        # filepath = '../notebooks/test.bvh'
    # elif mf == 'pos':
        # c = list(mocap.values.columns)

        # for cc in c:
            # if 'rotation' in cc:
                # c.remove(cc)
        # mocap.values.to_csv('test.csv', index=False, columns=c)
        
        # filepath = '../notebooks/test.csv'
    # else:
        # return
    
    # url = '%s/mocapplayer/player.html?data_url=%s&scale=%f&cz=200&order=xzyi&frame_time=%f'%(base_url, filepath, scale, frame_time)
    # iframe = '<iframe src=' + url + ' width="100%" height=500></iframe>'
    # link = '<a href=%s target="_blank">New Window</a>'%url
    # return IPython.display.HTML(iframe+link)

# def nb_play_mocap(mocap, mf, meta=None, frame_time=1/30, scale=1, camera_z=500, base_url=None):
    # data_template = 'var dataBuffer = `$$DATA$$`;'
    # data_template += 'var metadata = $$META$$;'
    # data_template += 'start(dataBuffer, metadata, $$CZ$$, $$SCALE$$, $$FRAMETIME$$);'
    # dir_path = os.path.dirname(os.path.realpath(__file__))


    # if base_url is None:
        # base_url = os.path.join(dir_path, 'mocapplayer/playBuffer.html')
    
    # # print(dir_path)

    # if mf == 'bvh':
        # pass
    # elif mf == 'pos':
        # cols = list(mocap.values.columns)
        # for c in cols:
            # if 'rotation' in c:
                # cols.remove(c)
        
        # data_csv = mocap.values.to_csv(index=False, columns=cols)

        # if meta is not None:
            # lines = [','.join(item) for item in meta.astype('str')]
            # meta_csv = '[' + ','.join('[%s]'%l for l in lines) +']'            
        # else:
            # meta_csv = '[]'
        
        # data_assigned = data_template.replace('$$DATA$$', data_csv)
        # data_assigned = data_assigned.replace('$$META$$', meta_csv)
        # data_assigned = data_assigned.replace('$$CZ$$', str(camera_z))
        # data_assigned = data_assigned.replace('$$SCALE$$', str(scale))
        # data_assigned = data_assigned.replace('$$FRAMETIME$$', str(frame_time))

    # else:
        # return
    
    

    # with open(os.path.join(dir_path, 'mocapplayer/data.js'), 'w') as oFile:
        # oFile.write(data_assigned)

    # url = '%s?&cz=200&order=xzyi&frame_time=%f&scale=%f'%(base_url, frame_time, scale)
    # iframe = '<iframe frameborder="0" src=' + url + ' width="100%" height=500></iframe>'
    # link = '<a href=%s target="_blank">New Window</a>'%url
    # return IPython.display.HTML(iframe+link)
    