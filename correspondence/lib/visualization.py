
c_red = (224. / 255., 0 / 255., 125 / 255.)
c_pink = (224. / 255., 75. / 255., 232. / 255.)
c_blue = (0. / 255., 0. / 255., 255. / 255.)
c_green = (0. / 255., 255. / 255., 0. / 255.)
c_gray1 = (100. / 255., 100. / 255., 100. / 255.)
c_gray2 = (175. / 255., 175. / 255., 175. / 255.)

def viz_flow_mayavi( s_pc,flow = None, s_pc_deformed=None, t_pc=None, scale_factor = 0.02):
    import mayavi.mlab as mlab

    mlab.points3d(s_pc[:, 0], s_pc[:, 1], s_pc[:, 2], scale_factor=scale_factor, color=c_red)

    if flow is not None:
        mlab.quiver3d(s_pc[:, 0], s_pc[:, 1], s_pc[:, 2],
                      flow[:, 0], flow[:, 1], flow[:, 2], scale_factor=1)

    if t_pc is not None:
        mlab.points3d(t_pc[:, 0], t_pc[:, 1], t_pc[:, 2], scale_factor=scale_factor, color=c_blue)

    if s_pc_deformed is not None:
        mlab.points3d(s_pc_deformed[:, 0], s_pc_deformed[:, 1], s_pc_deformed[:, 2], scale_factor=scale_factor, color=c_green)

    mlab.show()


def viz_flow_open3d( s_pc,flow = None, s_pc_deformed=None, t_pc=None, scale_factor = 0.02):
    import open3d as o3d
    import numpy as np

    vis_list = []

    s_pcd_o3d = o3d.geometry.PointCloud()
    s_pcd_o3d.points = o3d.utility.Vector3dVector(s_pc)
    s_pcd_o3d.paint_uniform_color(c_red)
    vis_list.append(s_pcd_o3d)

    if flow is not None:
        # create lines for flow
        lines = []
        colors = []
        for i in range(len(s_pc)):
            lines.append([i, i + len(s_pc)])
            colors.append(c_gray1)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack((s_pc, s_pc + flow)))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis_list.append(line_set)

    if t_pc is not None:
        t_pcd_o3d = o3d.geometry.PointCloud()
        t_pcd_o3d.points = o3d.utility.Vector3dVector(t_pc)
        t_pcd_o3d.paint_uniform_color(c_blue)
        vis_list.append(t_pcd_o3d)

    if s_pc_deformed is not None:
        s_pcd_def_o3d = o3d.geometry.PointCloud()
        s_pcd_def_o3d.points = o3d.utility.Vector3dVector(s_pc_deformed)
        s_pcd_def_o3d.paint_uniform_color(c_green)
        vis_list.append(s_pcd_def_o3d)

    o3d.visualization.draw_geometries(vis_list)



def viz_coarse_nn_correspondence_mayavi(s_pc, t_pc, correspondence, f_src_pcd=None, f_tgt_pcd=None, scale_factor = 0.02):
    '''
    @param s_pc:  [S,3]
    @param t_pc:  [T,3]
    @param correspondence: [2,K]
    @param f_src_pcd: [S1,3]
    @param f_tgt_pcd: [T1,3]
    @param scale_factor:
    @return:
    '''
    import mayavi
    import mayavi.mlab as mlab

    if f_src_pcd is not None:
        mlab.points3d(f_src_pcd[:, 0], f_src_pcd[:, 1], f_src_pcd[:, 2], scale_factor=scale_factor * 0.25, color=c_gray1)
    else:
        mlab.points3d(s_pc[:, 0], s_pc[:, 1], s_pc[:, 2], scale_factor=scale_factor*0.75, color=c_gray1)

    if f_tgt_pcd is not None:
        mlab.points3d(f_tgt_pcd[:, 0], f_tgt_pcd[:, 1], f_tgt_pcd[:, 2], scale_factor=scale_factor * 0.25, color=c_gray2)
    else :
        mlab.points3d(t_pc[:, 0], t_pc[:, 1], t_pc[:, 2], scale_factor=scale_factor*0.75, color=c_gray2)

    s_cpts = s_pc[correspondence[0]]
    t_cpts = t_pc[correspondence[1]]
    flow =  t_cpts-s_cpts

    mlab.points3d(s_cpts[:, 0], s_cpts[:, 1], s_cpts[:, 2], scale_factor=scale_factor , color=c_red)
    mlab.points3d(t_cpts[:, 0], t_cpts[:, 1], t_cpts[:, 2], scale_factor=scale_factor , color=c_blue)
    mlab.quiver3d(s_cpts[:, 0], s_cpts[:, 1], s_cpts[:, 2], flow[:, 0], flow[:, 1], flow[:, 2],
                  scale_factor=1, mode='2ddash', line_width=1.)

    mlab.show()

def viz_coarse_nn_correspondence_open3d(s_pc, t_pc, correspondence, f_src_pcd=None, f_tgt_pcd=None, scale_factor = 0.02):
    import open3d as o3d
    import numpy as np

    vis_list = []
    if f_src_pcd is not None:
        s_pcd_o3d = o3d.geometry.PointCloud()
        s_pcd_o3d.points = o3d.utility.Vector3dVector(f_src_pcd)
        s_pcd_o3d.paint_uniform_color(c_gray1)
    else:
        s_pcd_o3d = o3d.geometry.PointCloud()
        s_pcd_o3d.points = o3d.utility.Vector3dVector(s_pc)
        s_pcd_o3d.paint_uniform_color(c_gray1)
    if f_tgt_pcd is not None:
        t_pcd_o3d = o3d.geometry.PointCloud()
        t_pcd_o3d.points = o3d.utility.Vector3dVector(f_tgt_pcd)
        t_pcd_o3d.paint_uniform_color(c_gray2)
    else:
        t_pcd_o3d = o3d.geometry.PointCloud()
        t_pcd_o3d.points = o3d.utility.Vector3dVector(t_pc)
        t_pcd_o3d.paint_uniform_color(c_gray2)
    vis_list.append(s_pcd_o3d)
    vis_list.append(t_pcd_o3d)

    s_cpts = s_pc[correspondence[0]]
    t_cpts = t_pc[correspondence[1]]
    flow = t_cpts - s_cpts
    s_cpts_o3d = o3d.geometry.PointCloud()
    s_cpts_o3d.points = o3d.utility.Vector3dVector(s_cpts)
    s_cpts_o3d.paint_uniform_color(c_red)
    t_cpts_o3d = o3d.geometry.PointCloud()
    t_cpts_o3d.points = o3d.utility.Vector3dVector(t_cpts)
    t_cpts_o3d.paint_uniform_color(c_blue)
    vis_list.append(s_cpts_o3d)
    vis_list.append(t_cpts_o3d)

    # create lines for flow
    lines = []
    colors = []
    for i in range(len(s_cpts)):
        lines.append([i, i + len(s_cpts)])
        colors.append(c_gray1)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack((s_cpts, s_cpts + flow)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis_list.append(line_set)
    o3d.visualization.draw_geometries(vis_list)