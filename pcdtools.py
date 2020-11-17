# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:26:30 2019

@author: szabo
"""

import open3d as o3d
import numpy as np
import transformationtools as transformationtools



def project_planar_pcd(pcd_image,z_image=1000):
    xyz_image=np.asarray(pcd_image.points)
    #base_scale=(10000*np.tan(Lidar_FOV*np.pi/180)/(np.max(xyz_image[:,0])-np.min(xyz_image[:,0])))
    scale=z_image/np.max(xyz_image[:,2])
    xyz_image*=scale
    colors=pcd_image.colors
    pcd_image.clear
    pcd_image.points=o3d.utility.Vector3dVector(xyz_image)
    pcd_image.colors=colors
    return pcd_image

def transform_to_origo(vis):
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    pose = np.eye(4)
    cam.extrinsic = pose
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    
#    vis.update_geometry()
#    vis.poll_events()
#    vis.update_renderer()
    return True

#def pcd_from_lidardata(lidar_rows, frameMode=True):
#    Z=np.asarray(lidar_rows['x'].values)
#    X=np.asarray(lidar_rows['y'].values)
#    Y=np.asarray(lidar_rows['z'].values)    
#    
#    xyz=np.vstack((X, Y, Z))
#    xyz=xyz.transpose()
#    
#    pcd = o3d.geometry.PointCloud()
#    pcd.points = o3d.utility.Vector3dVector(xyz)
#    
#    # mirror to image coordinate system
#    if frameMode:
#        pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#    return pcd

def pcd_from_planarimage(frame,f,cx,cy,z_image=2000,voxel_size=10):
    o3d_image = o3d.geometry.Image(frame)
    
    o3d_depth = o3d.geometry.Image(1*np.ones((frame.shape[0],frame.shape[1]),dtype='uint8'))
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth)
    
    pcd_image = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsic(frame.shape[1],frame.shape[0],f,f,cx,cy)))
    pcd_image=project_planar_pcd(pcd_image,z_image=z_image)
    pcd_image = pcd_image.voxel_down_sample(voxel_size=10)

    return pcd_image

class VisualizePCD():
    def __init__(self,width=1024,height=1024,fov=20,intrinsics={'f':333,'cx':150,'cy':200},extrinsics={'angles':(0,0,0),'T':(0,0,0)}):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='pcd', visible=True, width=1024, height=1024)
        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.asarray([0, 0, 0])
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
        self.ctr = self.vis.get_view_control()
        self.vis.add_geometry(mesh_frame)
        self.vis.get_render_option().point_size=2
        self.ctr.change_field_of_view(step=fov)
        self.line_set = o3d.geometry.LineSet()
        self.pcd = o3d.geometry.PointCloud()
        
        self.cx=intrinsics['cx']
        self.cy=intrinsics['cy']
        self.f=intrinsics['f']
        # extrinsics in Lidar coordinate frame
        self.ext_angles=extrinsics['angles']
        self.ext_T=extrinsics['T']
        
        self.prev_pcd=None
    
        
    def __del__(self):
        self.vis.destroy_window()
        self.vis.close()
                
    
    def set_coordinate_and_update(self,pitch=0,yaw=0,roll=0,dx=0,dy=0,dz=0):
        cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        R=transformationtools.euler_matrix(pitch, yaw, roll, axes='rxyz')
        T=transformationtools.translation_matrix([dx,dy,dz])
        R[:,3]=T[:,3]
        cam.extrinsic = R #np.multiply(T,R)
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()
        
#    def remove_geometry(self,pcd):
#        self.vis.remove_geometry(pcd)
        
    def refresh_pcd(self,xyz):
        # pcd axis: z: parallel to image plane, x, horizontal, y vertical
        # convert to camera frame, x to left, y to down
        #pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        self.vis.remove_geometry(self.pcd)
        self.vis.remove_geometry(self.line_set)
        # transform to image plane
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        #pcd.paint_uniform_color([1,0,0])
        #Frame mode
        self.pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        self.vis.add_geometry(self.pcd)

    def filter_with_bbox(self,xyz,bbox):
        X_min=((bbox[0]-self.cx)*xyz[:,2])/self.f
        X_max=((bbox[0]+bbox[2]-self.cx)*xyz[:,2])/self.f

        Y_min=((bbox[1]-self.cy)*xyz[:,2])/self.f
        Y_max=((bbox[1]+bbox[3]-self.cy)*xyz[:,2])/self.f
        
        # filter indices
        filter_indices=np.squeeze(np.argwhere((xyz[:,0]>X_min) & (xyz[:,0]<X_max) & (xyz[:,1]>Y_min) & (xyz[:,1]<Y_max)))
        xyz_filter=xyz[filter_indices]  
        return xyz_filter

    
    def refresh_bbox(self,bbox,Z_max=np.Inf):
                
        xyz=np.asarray(self.pcd.points)
        if Z_max==np.Inf:
            Z_max=max(xyz[:,2])
        X_min=((bbox[0]-self.cx)*Z_max)/self.f
        X_max=((bbox[0]+bbox[2]-self.cx)*Z_max)/self.f

        Y_min=((bbox[1]-self.cy)*Z_max)/self.f
        Y_max=((bbox[1]+bbox[3]-self.cy)*Z_max)/self.f        
        
        points = [[0, 0, 0], [X_min, Y_min, Z_max], [X_min, Y_max, Z_max], [X_max, Y_max, Z_max], [X_max, Y_min, Z_max]]
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        self.line_set.points = o3d.utility.Vector3dVector(points)
        self.line_set.lines = o3d.utility.Vector2iVector(lines)
        self.line_set.colors = o3d.utility.Vector3dVector(colors)
      
        self.vis.add_geometry(self.line_set)
        
    def measure_distance_and_height(self,xyz_filtered,bbox,min_dist=4000,dx=200):
        
        vals,bins=np.histogram(xyz_filtered[xyz_filtered[:,2]<min_dist,2],100)
        object_distance=bins[np.argmax(vals)+1]
        xyz_distance=xyz_filtered[xyz_filtered[:,2]<object_distance,:]
        object_center=(bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2)
        xyz_middle=xyz_distance[(xyz_distance[:,0]>object_center[0]+dx) &
                                (xyz_distance[:,0]<object_center[0]-dx),:]
        
        try:
            if xyz_middle.shape[0]>0:
                object_height=-np.min(xyz_middle[xyz_middle[:,2]<object_distance,1])-self.ext_T[1]
            else:
                object_height=-np.min(xyz_distance[xyz_distance[:,2]<object_distance,1])-self.ext_T[1]
        except:
            object_height=0
        return xyz_distance,object_distance, object_height
        




        
