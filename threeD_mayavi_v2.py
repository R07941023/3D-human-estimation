import os
import time
import logging
import cv2
import numpy as np
import pandas as pd
import anytree as antr
import mayavi.mlab as mlab
from tvtk.api import tvtk
from tvtk.common import configure_input
from opt_mayavi import opt as opt_mayavi

class SkeletonTree:
    def __init__(self, node_data=None, ifRecord=False):
        self.images = []
        self.figure = mlab.figure(size=(800, 600))
        self.engine = mlab.get_engine()
        self.scene = self.figure.scene
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = float(opt_mayavi.xrange[0]), float(opt_mayavi.xrange[1]), float(opt_mayavi.yrange[0]), float(opt_mayavi.yrange[1]), float(opt_mayavi.zrange[0]), float(opt_mayavi.zrange[1])
        self.view = [float(opt_mayavi.view[0]), float(opt_mayavi.view[1])]
        self.node_tree = antr.AnyNode(parent=None, name="root")
        if node_data is not None:
            self.updateNodeData(node_data)
        self.ifRecord = ifRecord
        if self.ifRecord:
            self.writer = tvtk.PNGWriter(write_to_memory=True)

    def scene2Img(self):
        configure_input(self.writer, self.scene._get_window_to_image())
        self.writer.write()
        return cv2.imdecode(self.writer.result.to_array(), cv2.IMREAD_COLOR)

    def saveVideo(self, file, size, fps, codec=cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')):
        LOG.debug("Saving video ...")
        i = 0
        file = os.path.abspath(file)
        video = cv2.VideoWriter(file, codec, fps, size)
        for image in self.images:
            image = cv2.rectangle(image, (250, 10), (550, 60), (0, 0, 0), -1)
            cv2.putText(image, 'NTU-YR-LAB', (270, 45), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, 'frame= ' + str(i + 1), (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'ball velocity= ' + opt_mayavi.ball_velocity, (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1,cv2.LINE_AA)
            video.write(image)
            i += 1
        video.release()
        LOG.debug("Saving video ... OK!")

    def saveImageSeries(self, file):
        LOG.debug("Saving images ...")
        file = os.path.abspath(file)
        file_iter = file.split(".png")
        count = 1
        file_iter += [str(count), ".png"]
        for image in self.images:
            file_iter[-2] = str(count)
            cv2.imwrite("".join(file_iter), image)
            count += 1
        LOG.debug("Saving images ... OK!")

    def updateNodeData(self, node_data):
        for node in antr.LevelOrderIter(self.node_tree, filter_=lambda n: n.name != "root"):
            node.x, node.y, node.z = node_data["{}_x [m]".format(node.name)], node_data["{}_y [m]".format(node.name)], node_data["{}_z [m]".format(node.name)]

class AlphaPoseWithCoccygis(SkeletonTree):
    def __init__(self, node_data=None, ifRecord=False):
        LOG.debug("initialize AlphaPoseWithCOM(SkeletonTree)")
        super().__init__(node_data=node_data, ifRecord=ifRecord)
        self.vicon_factor = 3
        self.scale_factor = 0.05
        self.frame_count = 0
        self.generateSkeleton()
        self.initializeMayaviSource()
        self.initializeMayaviScene()

    def generateSkeleton(self):
        LOG.debug("build skeleton node tree")
        node_0 = antr.AnyNode(parent=self.node_tree, name="0", point=np.zeros(3), vicon_node=False)
        node_1 = antr.AnyNode(parent=node_0, name="1", point=np.zeros(3), vicon_node=False)
        node_2 = antr.AnyNode(parent=node_0, name="2", point=np.zeros(3), vicon_node=False)
        node_17 = antr.AnyNode(parent=node_0, name="17", point=np.zeros(3), vicon_node=True, vicon_ref1=None, vicon_ref2=None)
        node_3 = antr.AnyNode(parent=node_1, name="3", point=np.zeros(3), vicon_node=False)
        node_4 = antr.AnyNode(parent=node_2, name="4", point=np.zeros(3), vicon_node=False)
        node_5 = antr.AnyNode(parent=node_17, name="5", point=np.zeros(3), vicon_node=True, vicon_ref1=None, vicon_ref2=None)
        node_6 = antr.AnyNode(parent=node_17, name="6", point=np.zeros(3), vicon_node=True, vicon_ref1=None, vicon_ref2=None)
        node_18 = antr.AnyNode(parent=node_17, name="18", point=np.zeros(3), vicon_node=False)
        node_7 = antr.AnyNode(parent=node_5, name="7", point=np.zeros(3), vicon_node=True, vicon_ref1=None, vicon_ref2=None)
        node_8 = antr.AnyNode(parent=node_6, name="8", point=np.zeros(3), vicon_node=True, vicon_ref1=None, vicon_ref2=None)
        node_9 = antr.AnyNode(parent=node_7, name="9", point=np.zeros(3), vicon_node=False)
        node_10 = antr.AnyNode(parent=node_8, name="10", point=np.zeros(3), vicon_node=False)
        node_11 = antr.AnyNode(parent=node_18, name="11", point=np.zeros(3), vicon_node=True, vicon_ref1=None, vicon_ref2=None)
        node_12 = antr.AnyNode(parent=node_18, name="12", point=np.zeros(3), vicon_node=True, vicon_ref1=None, vicon_ref2=None)
        node_13 = antr.AnyNode(parent=node_11, name="13", point=np.zeros(3), vicon_node=True, vicon_ref1=None, vicon_ref2=None)
        node_14 = antr.AnyNode(parent=node_12, name="14", point=np.zeros(3), vicon_node=True, vicon_ref1=None, vicon_ref2=None)
        node_15 = antr.AnyNode(parent=node_13, name="15", point=np.zeros(3), vicon_node=False)
        node_16 = antr.AnyNode(parent=node_14, name="16", point=np.zeros(3), vicon_node=False)
        LOG.debug("define skeleton vicon reference node")
        node_17.vicon_ref1, node_17.vicon_ref2 = node_5, node_6
        node_5.vicon_ref1, node_5.vicon_ref2 = node_17, node_7
        node_6.vicon_ref1, node_6.vicon_ref2 = node_17, node_8
        node_7.vicon_ref1, node_7.vicon_ref2 = node_5, node_9
        node_8.vicon_ref1, node_8.vicon_ref2 = node_6, node_10
        node_11.vicon_ref1, node_11.vicon_ref2 = node_18, node_13
        node_12.vicon_ref1, node_12.vicon_ref2 = node_18, node_14
        node_13.vicon_ref1, node_13.vicon_ref2 = node_11, node_15
        node_14.vicon_ref1, node_14.vicon_ref2 = node_12, node_16
        LOG.debug("generate line data between node connection")
        self.node_number, self.line_data = 1, []
        for node in antr.LevelOrderIter(self.node_tree, filter_=lambda n: n.name not in ["root", "0"]):
            self.node_number += 1
            self.line_data.append((int(node.name), int(node.parent.name)))
        self.line_data = np.array(self.line_data) * self.vicon_factor

    def initializeMayaviSource(self):
        zero_vicon_data = np.zeros(self.node_number * self.vicon_factor)
        self.quiver3d = mlab.quiver3d(zero_vicon_data.copy(), zero_vicon_data.copy(), zero_vicon_data.copy(), zero_vicon_data.copy(), zero_vicon_data.copy(), zero_vicon_data.copy(), scale_factor=self.scale_factor, name="Vicon_vector")
        self.quiver3d.mlab_source.dataset.lines = self.line_data
        mlab.pipeline.surface(self.quiver3d, color=(1, 1, 1), representation="wireframe", line_width=4, name="Skeleton_line")
        mlab.pipeline.glyph(self.quiver3d, color=(0, 0, 0), scale_factor=self.scale_factor, name="Skeleton_node")
        mlab.axes(extent=[self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax], nb_labels=5)
        mlab.outline(extent=[self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax])


    def initializeMayaviScene(self):
        if opt_mayavi.x_rotate or opt_mayavi.z_rotate:
            self.view[0] += opt_mayavi.x_rotate
            self.view[1] += opt_mayavi.z_rotate
        mlab.view(azimuth=270 + self.view[0], elevation=90 + self.view[1], distance=5.0)
        # self.scene.camera.position = [3.37, 5.42, 2.20]
        # self.scene.camera.focal_point = [0.47, 0.32, -0.11]
        # self.scene.camera.view_angle = 30.0
        # self.scene.camera.view_up = [-0.17, -0.32, 0.93]
        # self.scene.camera.clipping_range = [2.96, 11.51]
        self.scene.camera.compute_view_plane_normal()
        self.scene.render()

    def calculateViconVectors(self, node):
        vector_1 = node.point - node.vicon_ref1.point
        vector_1 /= np.add.reduce(vector_1 ** 2) ** 0.5
        vector_2 = node.point - node.vicon_ref2.point
        vector_2 /= np.add.reduce(vector_2 ** 2) ** 0.5
        vector_3 = np.array([vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1], vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2], vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]])
        vector_3 /= np.add.reduce(vector_3 ** 2) ** 0.5
        return vector_1, vector_2, vector_3

    def addCoccygisPoint(self, node_data, left_point_index=11, right_point_index=12):
        left_point_index *= 3
        right_point_index *= 3
        return np.concatenate((node_data, (node_data[left_point_index: left_point_index + 3] + node_data[right_point_index: right_point_index + 3]) * 0.5), axis=0)

    def updateSkeletonPoints(self, node_data):
        for node in antr.LevelOrderIter(self.node_tree, filter_=lambda n: n.name != "root"):
            source_index = int(node.name) * self.vicon_factor
            node.point[:] = node_data[int(node.name) * 3:int(node.name) * 3 + 3]
            self.quiver3d.mlab_source.points[source_index:source_index + self.vicon_factor, :] = node.point, node.point, node.point

    def updateViconVectors(self):
        for node in antr.LevelOrderIter(self.node_tree, filter_=lambda n: n.name != "root" and n.vicon_node):
            source_index = int(node.name) * self.vicon_factor
            self.quiver3d.mlab_source.vectors[source_index:source_index + self.vicon_factor, :] = self.calculateViconVectors(node)

    def updateSkeletonNodeData(self, node_data):
        if opt_mayavi.x_rotate or opt_mayavi.z_rotate:
            self.initializeMayaviScene()
        self.frame_count += 1
        LOG.debug("frame: {}".format(self.frame_count))
        node_data = self.addCoccygisPoint(node_data, 11, 12)
        self.updateSkeletonPoints(node_data)
        self.updateViconVectors()
        self.quiver3d.mlab_source.dataset.set(points=self.quiver3d.mlab_source.points, trait_change_notify=False)
        if self.ifRecord:
            self.images.append(self.scene2Img())

@mlab.animate(delay=1e-9, ui=opt_mayavi.UI)
def animate(dataframes, skeleton, video_path):
    t_start = time.clock()
    for dataframe in dataframes:
        skeleton.updateSkeletonNodeData(dataframe)
        yield
    if skeleton.ifRecord:
        shape = skeleton.images[-1].shape
        # skeleton.saveImageSeries("./printer/mayavi.png")
        skeleton.saveVideo(video_path, (shape[1], shape[0]), opt_mayavi.fps)
    t_end = time.clock()
    LOG.debug("FPS: {:.2f}".format(dataframes.shape[0] / (t_end - t_start)))
    os._exit(os.EX_OK)



if __name__ == "__main__":
    LOG = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    DATA_PATH = opt_mayavi.path
    dataframes = pd.read_csv(os.path.abspath(DATA_PATH)).values[:, 1:]
    # dataframes = dataframes[:400, :]
    skeleton = AlphaPoseWithCoccygis(ifRecord=opt_mayavi.record)
    anim = animate(dataframes, skeleton, video_path=opt_mayavi.video_path)
    mlab.show()
