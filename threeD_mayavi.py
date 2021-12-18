import os
import time
import logging
import numpy as np
import pandas as pd
import anytree as antr
import mayavi.mlab as mlab
from opt_mayavi import opt as opt_mayavi

class SkeletonTree:

    def __init__(self, node_data=None, ifRecord=False):
        self.figure = mlab.figure(size=(800, 600))
        # self.view = mlab.view
        self.engine = mlab.get_engine()
        self.scene = self.figure.scene
        self.scene.movie_maker.record = ifRecord
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = float(opt_mayavi.xrange[0]), float(opt_mayavi.xrange[1]), float(opt_mayavi.yrange[0]), float(opt_mayavi.yrange[1]), float(opt_mayavi.zrange[0]), float(opt_mayavi.zrange[1])
        self.view = [float(opt_mayavi.view[0]), float(opt_mayavi.view[1])]
        self.node_tree = antr.AnyNode(parent=None, name="root")
        if node_data is not None:
            self.updateNodeData(node_data)

    def updateNodeData(self, node_data):
        for node in antr.LevelOrderIter(self.node_tree, filter_=lambda n: n.name != "root"):
            node.x, node.y, node.z = node_data["{}_x [m]".format(node.name)], node_data["{}_y [m]".format(node.name)], node_data["{}_z [m]".format(node.name)]

class AlphaPoseWithCoccygis(SkeletonTree):

    def __init__(self, node_data=None, ifRecord=False):
        LOG.debug("initialize AlphaPoseWithCOM(SkeletonTree)")
        super().__init__(node_data=node_data, ifRecord=ifRecord)
        self.vicon_factor = 3
        self.scale_factor = 0.05
        self.generateSkeleton()
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

    def initializeMayaviScene(self):
        zero_vicon_data = np.zeros(self.node_number * self.vicon_factor)
        self.quiver3d = mlab.quiver3d(zero_vicon_data.copy(), zero_vicon_data.copy(), zero_vicon_data.copy(), zero_vicon_data.copy(), zero_vicon_data.copy(), zero_vicon_data.copy(), scale_factor=self.scale_factor, name="Vicon_vector")
        self.quiver3d.mlab_source.dataset.lines = self.line_data
        mlab.pipeline.surface(self.quiver3d, color=(1, 1, 1), representation="wireframe", line_width=4, name="Skeleton_line")
        mlab.pipeline.glyph(self.quiver3d, color=(0, 0, 0), scale_factor=self.scale_factor, name="Skeleton_node")
        mlab.axes(extent=[self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax], nb_labels=5)
        mlab.outline(extent=[self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax])
        mlab.view(azimuth=270+self.view[0], elevation=90+self.view[1], distance=5.0)
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

    def updateNodeData(self, node_data):
        LOG.debug("frame: {}".format(node_data["frame"]))
        node_data = pd.concat([node_data, pd.Series([(node_data["11_x [m]"] + node_data["12_x [m]"]) * 0.5, (node_data["11_y [m]"] + node_data["12_y [m]"]) * 0.5, (node_data["11_z [m]"] + node_data["12_z [m]"]) * 0.5], index=["18_x [m]", "18_y [m]", "18_z [m]"])])
        for node in antr.LevelOrderIter(self.node_tree, filter_=lambda n: n.name != "root"):
            source_index = int(node.name) * self.vicon_factor
            node.point[:] = node_data["{}_x [m]".format(node.name)], node_data["{}_y [m]".format(node.name)], node_data["{}_z [m]".format(node.name)]
            self.quiver3d.mlab_source.points[source_index:source_index + 3, :] = node.point, node.point, node.point
        for node in antr.LevelOrderIter(self.node_tree, filter_=lambda n: n.name != "root" and n.vicon_node):
            source_index = int(node.name) * self.vicon_factor
            self.quiver3d.mlab_source.vectors[source_index:source_index + 3, :] = self.calculateViconVectors(node)
        self.quiver3d.mlab_source.dataset.set(points=self.quiver3d.mlab_source.points, trait_change_notify=False)

@mlab.animate(delay=1, ui=opt_mayavi.UI)

def animate(dataframes, skeleton):
    t_start = time.clock()
    for dataframe in dataframes.iterrows():
        skeleton.updateNodeData(dataframe[1])
        yield
    t_end = time.clock()
    LOG.debug("FPS: {}".format(dataframes.to_numpy().shape[0]/(t_end - t_start)))
    os._exit(os.EX_OK)

if __name__ == "__main__":
    LOG = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    DATA_PATH = opt_mayavi.path
    dataframes = pd.read_csv(os.path.abspath(DATA_PATH))
    skeleton = AlphaPoseWithCoccygis(ifRecord=opt_mayavi.record)
    anim = animate(dataframes, skeleton)
    mlab.show()


