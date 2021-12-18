import fn
import numpy as np
field = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/'


camera_int_path, camera_dist_path, rvec_path, tvec_path = [], [], [], []
tr_mtx = np.load(field+'NTUS_transformation.npy')
print(tr_mtx)
camera_int_path.append(field+'NTUS_Camera_1_checkerboard_mtx.npy')
camera_int_path.append(field+'NTUS_Camera_2_checkerboard_mtx.npy')
camera_int_path.append(field+'NTUS_Camera_3_checkerboard_mtx.npy')
camera_dist_path.append(field+'NTUS_Camera_1_checkerboard_dist.npy')
camera_dist_path.append(field+'NTUS_Camera_2_checkerboard_dist.npy')
camera_dist_path.append(field+'NTUS_Camera_3_checkerboard_dist.npy')
rvec_path.append(field+'NTUS_relational_0_R.npy')
rvec_path.append(field+'NTUS_relational_1_R.npy')
tvec_path.append(field+'NTUS_relational_0_T.npy')
tvec_path.append(field+'NTUS_relational_1_T.npy')
coord = np.array((1.0237046, 0.1557803, 0.3311974, 1))  # the refer coordinates from R camera
c1 = (np.linalg.inv(tr_mtx).dot(coord))[0:3]
# print(c1)
# print(type(c1))
c1_ = np.append(c1,1)
c1_ = tr_mtx.dot(c1_)[0:3]
# print(tr_mtx.dot(c1_)[0:3])
c2 = coord.dot(np.linalg.inv(tr_mtx))[0:3]


img_coord_set = []
img_coord_set.append([529.47, 246.36])
img_coord_set.append([318.47, 172.18])
img_coord_set.append([177.23, 171.17])

print(c1)
print(c1_)

loss1 = fn.camera_orientation().multi_projection_loss(c1, img_coord_set, camera_int_path, camera_dist_path, rvec_path, tvec_path)
loss2 = fn.camera_orientation().multi_projection_loss(c1_, img_coord_set, camera_int_path, camera_dist_path, rvec_path, tvec_path)

print(loss1)
print(loss2)