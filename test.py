import fn
import matplotlib.pyplot as plt
# valid_point = [6, 8, 10, 12, 14, 16]  # right pitcher
valid_point = [5, 7, 9, 11, 13, 15]  # left pitcher
# valid_point = [5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]
basic_dataset = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20/B20_human20/B20_human20_'
aim_dataset = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20/B20_human367/B20_human367_'
basic_camera_dataset = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/NTUS_'
aim_camera_dataset = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/NTUS_'
basic_path = basic_dataset + 'threeDdata_post.csv'
aim_path = aim_dataset + 'threeDdata_post.csv'
basic_video = [basic_dataset+'Camera_1_reprojection_post.avi', basic_dataset+'Camera_2_reprojection_post.avi', basic_dataset+'Camera_3_reprojection_post.avi']
aim_video = [aim_dataset+'Camera_1_reprojection_post.avi', aim_dataset+'Camera_2_reprojection_post.avi', aim_dataset+'Camera_3_reprojection_post.avi']
basic_mtx = [basic_camera_dataset+'Camera_1_checkerboard_mtx.npy', basic_camera_dataset+'Camera_2_checkerboard_mtx.npy', basic_camera_dataset+'Camera_3_checkerboard_mtx.npy']
basic_dist = [basic_camera_dataset+'Camera_1_checkerboard_dist.npy', basic_camera_dataset+'Camera_2_checkerboard_dist.npy', basic_camera_dataset+'Camera_3_checkerboard_dist.npy']
basic_R = [basic_camera_dataset+'relational_0_R.npy', basic_camera_dataset+'relational_1_R.npy']
basic_T = [basic_camera_dataset+'relational_0_T.npy', basic_camera_dataset+'relational_1_T.npy']
basic_transformation =[basic_camera_dataset+'transformation.npy']
aim_mtx = [aim_camera_dataset+'Camera_1_checkerboard_mtx.npy', aim_camera_dataset+'Camera_2_checkerboard_mtx.npy', aim_camera_dataset+'Camera_3_checkerboard_mtx.npy']
aim_dist = [aim_camera_dataset+'Camera_1_checkerboard_dist.npy', aim_camera_dataset+'Camera_2_checkerboard_dist.npy', aim_camera_dataset+'Camera_3_checkerboard_dist.npy']
aim_R = [aim_camera_dataset+'relational_0_R.npy', aim_camera_dataset+'relational_1_R.npy']
aim_T = [aim_camera_dataset+'relational_0_T.npy', aim_camera_dataset+'relational_1_T.npy']
aim_transformation = [aim_camera_dataset+'transformation.npy']
loss_set, shift_basic_frame, shift_aim_frame, H = fn.joint_analysis().action_difference(basic_path=basic_path, aim_path=aim_path, valid_point=valid_point)

plt.plot(loss_set)
plt.xlabel('frame')
plt.ylabel('loss')
plt.show()
fn.joint_analysis().mix_video(output='./test/', basic_path=basic_path, basic_video=basic_video, aim_path=aim_path, aim_video=aim_video, valid_point=valid_point, shift_basic_frame=shift_basic_frame, shift_aim_frame=shift_aim_frame, ICP=H, basic_mtx=basic_mtx, basic_dist=basic_dist, basic_R=basic_R, basic_T=basic_T, basic_transformation=basic_transformation, aim_mtx=aim_mtx, aim_dist=aim_dist, aim_R=aim_R, aim_T=aim_T, aim_transformation=aim_transformation, UI=False)









