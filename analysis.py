import matplotlib.pyplot as plt
import numpy as np
import fn
import os
from itertools import combinations

class data_analysis(object):

    def __init__(self):  # Run it once
        pass

    def scores2prop(self, data, threshold=0.05, pooling=True):
        n_set = []
        for mini_data in data:
            n_set.append(mini_data.shape[0])
        n = min(n_set)
        align_data = []
        for mini_data in data:
            align_data.append(mini_data[:n, :])
        align_data = np.average(np.array(align_data), 0)
        prop = align_data-threshold
        prop[prop >= 0] = 1
        prop[prop < 0] = 0
        if pooling:
            prop = np.sum(prop, 0)/n
            return prop
        else:
            return prop

    def plot_prop(self, angle, prop, total_prop, point_set=[0], threshold_set=[0]):
        threshold_data_set, point_data_set = {}, {}
        threshold_data_set['angle'], point_data_set['angle'] = np.array(angle), np.array(angle)
        for j in range(prop.shape[0]):
            for i in range(prop.shape[2]):
                if i in point_set:
                    pass
                    # point_data_set['threshold='+str(threshold_set[j])+' + point='+str(i)] = prop[j, :, i]
                    # plt.title(' Point: '+str(i))
                    # plt.xlabel('angle')
                    # plt.ylabel('rate')
                    # plt.plot(angle, prop[j, :, i])
                    # min_index = np.argwhere(prop[j, :, i]==np.min(prop[j, :, i])).reshape(-1)[0]
                    # print('point is at = ', i, '. Min loss is at: ', angle[min_index])
                    # plt.show()
            threshold_data_set[str(threshold_set[j])] = total_prop[j]
            plt.plot(angle, total_prop[j], label='threshold='+str(threshold_set[j]))
            for hh in range(len(angle)):
                print(angle[hh], total_prop[j][hh])
        self.dict2csv(dict=point_data_set, path='./alhpapose_point.csv')
        self.dict2csv(dict=threshold_data_set, path='./alphapose_all.csv')
        plt.title(' Total')
        plt.xlabel('angle')
        plt.ylabel('rate')
        plt.legend(loc='best')
        plt.show()

    def fitting_angle(self, path, threshold_set=[0.05], point_set=[0], camera_n=[2, 3]):
        for threshold in threshold_set:
            print('threshold = ', threshold)
            angle_prop = {}
            for path_index in path:
                time_data = []
                for mini_path in path[path_index]:
                    _, mini_data = fn.tool().Loadcsv_2d_scores(mini_path)
                    time_data.append(mini_data)
                mini_prop = self.scores2prop(data=time_data, threshold=threshold, pooling=False)
                angle_prop[path_index] = mini_prop
            # for xx in angle_prop:
            #     print(xx, angle_prop[xx].shape)
            print('camera assessment...')
            for mini_camera in camera_n:
                rate = {}
                camera_combins = [c for c in combinations(list(angle_prop.keys()), mini_camera)]
                for camera_set in camera_combins:
                    angle_prop_set = []
                    for camera in camera_set:
                        angle_prop_set.append(angle_prop[camera])
                    rate[camera_set] = np.average(self.scores2prop(data=angle_prop_set, threshold=2/mini_camera, pooling=True))
                camera_rate = sorted(rate.items(), key=lambda item:item[1], reverse=True)
                rate = {}
                for mini_camera_rate in camera_rate:
                    rate[str(mini_camera_rate[0])] = mini_camera_rate[1]
                self.dict2csv(dict=rate, path='./fitting_C'+str(mini_camera)+'_T'+str(int(threshold*100))+'.csv')
                print('camera number=', mini_camera, ' - ', rate)

    def period_prop(self, data, threshold, period, point_set=[6, 8, 10, 11, 13, 15, 5, 7, 9, 12, 14, 16]):
        prop = data - threshold
        prop[prop >= 0] = 1
        prop[prop < 0] = 0
        for mini_period in period:
            for point in point_set:
                tmp = prop[mini_period[0]:mini_period[1], point]
                tmp = np.sum(tmp)/tmp.shape[0]

    def scores_2d(self, path, threshold_set=[0.05], point_set=[0]):
        threshold_prop = []
        for threshold in threshold_set:
            angle_sequence, angle_prop = [], []
            for path_index in path:
                angle_sequence.append(path_index)
                time_prop = []
                for mini_path in path[path_index]:
                    _, mini_data = fn.tool().Loadcsv_2d_scores(mini_path)
                    # period_prop = self.period_prop(data=mini_data, threshold=threshold, period=[[1, 40], [41, 45], [46, 47], [48, 63]])
                    mini_prop = self.scores2prop(data=[mini_data], threshold=threshold)
                    time_prop.append(mini_prop)
                time_prop = np.average(np.array(time_prop), 0)
                angle_prop.append(time_prop)
            angle_prop = np.array(angle_prop)
            threshold_prop.append(angle_prop)
        threshold_prop = np.array(threshold_prop).reshape(-1, angle_prop.shape[0], angle_prop.shape[1])
        total_prop = np.average(threshold_prop[:, :, point_set], 2)
        self.plot_prop(angle_sequence, threshold_prop, total_prop, point_set=point_set, threshold_set=threshold_set)

    def loss_case(self, base_dataset, folder_aim_path, dim=3, point_set=[10], camera_set=[1, 2], padding=1e10):
        loss = []
        for aim_path in folder_aim_path:
            n, temp_loss = 0, 0
            aim_dataset = fn.tool().Loadcsv_project(folder_aim_path[aim_path], dim=dim, camera_set=camera_set)
            for aim_prject in aim_dataset:
                if aim_prject in base_dataset:
                    for camera in range(base_dataset[aim_prject].shape[0]):
                        base = base_dataset[aim_prject][camera][:, point_set, :].copy()
                        aim = aim_dataset[aim_prject][camera][:, point_set, :].copy()
                        if dim == 3:
                            base[base >= padding / 2] = 0
                            aim[aim >= padding / 2] = 0
                        elif dim == 2:
                            base[base == -1] = 0
                            aim[aim == -1] = 0
                            base[np.abs(base) >= 1e4] = 0
                            aim[np.abs(aim) >= 1e4] = 0

                        # if n == 3:
                        #     for xx in range(2):
                        #         plt.plot(aim[:, 1, xx])
                        #         plt.show()
                        #     exit()

                        d = (base - aim).reshape(aim_dataset[aim_prject].shape[1], -1, aim_dataset[aim_prject].shape[3])

                        # plt.title(str(n))
                        # plt.plot(d[:, 1])
                        # plt.show()

                        temp_loss += np.sum(np.sqrt(np.sum(np.power(d, 2), 2))) / (d.shape[0] * d.shape[1])
                        n += 1
            loss.append(temp_loss / n)
        loss = np.array(loss)
        return loss

    def loss_frame(self, base_dataset, folder_aim_path, dim=3, point_set=[10], camera_set=[1, 2], padding=1e10):
        loss = {}
        for aim_path in folder_aim_path:
            aim_dataset = fn.tool().Loadcsv_project(folder_aim_path[aim_path], dim=dim, camera_set=camera_set)
            n = 0
            for aim_prject in aim_dataset:
                if aim_prject in base_dataset:
                    if n == 0:
                        temp_loss = np.zeros((base_dataset[aim_prject].shape[1], 1))
                    for camera in range(base_dataset[aim_prject].shape[0]):
                        base = base_dataset[aim_prject][camera][:, point_set, :].copy()
                        aim = aim_dataset[aim_prject][camera][:, point_set, :].copy()
                        if dim == 3:
                            base[base >= padding / 2] = 0
                            aim[aim >= padding / 2] = 0
                        elif dim == 2:
                            base[np.abs(base) >= 1e4] = 0
                            aim[np.abs(aim) >= 1e4] = 0
                            # base[base >= 1000 / 2] = 0
                            # aim[aim >= 1000 / 2] = 0
                        # uniform length
                        d = (base - aim).reshape(aim_dataset[aim_prject].shape[1], -1, aim_dataset[aim_prject].shape[3])
                        temp_loss += np.sqrt(np.sum(np.power(d, 2), 2))
                        n += 1
            temp_loss = temp_loss / n
            loss[aim_path] = temp_loss
        return loss

    def dict2csv(self, dict, path):
        import pandas as pd
        # print(dict)
        title, data = [], []
        for name in dict:
            title.append(name)
            data.append(dict[name])
        title = np.array(title).reshape(-1, 1)
        print(title.shape)
        print(np.array(data).shape)
        data = np.array(data).astype(np.str).reshape(title.shape[0], -1)
        dataframe = np.hstack((title, data)).astype(np.str).T
        df = pd.DataFrame(dataframe)
        df.to_csv(path, encoding="gbk", index=False, header=False)

    def loss_2d3d(self, folder_base_path, folder_aim_path, camera_set=[1, 2], point_set=[10], padding=1e10):
        # 3D
        output_data = {}
        base_dataset = fn.tool().Loadcsv_project(folder_base_path, padding=1e10, dim=3)
        # all
        loss = self.loss_case(base_dataset=base_dataset, folder_aim_path=folder_aim_path, point_set=point_set, padding=padding, dim=3)
        output_data['all'] = loss * 1000
        plt.plot(np.array(loss)*1000, label='all')
        # every point
        for point in point_set:
            loss = self.loss_case(base_dataset=base_dataset, folder_aim_path=folder_aim_path, point_set=point, padding=padding, dim=3)
            output_data[str(point)] = loss * 1000
            plt.plot(np.array(loss) * 1000, label='point '+str(point))
        self.dict2csv(dict=output_data, path='./average_3d.csv')
        plt.title('3D loss')
        plt.xlabel('case')
        plt.ylabel('eucildea distance [mm]')
        plt.legend(loc='upper right')
        plt.show()
        # detail loss for every loss
        for point in point_set:
            print('point = ', point)
            loss = self.loss_frame(base_dataset=base_dataset, folder_aim_path=folder_aim_path, point_set=point, padding=padding)
            for method in loss:
                loss[method] = loss[method] * 1000
                print(method)
                if method == 'fft_cs_poly':
                    print('0-260')
                    print(np.mean(loss[method][:260]))
                    print('260-490')
                    print(np.mean(loss[method][260:490]))
                    print('490-520')
                    print(np.mean(loss[method][490:520]))
                    print('520-545')
                    print(np.mean(loss[method][520:545]))
                    print('545-615')
                    print(np.mean(loss[method][545:615]))
                    print('615-')
                    print(np.mean(loss[method][615:]))
                plt.plot(loss[method], label=method)
            self.dict2csv(dict=loss, path='./loss_'+str(point)+'_3d.csv')
            plt.title('point = '+str(point))
            plt.xlabel('frame')
            plt.ylabel('eucildea distance [mm]')
            plt.legend(loc='best')
            plt.show()
        # # 2D
        # output_data = {}
        # base_dataset = fn.tool().Loadcsv_project(folder_base_path, camera_set=camera_set, dim=2)
        # # all
        # loss = self.loss_case(base_dataset=base_dataset, folder_aim_path=folder_aim_path, point_set=point_set, padding=padding, dim=2, camera_set=camera_set)
        # output_data['all'] = loss
        # plt.plot(np.array(loss), label='all')
        # for point in point_set:
        #     loss = self.loss_case(base_dataset=base_dataset, folder_aim_path=folder_aim_path, point_set=point, padding=padding, dim=2, camera_set=camera_set)
        #     plt.plot(np.array(loss), label='point'+str(point))
        #     output_data[str(point)] = loss
        # plt.title('2D loss')
        # plt.xlabel('case')
        # plt.ylabel('pixel')
        # plt.legend(loc='upper right')
        # plt.show()
        # self.dict2csv(dict=output_data, path='./average_2d.csv')
        # # every point
        # for point in point_set:
        #     loss = self.loss_frame(base_dataset=base_dataset, folder_aim_path=folder_aim_path, point_set=point, padding=padding, dim=2, camera_set=camera_set)
        #     for method in loss:
        #         plt.plot(loss[method], label=method)
        #     plt.title('point = ' + str(point))
        #     plt.xlabel('frame')
        #     plt.ylabel('pixel')
        #     plt.legend(loc='best')
        #     plt.show()
        #     self.dict2csv(dict=loss, path='./loss_'+str(point)+'_2d.csv')

    def compare_signal(self, folder_base_path, folder_aim_path, case, camera_set=[1], point=6, padding=1e10):
        output_data = {}
        axis_name = ['X', 'Y', 'Z']
        # 3D
        axis_set = [0, 1, 2]
        aim_dataset_set = {}
        base_dataset = fn.tool().Loadcsv_project(folder_base_path, padding=padding, dim=3)
        for aim_path in folder_aim_path:
            aim_dataset_set[aim_path] = fn.tool().Loadcsv_project(folder_aim_path[aim_path], padding=padding, dim=3)
        if case in base_dataset:
            for axis in axis_set:
                output_data['3D_' + str(axis_name[axis]) + '_ground truth'] = base_dataset[case][0][:, point, axis]*1000
                plt.plot(base_dataset[case][0][:, point, axis]*1000, label='ground truth')
                for aim_name in aim_dataset_set:
                    if case in aim_dataset_set[aim_name]:
                        output_data['3D_' + str(axis_name[axis]) + '_' + aim_name] = aim_dataset_set[aim_name][case][0][:, point, axis]*1000
                        plt.plot(aim_dataset_set[aim_name][case][0][:, point, axis]*1000, label=aim_name)
                plt.title('3D point = ' + str(point))
                plt.xlabel('frame')
                plt.ylabel(axis_name[axis] + ' eucildea distance [mm]')
                plt.legend(loc='best')
                plt.show()
        else:
            print('Sorry, the case can not be find from dataset')
        # 2D
        axis_set = [0, 1]
        aim_dataset_set = {}
        base_dataset = fn.tool().Loadcsv_project(folder_base_path, camera_set=camera_set, dim=2)
        for aim_path in folder_aim_path:
            aim_dataset_set[aim_path] = fn.tool().Loadcsv_project(folder_aim_path[aim_path], camera_set=camera_set, dim=2)
        if case in base_dataset:
            for camera in range(len(camera_set)):
                for axis in axis_set:
                    output_data['2D_C'+str(camera) + '_' + str(axis_name[axis]) + '_ground truth'] = base_dataset[case][camera][:, point, axis]
                    plt.plot(base_dataset[case][camera][:, point, axis], label='ground truth')
                    for aim_name in aim_dataset_set:
                        if case in aim_dataset_set[aim_name]:
                            output_data['2D_C'+str(camera) + '_' + str(axis_name[axis]) + '_' + aim_name] = aim_dataset_set[aim_name][case][camera][:, point, axis]
                            plt.plot(aim_dataset_set[aim_name][case][camera][:, point, axis], label=aim_name)
                    plt.title('2D C'+str(camera) + ' point = ' + str(point))
                    plt.xlabel('frame')
                    plt.ylabel(axis_name[axis] + ' eucildea distance [mm]')
                    plt.legend(loc='best')
                    plt.show()
        else:
            print('Sorry, the case can not be find from dataset')
        self.dict2csv(dict=output_data, path='./P'+str(point) + '_' + case + '.csv')



# # the filter of analysis for mutil-project
# if __name__ == '__main__':
#     folder_aim_path = {}
#     case = 'B20_human365'  # 365 Z ok
#     point = 15
#     camera_set = [1, 2, 3]
#     folder_base_path = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_gt/'
#     # folder_aim_path['two'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_two/'
#     folder_aim_path['three'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_three/'
#     folder_aim_path['fft'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_fft/'
#     # folder_aim_path['fft_cs'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_fft_cs/'
#     folder_aim_path['fft_cs_poly'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_fft_cs_poly/'
#     # folder_aim_path['fft_cs_poly_nd'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_fft_cs_poly_nd/'
#     data_analysis().compare_signal(folder_base_path=folder_base_path, folder_aim_path=folder_aim_path, point=point, case=case, camera_set=camera_set)
#
#
# # the analysis of pose loss
# if __name__ == '__main__':
#     folder_aim_path = {}
#     folder_base_path = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_gt/'
#     # folder_aim_path['two'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_two/'
#     folder_aim_path['three'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_three/'
#     folder_aim_path['fft'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_fft/'
#     # folder_aim_path['fft_cs'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_fft_cs/'
#     folder_aim_path['fft_cs_poly'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_fft_cs_poly/'
#     # folder_aim_path['fft_cs_poly_nd'] = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/project/B20_compare/B20_fft_cs_poly_nd/'
#     point_set = [15]
#     camera_set = [1, 2, 3]
#     data_analysis().loss_2d3d(folder_base_path=folder_base_path, folder_aim_path=folder_aim_path, point_set=point_set, camera_set=camera_set)



# the analysis of 2d alphapose angle
if __name__ == '__main__':
    path, point_set = {}, []
    threshold_set = [0.25]
    # threshold_set = [0.25]
    # type = 'score'  # score / manual
    type = 'manual'  # score / manual
    max_n = 3  # data times
    folder_path = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200706/video_alphapose/'
    angle_set = [25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295, 305, 315, 325, 335, 345, 355]
    # angle_set = [25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 245, 255, 265, 275, 285, 295, 305, 315, 325, 335, 345, 355]
    # angle_set = [175]
    # point_set.append([6, 8, 10, 11, 13, 15])  # right pitcher
    point_set.append([5, 7, 9, 12, 14, 16])  # left pitcher
    # point_set.append([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # all
    for angle in angle_set:
        for n in range(max_n):
            file_name = folder_path + 'AlphaPose_'+str(angle)+'_'+str(n+1)+'_'+type+'.csv'
            if os.path.isfile(file_name):
                if angle not in path:
                    path[angle] = []
                path[angle].append(file_name)
    for point in point_set:
        data_analysis().scores_2d(path=path, threshold_set=threshold_set, point_set=point)
        # data_analysis().fitting_angle(path=path, threshold_set=threshold_set, point_set=point)


# # plot Plot_baseball_field
# import fn
# if __name__ == '__main__':
#     r = 50  # [m]
#     view = [-95, 0]
#     xlim = [-80, 80]
#     ylim = [-80, 80]
#     zlim = [0, 160]
#     camera_set = np.arange(25, 365, 10)
#     camera_center, camera_length, camera_name, line_camera = [], [], [], []
#     camera_name.append('auxiliary')
#     camera_center.append([0, 0, 0])
#     camera_length.append(0.1)
#     for c in camera_set:
#         camera_name.append('C '+ str(c))
#         camera_center.append([-np.cos(c*np.pi/180)*r, -np.sin(c*np.pi/180)*r, 0])
#         camera_length.append(0.1)
#         line_camera.append(['C '+ str(c), 'auxiliary'])
#     fn.tool().Plot_baseball_field(camera_center, camera_length, camera_name, line_camera, xlim=xlim, ylim=ylim, zlim=zlim, view=view, img_path='./xxx.png')