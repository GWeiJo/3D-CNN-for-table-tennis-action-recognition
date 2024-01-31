# Data_filter is for using different method to cutting length of video, filter out features #
import cv2 as cv
import numpy as np
import csv
from tqdm import trange
##########################################################
TTset_all_data_path = "TTset\TTset_all_data_list.csv"
TTset_tset_path = "TTset\TTset_training_data_list.csv"
TTset_vset_path = "TTset\TTset_validation_data_list.csv"
##########################################################
TTset_all_data_by_choose_path = "TTset\TTset_all_data_by_choose_list.csv"
TTset_tset_by_choose_path = "TTset\TTset_training_data_by_choose_list.csv"
TTset_vset_by_choose_path = "TTset\TTset_validation_data_by_choose_list.csv"
##########################################################
filted_tset_path = "TTset\Filted_TTset_training_data_list.csv"
filted_vset_path = "TTset\Filted_TTset_validation_data_list.csv"
filted_choosed_tset_path = "TTset\Filted_choosed_TTset_training_data_list.csv"
filted_choosed_vset_path = "TTset\Filted_choosed_TTset_validation_data_list.csv"
##########################################################

def split_by_view_type(split_data_by_choose): # data with all calss
    if split_data_by_choose == False: # split training data and validation data by random
        for i in range (1, 5, 1):
            # Split training set
            with open(filted_tset_path, 'r') as file: # Read csv file
                tmp2 = [] # for data list
                rows = csv.reader(file)
                for row in rows:
                    if int(row[3]) == i:
                        tmp1 = [] # for 4 elements
                        tmp1.append(row[0]) # video path
                        tmp1.append(row[1]) # action label
                        tmp1.extend([row[2]]) # number of frames
                        tmp1.extend([row[3]]) # which type
                        tmp2.append(tmp1) # combine 4 elements in 1 element, means 1 data has 4 elements
            file.close()
            with open("TTset\Filted_TTset_training_data_list_view_type{}.csv".format(i), 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for j in range (len(tmp2)):
                    writer.writerow([tmp2[j][0], tmp2[j][1], tmp2[j][2], tmp2[j][3]]) # Save as csv file
            file.close()

            # Split validation set
            with open(filted_vset_path, 'r') as file:
                tmp2 = []
                rows = csv.reader(file)
                for row in rows:
                    if int(row[3]) == i:
                        tmp1 = [] # for 4 elements
                        tmp1.append(row[0]) # video path
                        tmp1.append(row[1]) # action label
                        tmp1.extend([row[2]]) # number of frames
                        tmp1.extend([row[3]]) # which type
                        tmp2.append(tmp1)
            file.close()
            with open("TTset\Filted_TTset_validation_data_list_view_type{}.csv".format(i), 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for j in range (len(tmp2)):
                    writer.writerow([tmp2[j][0], tmp2[j][1], tmp2[j][2], tmp2[j][3]]) # save as csv file
            file.close()


    if split_data_by_choose == True: # data with choosed calss
        for i in range (1, 5, 1):
            # Split training data
            with open(filted_choosed_tset_path, 'r') as file:
                tmp2 = []
                rows = csv.reader(file)
                for row in rows:
                    if int(row[3]) == i:
                        tmp1 = [] # for 4 elements
                        tmp1.append(row[0]) # video path
                        tmp1.append(row[1]) # action label
                        tmp1.extend([row[2]]) # number of frames
                        tmp1.extend([row[3]]) # which type
                        tmp2.append(tmp1)
            file.close()
            with open("TTset\Filted_choosed_TTset_training_data_list_view_type{}.csv".format(i), 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for j in range (len(tmp2)): 
                    writer.writerow([tmp2[j][0], tmp2[j][1], tmp2[j][2], tmp2[j][3]]) # save as csv file
            file.close()
            
            # Split validation data
            with open(filted_choosed_vset_path, 'r') as file:
                tmp2 = []
                rows = csv.reader(file)
                for row in rows:
                    tmp1 = []
                    if int(row[3]) == i:
                        tmp1 = [] # for 4 elements
                        tmp1.append(row[0]) # video path
                        tmp1.append(row[1]) # action label
                        tmp1.extend([row[2]]) # number of frames
                        tmp1.extend([row[3]]) # which type
                        tmp2.append(tmp1)
            file.close()
            with open("TTset\Filted_choosed_TTset_training_data_list_view_type{}.csv".format(i), 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for j in range (len(tmp2)):
                    writer.writerow([tmp2[j][0], tmp2[j][1], tmp2[j][2], tmp2[j][3]]) # save as csv file
            file.close()


def viwe_type(data_path): # split video into different type of view by label of file title
    c_name = data_path[11:17]
    if c_name == "ITTSIN" or c_name == "ITTBUD":
            c_name = data_path[11:23]
    c_name = str(c_name)
    # print(c_name)
    if c_name == "ETCWAW" or c_name == "ETCMUC" or c_name == "CTATPH":
        return int(1) # Type 1
    
    if c_name == "ITTZAG" or c_name == "ITTTUN" or c_name == "ITTSKP" or c_name == "ITTSIN220312" or c_name == "ITTSIN220311" \
    or c_name == "ITTSCL" or c_name == "ITTMCT" or c_name == "ITTBUD220712" or c_name == "ITTBUD220711":
        return int(2) # Type 2
    
    if c_name == "ITTSIN220317" or c_name == "ITTSIN211204" or c_name == "ITTMFM" or c_name == "ITTCTU" or c_name == "ITTBUD220722"\
       or c_name == "ITTBUD220715" or c_name == "ITTBUD220714":
        return int(3) # Type 3
    
    if c_name == "ITTDOH":
        return int(4) # Type 4

def fixed_length_video_cut(data_path, tp): # Cut same length of serving or receiving action video, and abandon the rest
    idx_of_frames = [] 
    for i in trange (len(data_path)):
        tmp1 = []
        count = 0
        vt = viwe_type(data_path[i][0])
        cap = cv.VideoCapture(cv.samples.findFile(data_path[i][0])) # Start to paly the video
        while(1):
            count = count + 1
            ret, frame = cap.read()
            if not ret:
                break
        if tp == 1: # if is serving
            count = (count / 3) * 2 # cut off last 1/3
        if tp == 2: # if is receiving
            count = count / 2 # cut off last 1/2
        tmp1.append(data_path[i][0]) # video path
        tmp1.append(data_path[i][1]) # action label
        tmp1.extend([int(count)]) # number of video frames after cut
        tmp1.extend([vt]) # which type of view
        idx_of_frames.append(tmp1) # combine into 1 data
    return idx_of_frames

def Fixed_action_filter(input_path, output_path):
    serving_data = []
    receiving_data = []
    idx_of_serve = []
    idx_of_receive = []
    print("Filting list, path: {}".format(input_path))
    ######## Main ########
    with open(input_path, encoding='utf-8') as file:
        rows = csv.reader(file)
        for row in rows:
            if int(row[1]) > 0 and int(row[1]) <= 10: # Class 1 ~ 10 are serving class
                serving_data.append(row)
            if int(row[1]) >= 11 and int(row[1]) <= 20: # Class 10 ~ 20 are receiving class
                receiving_data.append(row)

    idx_of_serve = fixed_length_video_cut(serving_data, tp = 1)
    idx_of_receive = fixed_length_video_cut(receiving_data, tp = 2)

    # Save the file #
    with open(output_path, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for j in range (len(idx_of_serve)):
            writer.writerow([idx_of_serve[j][0], idx_of_serve[j][1], idx_of_serve[j][2], idx_of_serve[j][3]])
        for j in range (len(idx_of_receive)):
            writer.writerow([idx_of_receive[j][0], idx_of_receive[j][1], idx_of_receive[j][2], idx_of_receive[j][3]])
    csvfile.close()
    ######## Main ########

def fuzzy_video_cut(data_path, tp): #tp1 = serving, tp2 = receving
    idx_of_frames = []
    for i in trange (len(data_path)):
        tmp1 = []
        v_opt_flow = []
        count = 0
        vt = viwe_type(data_path[i][0])
        cap = cv.VideoCapture(cv.samples.findFile(data_path[i][0]))
        ret, frame1 = cap.read()
        frame1 = cv.resize(frame1, (230, 230))
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        amount_of_frames = 0
        hsv[..., 1] = 255
        while(1):
            amount_of_frames = amount_of_frames + 1
            ret, frame2 = cap.read()
            if not ret:
                break
            frame2 = cv.resize(frame2, (230, 230))
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            tmp = sum(sum(hsv[..., 2]))
            v_opt_flow.append(int(tmp))
        
        #### membership function value
        cut_point = 0
        c = 0 # counter of frames
        min_v_opt_flow = int(min(v_opt_flow))
        max_v_opt_flow = int(max(v_opt_flow))
        one_sixth_of_range_opt_flow = (max_v_opt_flow - min_v_opt_flow) / 6
        #### membership function value

        if tp == 1: # Fuzzy rules of serving
            for v in v_opt_flow:
                if c > 3:
                    oat = v_opt_flow[c-4:c-1] # oat = optical flow after present time
                    obt = v_opt_flow[c+1:c+3] # obt = optical flow before present time
                    if ((amount_of_frames/6) * 4) <= count and count >= ((amount_of_frames/6) * 5):
                        if v <= min_v_opt_flow + one_sixth_of_range_opt_flow:
                            if oat >= min_v_opt_flow + (one_sixth_of_range_opt_flow * 3) and oat <= min_v_opt_flow + (one_sixth_of_range_opt_flow * 6):
                                if obt >= min_v_opt_flow + (one_sixth_of_range_opt_flow * 3) and obt <= min_v_opt_flow + (one_sixth_of_range_opt_flow * 6):
                                    cut_point = int(c)
                                    break
                c = c + 1
                
        if tp == 2: # Fuzzy rules of receiving
            for v in v_opt_flow:
                if c > 3:
                    oat = v_opt_flow[c-4:c-1] # oat = optical flow after present time
                    obt = v_opt_flow[c+1:c+3] # obt = optical flow before present time
                    if ((amount_of_frames / 6) * 3) <= count and count >= ((amount_of_frames / 6) * 4):
                        if v <= min_v_opt_flow + one_sixth_of_range_opt_flow:
                            if oat >= min_v_opt_flow + (one_sixth_of_range_opt_flow * 3) and oat <= min_v_opt_flow + (one_sixth_of_range_opt_flow * 6):
                                if obt >= min_v_opt_flow + (one_sixth_of_range_opt_flow * 3) and obt <= min_v_opt_flow + (one_sixth_of_range_opt_flow * 6):
                                    cut_point = int(c)
                                    break
                c = c + 1

        if cut_point == 0: # If video is no match with any fuzzy rules, then cut fixed length
            if tp == 1:
                    cut_point = (c / 3) * 2
            if tp == 2:
                    cut_point = c / 2
        # print(int(cut_point))
        
        tmp1.append(data_path[i][0])
        tmp1.append(data_path[i][1])
        tmp1.extend([int(cut_point)])
        tmp1.extend([vt])
        idx_of_frames.append(tmp1)
    return idx_of_frames

def fuzzy_action_filter(input_path, output_path):
    serving_data = []
    receiving_data = []
    idx_of_serve = []
    idx_of_receive = []
    print("Filting list, path: {}".format(input_path))
    ######## Main ########
    with open(input_path, encoding='utf-8') as file:
        rows = csv.reader(file)
        for row in rows:
            if int(row[1]) > 0 and int(row[1]) <= 10: # Class 1 ~ 10 are serving class
                serving_data.append(row)
            if int(row[1]) >= 11 and int(row[1]) <= 20: # Class 10 ~ 20 are receiving class
                receiving_data.append(row)

    idx_of_serve = fuzzy_video_cut(serving_data, tp = 1)
    idx_of_receive = fuzzy_video_cut(receiving_data, tp = 2)

    # Save the file #
    with open(output_path, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for j in range (len(idx_of_serve)):
            writer.writerow([idx_of_serve[j][0], idx_of_serve[j][1], idx_of_serve[j][2], idx_of_serve[j][3]])
        for j in range (len(idx_of_receive)):
            writer.writerow([idx_of_receive[j][0], idx_of_receive[j][1], idx_of_receive[j][2], idx_of_receive[j][3]])
    csvfile.close()
    ######## Main ########


# DF("TTset/TTset_training_data_list.csv", "TTset/Filted_TTset_training_data_list.csv")
# split_by_view_type(split_data_by_choose=False)

# # Scan the opt-flow value of video (complete) #
# def opt_flow_scanning(data_path):
#     all_max = []
#     all_min = []
#     for i in trange (len(data_path)):
#         tmp1 = []
#         cap = cv.VideoCapture(cv.samples.findFile(data_path[i][0]))
#         ret, frame1 = cap.read()
#         frame1 = cv.resize(frame1, (230, 230))
#         prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
#         hsv = np.zeros_like(frame1)
#         hsv[..., 1] = 255
#         while(1):
#             ret, frame2 = cap.read()
#             if not ret:
#                 # print('No frames grabbed!')
#                 break
#             frame2 = cv.resize(frame2, (230, 230))
#             next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
#             flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#             mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#             # hsv[..., 0] = ang*180/np.pi/2
#             hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
#             tmp1.append(sum(sum(hsv[..., 2]))/100)
#         all_min.append(min(tmp1))
#         all_max.append(max(tmp1))
#     return all_min, all_max

# # Find the index of video (building...) #
# def opt_flow_idx_of_frames(data_path, opt_min, opt_max):
#     idx_of_frames = []
#     for i in trange (len(data_path)):    
#         frames = []
#         tmp1 = []
#         sum_hsv = 0
#         cap = cv.VideoCapture(cv.samples.findFile(data_path[i][0]))
#         ret, frame1 = cap.read()
#         frame1 = cv.resize(frame1, (230, 230))
#         prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
#         hsv = np.zeros_like(frame1)
#         count = 0
#         hsv[..., 1] = 255
#         while(1):
#             count = count + 1
#             ret, frame2 = cap.read()
#             if not ret:
#                 print('No frames grabbed!')
#                 break
#             frame2 = cv.resize(frame2, (230, 230))
#             next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
#             flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#             mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#             hsv[..., 0] = ang*180/np.pi/2
#             hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
#             sum_hsv = sum(sum(hsv[..., 2]))

#             # Rules #
#             if sum_hsv > opt_max:
#                 frames.extend(count)

#         tmp1.extend(data_path[i][0])
#         tmp1.extend(data_path[i][1])
#         tmp1.append(frames)
#         idx_of_frames.append(tmp1)

#     return idx_of_frames

# ser_min, ser_max = opt_flow_scanning(serving_data)
# print("Average of ser_min: ", end="")
# print(sum(ser_min)/len(ser_min))
# print("Average of ser_max: ", end="")
# print(sum(ser_max)/len(ser_max))

# rec_min, rec_max = opt_flow_scanning(receiving_data)
# print("Average of rec_min: ", end="")
# print(sum(rec_min)/len(rec_min))
# print("Average of rec_max: ", end="")
# print(sum(rec_max)/len(rec_max))

# idx_of_serve = opt_flow_idx_of_frames(serving_data, ser_min, ser_max)
# idx_of_receive = opt_flow_idx_of_frames(receiving_data, rec_min, rec_max)
