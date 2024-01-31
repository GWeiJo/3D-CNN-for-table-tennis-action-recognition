# TTset_parser is for split the path of action video and the label, and made it as csv file #
import os
import csv
import glob as glob
import random
import copy
from Data_filter import Fixed_action_filter, fuzzy_action_filter
from Data_filter import split_by_view_type

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

def creat_data_list(split_data_by_choose, action_filter, action_filter_type): # Creat the csv file which contain video path and label
                                                                              # Which means x and y
    tset_dir = []
    tset_label = []
    vset_dir = []
    vset_label = []
    data_path = []
    label_num = []
    combine = []
    data_root = "TTset\data"
    tem = []
    tem.extend(glob.glob(os.path.join(data_root, "*")))
    
    if split_data_by_choose == False: # data with all calss
        for i in range (len(tem)):
            class_num = tem[i][33:35] # Get action label from file title, will change accroding to different file path
            class_num = int(class_num)
            if class_num <= 20 and class_num > 0: # Only take class 1 to class 20
                data_path.append(tem[i])
                ano = class_num-1
                label_num.append(ano)

        if len(data_path) != len(label_num): # If number of data path doesn't match number of label, raise error
            print("Data number does not match with label number, pls check again...")
            raise()
        with open('TTset\TTset_all_data_list.csv', 'w', newline='', encoding="utf-8") as csvfile: 
            writer = csv.writer(csvfile)
            for j in range (len(label_num)):
                writer.writerow([data_path[j], label_num[j]]) # save video path and label as csv file
            csvfile.close()
        for j in range (len(label_num)):
            combine.append([data_path[j], label_num[j]])
        training_set, validation_set = split_data_all(data_path, combine)
        for i in range(len(training_set)):
            tset_dir.append(training_set[i][0])
            tset_label.append(training_set[i][1])
        for i in range (len(validation_set)):
            vset_dir.append(validation_set[i][0])
            vset_label.append(validation_set[i][1])

        with open('TTset\TTset_training_data_list.csv', 'w', newline='', encoding="utf-8") as csvfile: # Save video path and label as csv file
            writer = csv.writer(csvfile)
            for j in range (len(tset_label)):
                writer.writerow([tset_dir[j], tset_label[j]])
            csvfile.close()
        read_tset_classes(split_data_by_choose)
        with open('TTset\TTset_validation_data_list.csv', 'w', newline='', encoding="utf-8") as csvfile: # Save video path and label as csv file
            writer = csv.writer(csvfile)
            for j in range (len(vset_label)):
                writer.writerow([vset_dir[j], vset_label[j]])
            csvfile.close()
        read_vset_classes(split_data_by_choose)
        if action_filter == True:
            if action_filter_type == int(1):
                Fixed_action_filter(TTset_tset_path, filted_tset_path)
                Fixed_action_filter(TTset_vset_path, filted_vset_path)
            if action_filter_type == int(2):
                fuzzy_action_filter(TTset_tset_path, filted_tset_path)
                fuzzy_action_filter(TTset_vset_path, filted_vset_path)
        split_by_view_type(split_data_by_choose)
    
    if split_data_by_choose == True: # data with choosed calss
        for i in range (len(tem)):
            class_num = tem[i][33:35] # Get action label from file title, will change accroding to different file path
            class_num = int(class_num)
            if class_num <= 20 and class_num > 0:
                if class_num == 5:
                    data_path.append(tem[i])
                    ano = class_num-1
                    label_num.append(ano)

                if class_num == 18:
                    data_path.append(tem[i])
                    ano = class_num-1
                    label_num.append(ano)

                if class_num == 19:
                    data_path.append(tem[i])
                    ano = class_num-1
                    label_num.append(ano)

        if len(data_path) != len(label_num):
            print("Data number does not match with label number, pls check again...")
            raise()
        with open('TTset\TTset_all_data_by_choose_list.csv', 'w', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for j in range (len(label_num)):
                writer.writerow([data_path[j], label_num[j]])
            csvfile.close()
        for j in range (len(label_num)):
            combine.append([data_path[j], label_num[j]])

        training_set, validation_set = split_data_choose(data_path, combine)
        for i in range(len(training_set)):
            tset_dir.append(training_set[i][0])
            tset_label.append(training_set[i][1])
        for i in range (len(validation_set)):
            vset_dir.append(validation_set[i][0])
            vset_label.append(validation_set[i][1])

        with open('TTset\TTset_training_data_by_choose_list.csv', 'w', newline='', encoding="utf-8") as csvfile: # Save video path and label as csv file
            writer = csv.writer(csvfile)
            for j in range (len(tset_label)):
                writer.writerow([tset_dir[j], tset_label[j]])
            csvfile.close()
        read_tset_classes(split_data_by_choose)
        with open('TTset\TTset_validation_data_by_choose_list.csv', 'w', newline='', encoding="utf-8") as csvfile: # Save video path and label as csv file
            writer = csv.writer(csvfile)
            for j in range (len(vset_label)):
                writer.writerow([vset_dir[j], vset_label[j]])
            csvfile.close()
        read_vset_classes(split_data_by_choose)
        Fixed_action_filter(TTset_tset_by_choose_path, filted_choosed_tset_path)
        Fixed_action_filter(TTset_vset_by_choose_path, filted_choosed_vset_path)
        split_by_view_type(split_data_by_choose)

def split_data_all(a_data, a_data_combine): # Split data into training data and validation data
    print("-----------------------------------------------")
    all_count = 0
    v_data = []
    t_data = []
    t_data = copy.deepcopy(a_data_combine)
    for i in range (1, 21):
        path_buffer = []
        random_choose_number = []
        for c in range(len(a_data)):
            if int(a_data[c][33:35]) == i:
                path_buffer.append(a_data_combine[c])
                all_count = all_count + 1
        if int(len(path_buffer)) > 10: # The threshold of amount of data
            j = int(len(path_buffer))/20 # Take 5% of data as validation
            if j < 1: 
                j = 1
            j = int(j)
            random_choose_number = random.sample(range(1, int(len(path_buffer))), j)# Take different validation data every time
            for k in range (j):
                    v_data.append(path_buffer[random_choose_number[k]])
        print("Class {} has {} pieces of data.".format(i, int(len(path_buffer))))
    for i in range(len(v_data)):
        t_data.remove(v_data[i])
    print("Total has {} pieces of data.".format(all_count))
    print("Training data has {} pieces of data.".format(len(t_data)))
    print("Validation data has {} pieces of data.".format(len(v_data)))
    print("-----------------------------------------------")
    return t_data, v_data

def split_data_choose(a_data, a_data_combine): # Choose class on purpose
    print("-----------------------------------------------")
    all_count = 0
    v_data = []
    t_data = []
    t_data = copy.deepcopy(a_data_combine)
    for i in range (1, 21):
        path_buffer = []
        random_choose_number = []
        for c in range(len(a_data)):
            if int(a_data[c][33:35]) == i:
                path_buffer.append(a_data_combine[c])
                all_count = all_count + 1
        if int(len(path_buffer)) > 180: # The threshold of amount of data
            j = int(len(path_buffer))/20 # Take 5% of data as validation
            if j < 1: 
                j = 1
            j = int(j)
            random_choose_number = random.sample(range(1, int(len(path_buffer))), j) # Take different validation data every time
            for k in range (j):
                    v_data.append(path_buffer[random_choose_number[k]])
        print("Class {} has {} pieces of data.".format(i, int(len(path_buffer))))
    for i in range(len(v_data)):
        t_data.remove(v_data[i])
    print("Total has {} pieces of data.".format(all_count))
    # print("Training data has {} pieces of data.".format(len(t_data)))
    # print("Validation data has {} pieces of data.".format(len(v_data)))
    print("-----------------------------------------------")
    return t_data, v_data


def read_vset_classes(split_data_by_choose): # Show how many validation data in each class
    amount_of_data = 0
    total_count = 0
    if split_data_by_choose == False:
        print("-----------------------------------------------")
        for i in range (0, 20):
            with open("TTset\TTset_validation_data_list.csv", encoding="utf-8") as file:
                rows = csv.reader(file)
                for row in rows:
                    if (int(row[1]) == i):
                        amount_of_data = amount_of_data + 1
                        total_count = total_count + 1
                file.close()
            print("The validation set of class {} has {} data.".format((i+1), amount_of_data))
            amount_of_data = 0
        print("Validation set total has {} pieces data.".format(total_count))
        print("-----------------------------------------------")

    if split_data_by_choose == True:
        print("-----------------------------------------------")
        for i in range (0, 20):
            with open("TTset\TTset_validation_data_by_choose_list.csv", encoding="utf-8") as file:
                rows = csv.reader(file)
                for row in rows:
                    if (int(row[1]) == i):
                        amount_of_data = amount_of_data + 1
                        total_count = total_count + 1
                file.close()
            print("The validation set of class {} has {} data.".format((i+1), amount_of_data))
            amount_of_data = 0
        print("Validation set total has {} pieces data.".format(total_count))
        print("-----------------------------------------------")

def read_tset_classes(split_data_by_choose): # Show how many training data in each class
    amount_of_data = 0
    total_count = 0
    if split_data_by_choose == False:
        print("-----------------------------------------------")
        for i in range (0, 20):
            with open("TTset\TTset_training_data_list.csv", encoding="utf-8") as file:
                rows = csv.reader(file)
                for row in rows:
                    # print(row[1])
                    if (int(row[1]) == i):
                        amount_of_data = amount_of_data + 1
                        total_count = total_count + 1
                file.close()
            print("The training set of class {} has {} data.".format((i+1), amount_of_data))
            amount_of_data = 0
        print("Training set total has {} pieces data.".format(total_count))
        print("-----------------------------------------------")

    if split_data_by_choose == True:
        print("-----------------------------------------------")
        for i in range (0, 20):
            with open("TTset\TTset_training_data_by_choose_list.csv", encoding="utf-8") as file:
                rows = csv.reader(file)
                for row in rows:
                    # print(row[1])
                    if (int(row[1]) == i):
                        amount_of_data = amount_of_data + 1
                        total_count = total_count + 1
                file.close()
            print("The training set of class {} has {} data.".format((i+1), amount_of_data))
            amount_of_data = 0
        print("Training set total has {} pieces data.".format(total_count))
        print("-----------------------------------------------")

# creat_data_list(split_data_by_choose=True)