from statistics import mode
import torch
from torch.utils.data import DataLoader
from frame_loader import frame_loader
from torch import nn
from Model import arch10, arch22, arch19, I3D_transfer_2, X3D_transfer
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from TTset_parser import creat_data_list
from Data_filter import Fixed_action_filter, fuzzy_action_filter, split_by_view_type
import csv
# from src.i3dpt import Unit3Dpy
torch.cuda.empty_cache()

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
training_loss_save_path =     "I3D_transfer_2_epoch200_bs32_TFS_230_36fps_TTset_without_mine_2037_validation_set_chossed_random5%_baseline10_VLAM_imagecropv1_useNCC2data_tloss_20230414.txt"
training_acu_save_path =      "I3D_transfer_2_epoch200_bs32_TFS_230_36fps_TTset_without_mine_2037_validation_set_chossed_random5%_baseline10_VLAM_imagecropv1_useNCC2data_tacu_20230414.txt"
validation_acu_save_path =    "I3D_transfer_2_epoch200_bs32_TFS_230_36fps_TTset_without_mine_2037_validation_set_chossed_random5%_baseline10_VLAM_imagecropv1_useNCC2data_vacu_20230414.txt"
validation_loss_save_path =   "I3D_transfer_2_epoch200_bs32_TFS_230_36fps_TTset_without_mine_2037_validation_set_chossed_random5%_baseline10_VLAM_imagecropv1_useNCC2data_vloss_20230414.txt"
class_divided_acu_save_path = "I3D_transfer_2_epoch200_bs32_TFS_230_36fps_TTset_without_mine_2037_validation_set_chossed_random5%_baseline10_VLAM_imagecropv1_useNCC2data_class_divided_acu_20230414.txt"
best_model_weight_save_path = "I3D_transfer_2_epoch200_bs32_TFS_230_36fps_TTset_without_mine_2037_validation_set_chossed_random5%_baseline10_VLAM_imagecropv1_useNCC2data_best_model_20230414.pt"
model_load_path_for_training = ""
validation_only_model_path = "I3D_transfer_2_epoch200_bs32_TFS_230_36fps_TTset_without_mine_2037_validation_set_chossed_random5%_baseline10_VLAM_imagecropv1_useNCC2data_best_model_20230414.pt"
##########################################################
I3D_weight_RGB_pth = "I3D_weights/rgb_imagenet.pt"
best_acu = 0
training_acu_count = 0
training_loss_record = []
training_acu_record = []
vali_loss_record = []
vali_acu_record = []
class_divided_validation_acu_record = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Now using: {}".format(device))

def Training(dataloader, model, loss_function, optimizer): # Phase of training
    global training_acu_count
    optimizer.zero_grad()
    for i, (x, y) in tqdm(enumerate(dataloader)):
        model.train()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        t_loss = loss_function(pred, y)
        t_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        for i in range (int(len(y))):
            pred_convert = torch.argmax(pred[i], 0)
            pred_convert = pred_convert.item()
            if pred_convert == y[i]:
                training_acu_count = training_acu_count + 1

    print("Training Loss: {}".format(t_loss))
    loss_value = float(t_loss.item())
    training_loss_record.append(loss_value)
    ####### Save training loss #######
    file = open(training_loss_save_path, 'w')
    tem = str(training_loss_record)
    file.write(tem)
    file.close()

def Validation(dataloader, model, loss_function, present_epoch): # Phase of validation
    count_acu = 0
    count_all = 0
    global best_acu
    for i, (x, y) in tqdm(enumerate(dataloader)):
        model.eval()
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            v_loss = loss_function(pred, y)
            pred = torch.argmax(pred, 1)
            pred = pred.item()
            count_all = count_all + 1
            if pred == y:
                count_acu = count_acu + 1
    acu = (count_acu/count_all) * 100
    if acu >= best_acu:
        best_acu = acu
        print("Saving best moedl...")
        torch.save(model.state_dict(), best_model_weight_save_path)
    if present_epoch % 25 == 0: # Save model weight at progress of 25%, 50%, 75%
        torch.save(model.state_dict(), "I3D_transfer_2_epoch200_bs32_TFS_230_36fps_TTset_without_mine_2037_validation_set_chossed_random5%_baseline10_VLAM_imagecropv1_useNCC2data_model{}_20230414.pt".format(present_epoch))
    loss_value = float(v_loss.item())
    vali_acu_record.append(count_acu/count_all)
    vali_loss_record.append(loss_value)
    ####### Save validation loss #######
    file = open(validation_loss_save_path, 'w')
    tem = str(vali_loss_record)
    file.write(tem)
    file.close()
    ####### Save validation acu #######
    file = open(validation_acu_save_path, 'w')
    tem = str(vali_acu_record)
    file.write(tem)
    file.close()
    ####### Print validation loss and acu #######
    print("Validation loss : {}%".format(v_loss))
    print("Validation accuracy: {}%".format(acu))

def Validation_only(dataloader, model, loss_function, present_epoch): # Only do validation on model
    count_acu = 0
    count_all = 0
    global best_acu
    for i, (x, y) in tqdm(enumerate(dataloader)):
        model.eval()
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            v_loss = loss_function(pred, y)
            pred = torch.argmax(pred, 1)
            pred = pred.item()
            count_all = count_all + 1
            if pred == y:
                count_acu = count_acu + 1
    acu = (count_acu/count_all) * 100
    ####### Print validation loss and acu #######
    print("Validation loss : {}%".format(v_loss))
    print("Validation accuracy: {}%".format(acu))

def result_show(only_class_divided_validation): # Plot result as graph
    if only_class_divided_validation == False:
        # loss
        plt.plot(training_loss_record, 'r')
        plt.plot(vali_loss_record, 'g')
        plt.legend(['Loss'])
        # plt.title("Training loss of TTA20 dataset")
        plt.xlabel("Epoch")
        plt.show()

        # acu
        plt.ylim(0, 1)
        plt.plot(training_acu_record, 'r')
        plt.plot(vali_acu_record, 'g')
        plt.legend(['Accuracy'])
        # plt.title("Validation accuracy of TTA20 dataset")
        plt.xlabel("Epoch")
        plt.show()

        # acu of divided class
        plt.hist(class_divided_validation_acu_record, color='g')
        plt.legend(['Class divided accuracy'])
        plt.xlabel("Class")
        plt.show()
        
    # acu of divided class
    if only_class_divided_validation == True:
        plt.hist(class_divided_validation_acu_record, color='g')
        plt.legend(['Class divided accuracy'])
        plt.xlabel("Class")
        plt.show()

def load_TTset(tset_path, vset_path): # Load data
    # load training data
    tmp1 = []
    all_data = []
    label_dir = []
    with open(tset_path, newline='', encoding="utf-8") as file:
        rows = csv.reader(file)
        for row in rows:
            tmp1.append(row[0])
            tmp1.append(row[1])
            label_dir.append(row[1])
            tmp1.extend([row[2]])
            tmp1.extend([row[3]])
            all_data.append(tmp1)
            tmp1 = []
        file.close()
    TTset_training_dataset = frame_loader(all_data, action_filter_require)
    t_data_length = int(len(label_dir))

    # load validation data
    tmp1 = []
    all_data = []
    with open(vset_path, newline='', encoding="utf-8") as file:
        rows = csv.reader(file)
        for row in rows:
            tmp1.append(row[0])
            tmp1.append(row[1])
            tmp1.extend([row[2]])
            tmp1.extend([row[3]])
            all_data.append(tmp1)
            tmp1 = []
        file.close()
    TTset_vali_dataset = frame_loader(all_data, action_filter_require)

    TTset_training_dataloader = DataLoader(TTset_training_dataset, batch_size = bs, shuffle=True, drop_last=False)
    TTset_vali_dataloader = DataLoader(TTset_vali_dataset, batch_size=1, shuffle=True, drop_last=False)
    return TTset_training_dataloader, TTset_vali_dataloader, t_data_length

def class_divided_validation(dataloader, model): # Show each class's validation result
    count_acu = 0
    count_class = 0
    tem = []
    global class_divided_validation_acu_record
    print("-----------------------------------------------")
    print("Class divided validation: ")
    for cla in range (0, 20):
        for i, (x, y) in tqdm(enumerate(dataloader)):
            if int(y) == cla:
                model.eval()
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                pred = torch.argmax(pred, 1)
                pred = pred.item()
                count_class = count_class + 1
                if pred == y:
                    count_acu = count_acu + 1

        if count_class == 0:
            print("Class {} has no data to divided validation.".format((cla+1)))
            count_acu = 0
            count_class = 0
            class_divided_validation_acu_record.append(0)

        if count_class != 0:
            acu = (count_acu/count_class) * 100
            print("Class {} validation accuracy: {}%".format((cla+1), acu))
            count_acu = 0
            count_class = 0
            class_divided_validation_acu_record.append(acu)
    
    file = open(class_divided_acu_save_path, 'w')
    tem = str(class_divided_validation_acu_record)
    file.write(tem)
    file.close()
    print("-----------------------------------------------")

def training_acu(t_data_length):
    global training_acu_count
    global training_acu_record
    t_acu = (training_acu_count/t_data_length)
    training_acu_record.append(t_acu)
    print("Training accuracy: {}".format(t_acu*100))
    file = open(training_acu_save_path, 'w')
    tem = str(training_acu_record)
    file.write(tem)
    file.close()
    training_acu_count = 0

# main #
split_data_by_choose = False            # Only training and validation particular class
only_class_divided_validation = False   # Only show each class's validation result
use_old_data = True                     # Use same dataset as last time
old_data_action_filter_require = False  # Filte the old dataset
action_filter_require = False
validation_only = True                  # Only do validation
action_filter_type = int(2)
bs = 16
epoch = 2
model = I3D_transfer_2(15, 20).to(device)
# model.load_state_dict(torch.load(model_load_path_for_training))
loss_fn = nn.CrossEntropyLoss().to(device=device)
opt = torch.optim.Adam(model.parameters())

if split_data_by_choose == False and only_class_divided_validation == False:
    if use_old_data == True:
        if old_data_action_filter_require == True:
            if action_filter_type == int(1):
                Fixed_action_filter(TTset_tset_path, filted_tset_path)
                Fixed_action_filter(TTset_vset_path, filted_vset_path)
            if action_filter_type == int(2):
                fuzzy_action_filter(TTset_tset_path, filted_tset_path)
                fuzzy_action_filter(TTset_vset_path, filted_vset_path)
            split_by_view_type(split_data_by_choose)
        TTset_training, TTset_vali, training_data_length = load_TTset(filted_tset_path, filted_vset_path)
    else:
        creat_data_list(split_data_by_choose, action_filter_require, action_filter_type)
        TTset_training, TTset_vali, training_data_length = load_TTset(filted_tset_path, filted_vset_path)

if split_data_by_choose == True and only_class_divided_validation == False:
    if use_old_data == True:
        if old_data_action_filter_require == True:
            if action_filter_type == int(1):
                Fixed_action_filter(TTset_tset_path, filted_tset_path)
                Fixed_action_filter(TTset_vset_path, filted_vset_path)
            if action_filter_type == int(2):
                fuzzy_action_filter(TTset_tset_path, filted_tset_path)
                fuzzy_action_filter(TTset_vset_path, filted_vset_path)
            split_by_view_type(split_data_by_choose)
        TTset_training, TTset_vali, training_data_length = load_TTset(TTset_tset_by_choose_path, TTset_vset_by_choose_path)
    else:
        creat_data_list(split_data_by_choose, action_filter_require, action_filter_type)
        TTset_training, TTset_vali, training_data_length = load_TTset(TTset_tset_by_choose_path, TTset_vset_by_choose_path)

if only_class_divided_validation == False and validation_only == False:
    for training_i in range (epoch):
        print("Epoch: {}".format(training_i + 1))
        print("-----------------------------------------------")
        Training(TTset_training, model, loss_fn, opt) # Start training phase
        training_acu(training_data_length)            # Save training accuracy
        Validation(TTset_vali, model, loss_fn, (training_i + 1)) # Start validation phase
        print("-----------------------------------------------\n")
    print("Showing result...")
    model.load_state_dict(torch.load(best_model_weight_save_path))
    class_divided_validation(TTset_vali, model)
    result_show(only_class_divided_validation)
    print("Training done!")

if only_class_divided_validation == False and validation_only == True:
        validation_i = 0
        print("Validation starting...")
        model.load_state_dict(torch.load(validation_only_model_path))
        Validation_only(TTset_vali, model, loss_fn, (validation_i + 1)) # Start validation phase
        print("Validation done!")
        print("-----------------------------------------------")

if only_class_divided_validation == True:
    model.load_state_dict(torch.load(best_model_weight_save_path))
    TTset_training, TTset_vali, training_data_length = load_TTset(TTset_tset_by_choose_path, TTset_vset_by_choose_path)
    class_divided_validation(TTset_vali, model)
    result_show(only_class_divided_validation)
    print("Done!")