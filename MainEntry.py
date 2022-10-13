from tensorflow import keras
import tensorflow as tf
import efficientnet.keras as efn
import cv2
import keras.backend as K
import numpy as np
import pickle as pk
#from sklearn.externals import joblib
from TryInt import TryInt
from UpperStrip import UpperStrip
import os
from tkinter import *
from tkinter.tix import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as msg
from time import sleep
from time import time
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import multilabel_confusion_matrix as confusion_matrix
from sklearn import metrics
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from keras.models import model_from_json
from scipy.io import savemat
from keras.models import load_model
import h5py


win = Tk()
tip_win = Balloon(win)
tip_win.message.config(font=("haveltica 12 bold"))
print('THANK YOU FOR USING NRF AI TOOLKIT')
win.title('NRF AI TOOLKIT')
#   tip_win = Balloon(win)
win.resizable(False, False)  # prevent it from been resized by user
#feature_extractor = c3d.c3d_feature_extractor()
#print('I came here')
save_model_dir = 'C:\Model'
save_weight_dir = 'C:\Weight'
max_num_frames_per_video = 2    # Use None to use all the frames in each video, or desire max number of frames can be used. 2 is just for testing the code
image_hight_default = 240
image_width_default = 300
scaling_factor = 255/2
scaling_offset = -1
normal_learning_rate_default = 1e-3
fine_tuning_learning_rate_default =1e-6
number_of_batch_iteration_defualt = 300
main_training_epoch_default = 2000
fine_tuning_epoch_default = 20
#number_classes = 5
batch_video_size_default = 50      #   10 videos from each class
number_video_per_class_default = 10
first_layer_neuron_default = 512
second_layer_neuron_default = 128
lambda_1 = 0.00008
lambda_2 = 0.00008
test_weighted_mean_factor = 0.01
current_dir = os.getcwd()
frame_count_dir = os.getcwd()     #   'C:/FramesCount'  # this need to be created for pickle
frames_count_list_file = 'framesCount.pickl'
sub_directory_name = ['Training', 'Testing']
#class_labels = ['Robery', 'Bandit', 'BanditActivity', 'KeepNap', 'Normal'] Robery_Bandit_BanditActivity_KeepNap_Normal
video_folder_train = ['TrainRobery', 'TrainBandit', 'TrainBanditActivity', 'TrainKeepNap', 'TrainNormal']
video_folder_test = ['TestRobery', 'TestBandit', 'TestBanditActivity', 'TestKeepNap', 'TestNormal']
video_folder_validation = ['ValidationRobery', 'ValidationBandit', 'ValidationBanditActivity', 'ValidationKeepNap', 'ValidationNormal']
model_folder = 'Model'
model_weight_folder = 'ModelWeight'
number_of_batch_iteration_variable = tk.StringVar()
activation_function_variable = tk.StringVar()
class_label_variable = tk.StringVar()
model_variable = tk.StringVar()
normal_rate = tk.StringVar()
tuning_rate = tk.StringVar()
directry_path = tk.StringVar()
image_size_width = tk.StringVar()
image_size_higth = tk.StringVar()
normal_epoch = tk.StringVar()
tuning_epoch = tk.StringVar()
number_video_per_class_variable = tk.StringVar()
batch_video_size_variable = tk.StringVar()
dropdown_list = tk.StringVar()
first_layer_neuron_variable = tk.StringVar()
second_layer_neuron_variable = tk.StringVar()
#   number_video_per_batch_variable = tk.StringVar()
directry_edit = ttk.Entry(win, width=30, textvariable=directry_path)
image_size_width_edit =  ttk.Entry(win, textvariable=image_size_width)
image_size_higth_edit =  ttk.Entry(win, textvariable=image_size_higth)
batch_video_size_variable_edit =  ttk.Entry(win, textvariable=batch_video_size_variable)
#   number_video_per_class_variable_edit =  ttk.Entry(win, textvariable=number_video_per_class_variable)
normal_epoch_edit =  ttk.Entry(win, textvariable=normal_epoch)
tuning_epoch_edit =  ttk.Entry(win, textvariable=tuning_epoch)
normal_rate_edit =  ttk.Entry(win, textvariable=normal_rate)
tuning_rate_edit =  ttk.Entry(win, textvariable=tuning_rate)
first_layer_neuron_variable_edit =  ttk.Entry(win, textvariable=first_layer_neuron_variable)
second_layer_neuron_variable_edit =  ttk.Entry(win, textvariable=second_layer_neuron_variable)
number_of_batch_iteration_edit = ttk.Entry(win, textvariable=number_of_batch_iteration_variable)
class_labels_edit = ttk.Entry(win, textvariable=class_label_variable)
#   optionslabel = ttk.Label(win, text='')
DropDownActivatonFunction = ttk.Combobox(win,width=15, textvariable=activation_function_variable, state='readonly')
DropDownActivatonFunction['values'] = ('sigmoid_relu', 'sigmoid_sigmoid', 'relu_sigmoid', 'relu_relu')
DropDownModel =  ttk.Combobox(win,width=15, textvariable=model_variable, state='readonly')
DropDownModel['values'] = ('Xception', 'RestNet50', 'InCeptionV3', 'VGG16', 'EfficientNetB0', 'EfficientNetB7')
DropDownListName = ttk.Combobox(win,width=20, textvariable=dropdown_list, state='readonly')
DropDownListName['values'] = ('Create Model/Training', 'Test', 'Lunch Model', 'Visualize')
directry_label = ttk.Label(win, text='Video Directory')
image_size_label = ttk.Label(win, text='Image Size')
batch_video_size_label = ttk.Label(win, text='Batch Size')
video_per_class_label = ttk.Label(win, text='Video Per Class') # not needed
epoch_label = ttk.Label(win, text='Epoch')
learning_rate_label = ttk.Label(win, text='Learning Rate')
neuron_label = ttk.Label(win, text='Neurons')
number_of_batch_iteration_label = ttk.Label(win, text='Bacth Iteration')
class_labels_label = ttk.Label(win, text='Class Labels')
directry_label.grid(column=0, row=0,sticky='w', padx=2,pady=4)
directry_edit.grid(column=1, row=0,sticky='w', padx=2,pady=4)
DropDownModel.grid(column=2, row=0,sticky='w', padx=2,pady=4)
image_size_label.grid(column=0, row=1,sticky='w', padx=2,pady=4)
image_size_width_edit.grid(column=1, row=1,sticky='w', padx=2,pady=4)
image_size_higth_edit.grid(column=2, row=1,sticky='w', padx=2,pady=4)
batch_video_size_label.grid(column=0, row=2,sticky='w', padx=2,pady=4)
batch_video_size_variable_edit.grid(column=1, row=2,sticky='w', padx=2,pady=4)
DropDownListName.grid(column=2, row=2,sticky='w', padx=2,pady=4)
epoch_label.grid(column=0, row=3,sticky='w', padx=2,pady=4)
normal_epoch_edit.grid(column=1, row=3,sticky='w', padx=2,pady=4)
tuning_epoch_edit.grid(column=2, row=3,sticky='w', padx=2,pady=4)
learning_rate_label.grid(column=0, row=4,sticky='w', padx=2,pady=4)
normal_rate_edit.grid(column=1, row=4,sticky='w', padx=2,pady=4)
tuning_rate_edit.grid(column=2, row=4,sticky='w', padx=2,pady=4)
neuron_label.grid(column=0, row=5,sticky='w', padx=2,pady=4)
first_layer_neuron_variable_edit.grid(column=1, row=5,sticky='w', padx=2,pady=4)
second_layer_neuron_variable_edit.grid(column=2, row=5,sticky='w', padx=2,pady=4)
number_of_batch_iteration_label.grid(column=0, row=6,sticky='w', padx=2,pady=4)
number_of_batch_iteration_edit.grid(column=1, row=6,sticky='w', padx=2,pady=4)
DropDownActivatonFunction.grid(column=2, row=6,sticky='w', padx=2,pady=4)
class_labels_label.grid(column=0, row=7, sticky='w', padx=2,pady=4)
class_labels_edit.grid(column=1, row=7, sticky='w', padx=2,pady=4)
tip_win.bind_widget(class_labels_edit, balloonmsg= 'Class Labels Separated by comma')
tip_win.bind_widget(DropDownActivatonFunction, balloonmsg='Select Hiding Layers\nActivation Function')
tip_win.bind_widget(DropDownListName, balloonmsg='Select a Function')
tip_win.bind_widget(number_of_batch_iteration_edit, balloonmsg='Number of Batch Iteration')
tip_win.bind_widget(DropDownModel, balloonmsg='Select Feature Extraction Model')
tip_win.bind_widget(first_layer_neuron_variable_edit, balloonmsg='Number of Neurons\nIn First Layer Classifier')
tip_win.bind_widget(second_layer_neuron_variable_edit, balloonmsg='Number of Neurons\nIn Second Layer Classifier')
tip_win.bind_widget(normal_rate_edit, balloonmsg='Normal Learning Rate')
tip_win.bind_widget(tuning_rate_edit, balloonmsg='Fine Tuning Learning Rate')
tip_win.bind_widget(normal_epoch_edit, balloonmsg='Normal Epoch')
tip_win.bind_widget(tuning_epoch_edit, balloonmsg='Fine Tuning Epoch')
tip_win.bind_widget(batch_video_size_variable_edit, balloonmsg='Number of Videos Per Batch')
tip_win.bind_widget(image_size_width_edit, balloonmsg='Image Width Size')
tip_win.bind_widget(image_size_higth_edit, balloonmsg='Image Hieght Size')
tip_win.bind_widget(directry_edit, balloonmsg='Videos Directory')
#tip_win.bind_widget(directry_edit, balloonmsg='Videos Directory')
directry_edit.focus()


def action_function():
    #print('You Click Me')
    action = DropDownListName.get()
    if action == 'Create Model/Training':
        create_model()
    elif action == 'Test':
        testing()
    elif action == 'Lunch Model':
        pass
    elif action == 'Visualize':
        pass
    else:
        msg.showerror('Error', 'Please select a function')
        return


def number_of_classes_labels():
    class_labels = class_label_variable.get()
    if class_labels:
        class_labels = UpperStrip(class_labels)
        if class_labels:
            class_labels = class_labels.split(',')
        else:
            msg.showerror('Error', 'No Class Labels')
            return
    else:
        msg.showerror('Error', 'No Class Labels')
        return
    number_classes = len(class_labels)
    print('The number of classes are:\t', number_classes, ',\tClass Labels:\t', class_labels)
    return [class_labels, number_classes]


def create_model():
    classes_labels_number = number_of_classes_labels()
    if classes_labels_number is None:
        return
    class_labels = classes_labels_number[0]
    number_classes = classes_labels_number[1]
    first_layer_neuron = TryInt(first_layer_neuron_variable.get(), 'First Layer Neurons ')
    if first_layer_neuron is None:
        first_layer_neuron = first_layer_neuron_default
    second_layer_neuron = TryInt(second_layer_neuron_variable.get(), 'Second layer neurons ')
    if second_layer_neuron is None:
        second_layer_neuron =  second_layer_neuron_default
    image_width = TryInt(image_size_width.get(), 'Image width ')
    if image_width is None:
        image_width = image_width_default
    image_hight = TryInt(image_size_higth.get(), 'Image hiegth ')
    if image_hight is None:
        image_hight = image_hight_default
    batch_video_size = TryInt(batch_video_size_variable.get(), 'Batch size ')
    if batch_video_size is None:
        batch_video_size = batch_video_size_default
    number_video_per_class = int(batch_video_size/number_classes)
    fine_tuning_epoch = TryInt(tuning_epoch.get(), 'Tuning epoch ')
    if fine_tuning_epoch is None:
        fine_tuning_epoch = fine_tuning_epoch_default
    main_training_epoch = TryInt(normal_epoch.get(), 'Training epoch ')
    if main_training_epoch is None:
        main_training_epoch = main_training_epoch_default
    normal_learning_rate = normal_rate.get()
    try:
        normal_learning_rate = float(normal_learning_rate)
    except Exception as e:
        normal_learning_rate = 1e-3
    try:
        fine_tuning_learning_rate = float(tuning_rate.get())
    except Exception as e:
        fine_tuning_learning_rate = 1e-6
    activation_function = activation_function_variable.get()
    if activation_function:
        activation_function = activation_function.split('_')
    else:
        msg.showerror('Error', 'Please select the hiding layers activation function')
        return
    ###################
    if number_classes < 2:
        out_activation_function = 'sigmoid'
    else:
        out_activation_function = 'softmax'
    print('Activation Functions:\tFirst:\t', activation_function[0], ',\tSecond:\t', activation_function[1], ',\tOutput:\t', out_activation_function)
    number_of_batch_iteration = TryInt(number_of_batch_iteration_variable.get(), 'Number of Batch Iteration ')
    if number_of_batch_iteration == None:
        number_of_batch_iteration = number_of_batch_iteration_defualt

    base_pretrained_model = model_variable.get()
    if base_pretrained_model == 'VGG16':
        selected_model = 'VGG16'
        base_xception =keras.applications.vgg16.VGG16(include_top=False,
                                                  input_shape = (image_width, image_hight, 3),
                                                  weights="imagenet")
    elif base_pretrained_model == 'InCeptionV3':
        selected_model = 'InCeptionV3'
        base_xception = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                input_shape = (image_width, image_hight, 3),
                                                                weights="imagenet")
    elif base_pretrained_model == 'Xception':
        selected_model = 'Xception'
        base_xception = keras.applications.Xception(include_top=False,
                                           input_shape = (image_width, image_hight, 3),
                                           weights="imagenet")
    elif base_pretrained_model == 'RestNet50':
        selected_model = 'RestNet50'
        base_xception = keras.applications.ResNet50(include_top=False,
                                                input_shape = (image_width, image_hight, 3),
                                                weights="imagenet")
    elif base_pretrained_model == 'EfficientNetB0':
        selected_model = 'EfficientNetB0'
        base_xception = efn.EfficientNetB0(include_top=False,
                                       input_shape = (image_width, image_hight, 3),
                                       weights="imagenet")
    elif base_pretrained_model == 'EfficientNetB7':
        selected_model = 'EfficientNetB7'
        base_xception = efn.EfficientNetB7(include_top=False,
                                           input_shape = (image_width, image_hight, 3),
                                           weights="imagenet")
    else:
        msg.showerror('Error', 'Please Select The Features Extraction Model To Use')
        return
    base_xception.trainable = False # freeze to keep pretrained weigths
    print('The model selected is:\t', selected_model)
    input_layer = keras.layers.Input(shape=(image_width, image_hight, 3))
    scale_layer = keras.layers.Rescaling(scale=scaling_factor, offset=scaling_offset)
    scale_input = scale_layer(input_layer)
    con_model = base_xception(scale_input, training=False)  # initial line excluding the next two lines code training=False to maintain the statistics ( i.e. mean and std) for global bach normalization
    # base_xception = base_xception(scale_input, training=False)  # not sure if this assignment is currect
    # base_xception.trainable = False # freeze to keep pretrained weigths
    # for layer in base_xception.layers:  # this code is not necessary as the previous code has freezed all trainable weights
    #     layer.trainable = False
    # con_model = base_xception.output  # I think is the final output stage of the base model that should be passed to the classifier
    con_model = keras.layers.GlobalAveragePooling2D()(con_model)
    con_model = keras.layers.Dropout(0.2)(con_model) # Regularize with dropout
    linear = keras.layers.Dense(first_layer_neuron, activation=activation_function[0])(con_model)
    linear = keras.layers.Dropout(0.3)(linear)
    linear = keras.layers.Dense(second_layer_neuron, activation=activation_function[1])(linear)
    linear = keras.layers.Dropout(0.3)(linear)
    predictions = keras.layers.Dense(number_classes, activation=out_activation_function)(linear) #   was the 5 was originally 101
    model = keras.models.Model(inputs=input_layer, outputs=predictions)

    # compile the model with based model freezed
    epoch = main_training_epoch
    learning_rate = normal_learning_rate
    optimizer=keras.optimizers.Adam(learning_rate)

    ## Trainining the model
    model.compile(optimizer=optimizer, loss=objective_function)
    print('Freezed model summary:\n', model.summary())
    loss_graph = []
    start_time = time()
    over_all_iteration = 0
    for normal_fine_tune_index in range(0, 2):
        if normal_fine_tune_index == 1:     #   unfreezed and fine tune the model
            base_xception.trainable = True # unfreeze based model weigths
            epoch = fine_tuning_epoch
            learning_rate = fine_tuning_learning_rate
            optimizer=keras.optimizers.Adam(learning_rate)
            ## real compiled the model to maintain the new changes
            model.compile(optimizer=optimizer, loss=objective_function)
            print('Unfreezed model summary:\n', model.summary())
        for iteration_index in range(0, epoch):
            traing_data, targets, number_frames_per_video_list = prepare_video_frames_for_training(sub_directory_name[0])  # zero for training 1 for testing
            frame_count_file = open(frames_count_list_file, 'wb')
            pk.dump(number_frames_per_video_list,frame_count_file) # save frame per video to be load in objective function
            frame_count_file.close()
            batch_loss = model.train_on_batch(traing_data, targets)   # not sure if epoch is needed on trin.batch
            loss_graph = np.hstack((loss_graph, batch_loss))
            over_all_iteration = over_all_iteration + 1
            if iteration_index % 1 == 0 and normal_fine_tune_index == 0:
                print('Normal Iteration:\t', over_all_iteration, ',\tTime Taken in Minutes:\t',round(((time() - start_time)/60), 2), ',\t Loss:\t', batch_loss )
            elif iteration_index % 1 == 0 and normal_fine_tune_index == 1:
                print('Fine Tuning Iteration:\t', over_all_iteration, ',\tTime Taken in Minutes:\t',round(((time() - start_time)/60), 2), ',\t Loss:\t', batch_loss )
    #model_path = os.path.join(save_model_dir, base_pretrained_model + '.pickl')
    model_path = os.path.join(save_model_dir, base_pretrained_model + '.h5')
    #model_path = base_pretrained_model + '.pickl'
    print('Model Saving Dir:\t', model_path)
    # model_path_file = open(model_path, 'wb')
    # pk.dump(model,model_path_file)
    model.save(model_path)
    # model_path_file.close()
    print('Model created and trained successfully. Saved as/in:\t', model_path)
    msg.showinfo('Great Job!','Model created and trained successfully. Saved as/in:\t' + model_path)
    #   savemat(model_path, model) wrte a function to save the model and weight as .mat


def save_model(model, json_path, weight_path):
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)


# def load_model(json_path):
#     model = model_from_json(open(json_path).read())
#     return model



def testing():
    print('Testing Trained Model...')
    classes_labels_number = number_of_classes_labels()
    if classes_labels_number is None:
        return
    class_labels = classes_labels_number[0]
    number_classes = classes_labels_number[1]
    batch_video_size = TryInt(batch_video_size_variable.get(), 'Batch size ')
    if batch_video_size is None:
        batch_video_size = batch_video_size_default
    number_video_per_class = int(batch_video_size/number_classes)
    base_pretrained_model = model_variable.get()
    model_path = os.path.join(save_model_dir, base_pretrained_model + '.h5')
    print('Model Loading Dir:\t', model_path)
    #model_path = base_pretrained_model + '.pickl'
    # model_path_file = open(model_path, 'rb')
    # model = pk.load(model_path_file)
    # model_path_file.close()
    model = load_model(model_path, custom_objects={'objective_function': objective_function})
    #   model = sloadmat(model_path)     #   sio.loadmat can onlyb loadtheweight I needto change this code to load the model
    testing_data, targets, number_frame_per_video = prepare_video_frames_for_training(sub_directory_name[1])   #   0 for traininf folder, 1 for testing folder
    prediction = model.predict_on_batch(testing_data)
    prediction = np.array(prediction)
    print('Prediction"\n', prediction)
    start_frame_index = -1
    end_frame_index = 0
    each_class_video_count = 0
    class_score_list = []
    predicted_class_list = []
    actual_class_list = []
    number_frame_per_video_list = []
    for index in range(0, len(number_frame_per_video)):     # obtain cumulative sum of frames
        number_frame_per_video_list.append(sum(number_frame_per_video[0:index + 1]))
    number_frame_per_video_list = [0] + number_frame_per_video_list  # the [0] is for the initial count range
    print('number_frame_per_video_list:\t', number_frame_per_video_list)
    for true_class_index in range(0, number_classes):
        for ii_ii in range(0,number_video_per_class):   # and item_index < int((true_class_index + 1) * number_video_per_class):
            actual_class_list.append(class_labels[true_class_index])
    print('actual_class_list:\n', actual_class_list)
    for item_index in range(0, len(number_frame_per_video_list) - 1):
        start_frame_index = number_frame_per_video_list[item_index]
        end_frame_index = number_frame_per_video_list[item_index + 1]
        video_score_list = []
        for class_index in range(0, number_classes):
            all_scores = prediction[:, class_index]
            print(class_index, ':\tall_scores 0k1:\n', all_scores)
            all_scores = np.reshape(all_scores, -1)
            print(class_index, ':\tall_scores:\n', all_scores)
            required_scores = all_scores[start_frame_index : end_frame_index]
            print(class_index, ':\trequired_scores:\n', required_scores)
            max_index = np.argmax(required_scores)
            max_score = required_scores[max_index]
            others_mean = np.mean(np.delete(required_scores, max_index)) * test_weighted_mean_factor
            video_score_list.append(max_score + others_mean)
        #class_score_list.append(video_score_list)
        predicted_class_list.append(class_labels[np.argmax(video_score_list)])
    print('actual_class_list:\n', actual_class_list)
    print('predicted_class_list:\n', predicted_class_list)
    actual_class_list = np.array(actual_class_list)
    predicted_class_list = np.array(predicted_class_list)
    accurracy = metrics.accuracy_score(actual_class_list, predicted_class_list)
    precision = metrics.precision_score(actual_class_list, predicted_class_list, average='micro')
    recal_score = metrics.recall_score(actual_class_list, predicted_class_list, average='micro')
    print('Model:\t', base_pretrained_model, '\tCurrancy:\t', round(accurracy, 2), '\tPrecision:\t', round(precision, 2), '\tRecall Score:\t', round(recal_score, 2))
    conf_mat = confusion_matrix(actual_class_list, predicted_class_list, labels=class_labels)
    print('conf_mat:\n', conf_mat)
    conf_mat_df = pd.DataFrame(conf_mat, index=class_labels, columns=class_labels)
    plt.figure(figsize=(5,5))
    sns.heatmap(conf_mat_df, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()


def prepare_video_frames_for_training(train_test, test_num_video=None):
    classes_labels_number = number_of_classes_labels()
    if classes_labels_number is None:
        return
    class_labels = classes_labels_number[0]
    number_classes = classes_labels_number[1]
    if test_num_video is None:
        batch_video_size = TryInt(batch_video_size_variable.get(), 'Batch size ')
        if batch_video_size is None:
            batch_video_size = batch_video_size_default
        number_video_per_class = int(batch_video_size/number_classes)
    else:
        number_video_per_class = test_num_video
    target_list = []
    frames_list = []
    frames_per_video_list = []
    parent_directory = directry_path.get()
    if parent_directory:
        if UpperStrip(parent_directory):
            pass
        else:
            parent_directory = current_dir
    else:
        parent_directory = current_dir
    print('Parent Directory:\t', parent_directory)
    sub_parent_diretories = extract_names_of_sub_directories(parent_directory)
    print('Sub-Parent DIrectories:\t', sub_parent_diretories)
    required_dir = ''
    for item in sub_parent_diretories:
        if UpperStrip(item) == UpperStrip(train_test):
            required_dir = item
            break
    classes_folder = os.path.join(parent_directory, required_dir)
    class_folder_list = extract_names_of_sub_directories(classes_folder)
    print('Classes Folders:\t', class_folder_list)
    for class_index in range(0, len(class_labels)): # folder in class_folder_list:
        class_item = class_labels[class_index]
        class_list = [0] * number_classes
        for folder in class_folder_list:    #class_item in class_labels:
            if UpperStrip(class_item) == UpperStrip(folder):
                class_list[class_index] = 1
                print('folder for class:\t', class_index, '\tis\t', folder)
                actual_path = os.path.join(classes_folder, folder)
                break   # this can be removed to take care of repeated names folder
        if sum(class_list) <= 0:
            msg.showerror('Error', 'The Name of this Folder does not correspond to any class label:\t' + folder)
            return []
        elif sum(class_list) > 1:
            msg.showerror('Error', 'The Name of this Folder correspond to more than one class label:\t' + folder)
            return []
        else:
            frames, number_frames = extract_videos(actual_path, number_video_per_class)
            frames_list = frames_list + frames # this concatination not appending to avoid a lis within a lis
            frames_per_video_list = frames_per_video_list + number_frames
            for i in range(0, int(sum(number_frames))):
                target_list.append(class_list)      # this is appending as we have 5 classes to be converted to N by Num+class array
    frames_list = [np.array(frames_list_2) for frames_list_2 in frames_list]
    frames_list = np.array([frame.reshape((frame.shape[1], frame.shape[0], frame.shape[2])) for frame in frames_list])
    #target_list = np.array(target_list) # [np.array(target_list_2) for target_list_2 in target_list]
    frames_list = np.stack(frames_list, axis=0) #   image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    target_list = np.stack(target_list, axis=0)
    print('Image shape\n', frames_list.shape, '\nDimention\n', frames_list.ndim, '\nSize\n', frames_list.size )
    print('Frames:\n', frames_list, '\n Tager\n', target_list, '\nFrames per video\n', frames_per_video_list)
    return frames_list, target_list, frames_per_video_list


def extract_videos(folder_director, number_videos=None):
    print('Extracting Videos ...')
    video_frames_list = []
    number_frames_per_video_list = []
    video_names_list = extract_names_files_in_dir(folder_director)
    if number_videos is None:
        number_videos = len(video_names_list)
    elif number_videos > len(video_names_list):
        number_videos = len(video_names_list)
    random_video_index = np.random.permutation(number_videos)
    all_video_random_index = np.random.permutation(len(video_names_list))
    for index in random_video_index:
        desired_video_index = all_video_random_index[index]
        print('Extracting Frames From:\t', video_names_list[desired_video_index])
        video_path = os.path.join(folder_director, video_names_list[desired_video_index])
        frames, num_frames = extract_video_frames(video_path)
        video_frames_list = video_frames_list + frames  #  frames is list so concatinate  video_frames_list.append(frames)
        number_frames_per_video_list.append(num_frames) # num_frames is a scalar int so append
    return video_frames_list, number_frames_per_video_list


def extract_names_files_in_dir(directory_folder):
    file_list = []
    for files_only in os.listdir(directory_folder):
        # check if current path is a file
        if os.path.isfile(os.path.join(directory_folder, files_only)):
            file_list.append(files_only)
    return file_list


def extract_names_of_sub_directories(directory_path):
    sub_directory_list = []
    for file in os.listdir(directory_path):
        d = os.path.join(directory_path, file)
        if os.path.isdir(d):
            sub_directory_list.append(file)
    return sub_directory_list


def extract_video_frames(video_path):
    image_width = TryInt(image_size_width.get(), 'Image width ')
    if image_width is None:
        image_width = image_width_default
    image_hight = TryInt(image_size_higth.get(), 'Image hiegth ')
    if image_hight is None:
        image_hight = image_hight_default
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)   ##    get(cv2.CAP_PROP_FPS)
    frames = []
    number_frames = 0
    while (video.isOpened()):   # will True can be used here
        ret, frame = video.read()
        if ret == True:
            frame = cv2.resize(frame, (image_width, image_hight))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            number_frames = number_frames + 1
            if number_frames == max_num_frames_per_video:   #   This code is just to limit the number of frames for laptop training ssake
                break
        else:
            break
    video.release()
    return frames, number_frames



def objective_function(out_true, output_predict):
    classes_labels_number = number_of_classes_labels()
    if classes_labels_number is None:
        return
    class_labels = classes_labels_number[0]
    number_classes = classes_labels_number[1]
    batch_video_size = TryInt(batch_video_size_variable.get(), 'Batch size ')
    if batch_video_size is None:
        batch_video_size = batch_video_size_default
    number_video_per_class = int(batch_video_size/number_classes)
    out_true = K.reshape(out_true, [-1])
    output_predict = K.reshape(output_predict, [-1])
    out_true = K.reshape(out_true, (-1,number_classes))
    output_predict = K.reshape(output_predict, (-1, number_classes))
    #frame_count_file_path = frame_count_dir + frames_count_list_file + '.pickl'
    print('shape re Out:\n', K.shape(out_true), '\nshape pre Out:\n', K.shape(output_predict))
    print('True\n', out_true, '\nPredicted:\n', output_predict, '\n')
    frame_count_file = open(frames_count_list_file, 'rb')
    number_frame_per_video = pk.load(frame_count_file)
    frame_count_file.close()
    print('Pickle frames saved\n')
    number_frame_per_video_list = []
    for index in range(0, len(number_frame_per_video)):     # obtain cumulative sum of frames
        number_frame_per_video_list.append(sum(number_frame_per_video[0:index + 1]))
    number_frame_per_video_list = [0] + number_frame_per_video_list  # the [0] is for the initial count range
    final_objective_function_list = []
    z_scores_list = []
    for class_index in range(0, number_classes):
        max_scores_true_class_list = []
        max_scores_other_class_list = []
        z_scores_list = []
        temporal_constrains_list = []
        sparsity_constrains_list = []
        for class_video_index in range(0, int(number_video_per_class * number_classes)):
            video_predictions = output_predict[-1, class_index]
            video_predictions = K.reshape(video_predictions, [-1])
            video_predictions = video_predictions[number_frame_per_video_list[class_video_index] : number_frame_per_video_list[class_video_index + 1]]
            if class_index == 0:
                if class_video_index < number_video_per_class:
                    max_scores_true_class_list.append(K.max(video_predictions))
                    temporal_constrains_list.append(K.sum(K.pow(video_predictions[1:] - video_predictions[:-1], 2)))
                    sparsity_constrains_list.append(K.sum(video_predictions))
                else:
                    max_scores_other_class_list.append(K.max(video_predictions))
            else:   # elif class_index == 1:
                if class_video_index >= number_video_per_class * class_index and class_video_index < number_video_per_class * (class_index + 1):
                    max_scores_true_class_list.append(K.max(video_predictions))
                    temporal_constrains_list.append(K.sum(K.pow(video_predictions[1:] - video_predictions[:-1], 2)))
                    sparsity_constrains_list.append(K.sum(video_predictions))
                else:
                    max_scores_other_class_list.append(K.max(video_predictions))
            # elif class_index == 2:
            #     if class_video_index >= number_video_per_class * 2 and class_video_index < number_video_per_class * 3:
            #         max_scores_true_class_list.append(K.max(video_predictions))
            #         temporal_constrains_list.append(K.sum(K.pow(video_predictions[1:] - video_predictions[:-1], 2)))
            #         sparsity_constrains_list.append(K.sum(video_predictions))
            #     else:
            #         max_scores_other_class_list.append(K.max(video_predictions))
            # elif class_index == 3:
            #     if class_video_index >= number_video_per_class * 3 and class_video_index < number_video_per_class * 4:
            #         max_scores_true_class_list.append(K.max(video_predictions))
            #         temporal_constrains_list.append(K.sum(K.pow(video_predictions[1:] - video_predictions[:-1], 2)))
            #         sparsity_constrains_list.append(K.sum(video_predictions))
            #     else:
            #         max_scores_other_class_list.append(K.max(video_predictions))
            # else:
            #     if class_video_index >= number_video_per_class * 4 and class_video_index < number_video_per_class * 5:
            #         max_scores_true_class_list.append(K.max(video_predictions))
            #         temporal_constrains_list.append(K.sum(K.pow(video_predictions[1:] - video_predictions[:-1], 2)))
            #         sparsity_constrains_list.append(K.sum(video_predictions))
            #     else:
            #         max_scores_other_class_list.append(K.max(video_predictions))
        max_scores_true = K.stack(max_scores_true_class_list)
        max_scores_others = K.stack(max_scores_other_class_list)
        temporal_constrains = K.stack(temporal_constrains_list)
        sparsity_constrains = K.stack(sparsity_constrains_list)
        max_z = []
        i=0
        for item_index in range(0, len(max_scores_others)):
            item = max_scores_others[item_index]
            i = i + 1
            max_z = K.maximum(1 - max_scores_true + item, 0)
            if i <= 10:
                print(i, '\tmax_z\n', max_z, '\nitem\n', item)
            z_scores_list.append(K.sum(max_z))
        z_scores = K.stack(z_scores_list)
        z = K.mean(z_scores)
        final_objective_function_list.append(z + lambda_1 * K.sum(temporal_constrains) + lambda_2 * K.sum(sparsity_constrains))
    print('Objective Fitness Function:\t', K.mean(K.stack(final_objective_function_list)))
    return K.mean(K.stack((final_objective_function_list)))



#   create_model()
click_button = ttk.Button(win, text='OK', command=action_function)
click_button.grid(column=1, sticky='w', row=100, padx=2,pady=4)



win.mainloop()