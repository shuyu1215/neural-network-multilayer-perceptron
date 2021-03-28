import tkinter as tk
from os import listdir,system
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import math

mypath = './data/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
file_name = ''
next_W = {}
def load_data():
    with open(file_name,'r') as f :
        load = []
        load.clear()
        for line in f.readlines():
            load.append(list(map(float,line.strip().split(' '))))
        np.random.shuffle(load)
        load = np.array(load)
    load_X = []
    exp = []
    load_X.clear()
    exp.clear()
    for arr in load:
        temp = arr[:-1]
        tempX = [-1.0]
        exp.append(arr[-1])
        for element in temp:
            tempX.append(element)
        load_X.append(tempX)
        del tempX
    dim = len(arr)-1
    class_val = count_class(exp)
    return load,load_X, exp, dim, class_val

def count_class(exp):
    array = []
    for a in exp:
        array.append(a)
    array.sort()
    count = 1
    lenth = len(array)
    for i in range(1,lenth):
        if (array[i] != array[i-1]):
            count += 1
    return count

def get_weight():
    return round(random.uniform(-1, 1), 2)

def init_weight(Dim):
    weight = [-1]
    for i in range(0,Dim):
        tmp = get_weight()
        weight.append(tmp)
    return weight

def inner_product(inner_W,inner_X):
    #print('inner_W:',inner_W)
    #print('inner_X:',inner_X)
    np1 = np.array(inner_W)
    np2 = np.array(inner_X)
    mul = np1*np2
    ans = 0
    for i in mul:
        ans += i
    #print('inner_product_ans:',ans)
    return ans

def sigmoid(result):
    #print('exp(-num):',math.exp(-result))
    out = 1/(1+math.exp(-result))
    #print('sigmoid_out:',out)
    return round(out,5)

def check_sgn(sgn,label_0,label_1):
    if sgn > 0.5:
        sgn = label_1
    else:
        sgn = label_0
    return sgn

def set_labels(exp_ans):
    label_min = min(exp_ans)
    label_max = max(exp_ans)
    return label_min, label_max

def hide_delta(hiddenY,output_delta,hide_W):
    #print(hiddenY,'* 1-',hiddenY,'*',output_delta,'*',hide_W)
    out = round((hiddenY*(1-hiddenY)*output_delta*hide_W),5)
    #print('hidden_delta:',out)
    return out
       
def out_delta(out_exp,hiddenZ):
    return round((out_exp-hiddenZ)*hiddenZ*(1-hiddenZ),5)

def update_weight(W,delta,input_X,learn_rate):
    np_W = np.array(W)
    np_X = np.array(input_X)
    update_W = np_W + learn_rate*delta*np_X
    update = []
    update.clear()
    for arr_element in update_W:
        update.append(arr_element)
    return update

def get_rmse(rmse_arr):
    ans = 0
    for i in rmse_arr:
        ans += i
    # print('ans:',ans)
    rmse_result = ans/len(rmse_arr)
    return np.sqrt(rmse_result)

def compute_rmse(predict,exp_value):
    tmp = (predict - exp_value)**2
    return tmp
def check_exp(new_W,i,exp_val,class_val,label_min,label_max):
    Y_vector = [-1]
    hidden_W = []
    output_W = []
    for j in range(len(new_W)-1):
        hidden_W.append(new_W[j])
    for k in hidden_W:
        inner_product_result = inner_product(k,i)
        Y_vector.append(sigmoid(inner_product_result))
    #print('new_W[-1]:',new_W[-1])
    #print('Y_vector:',Y_vector)
    inner_product_temp = inner_product(new_W[-1],Y_vector)
    #print('inner_product_temp:',inner_product_temp)
    Z = sigmoid(inner_product_temp)
    #print('output_Z:',Z)
    Z_sgn = check_sgn(Z,label_min,label_max)
    #print('Z_sgn:',Z_sgn)
    #print('exp_val:',exp_val)
    RMSE = compute_rmse(Z,exp_val)
    #print('rmse:',RMSE)
    '''
    add = 1/class_val
    num = 0
    class_num = 0
    count = 0
    print('add:',add)
    for l in range(0,class_val):
        num = num + add
        if Z <= num:
            Z_val = class_num
        else:
            class_num += 1
    '''
    if Z_sgn != exp_val:
        #print('False!')
        return False, RMSE
    else:
        #print('True')
        return True, RMSE
    
def accuracy(total,errors_num):
    return round(float((total - errors_num) / total) * 100, 3)

def predict(input_W,input_X):
    hidden_W = []
    output_W = []
    hidden_cells = []
    hidden_W.clear()
    output_W.clear()
    hidden_cells.clear()
    output_cells = 0
    hidden_cells.append(-1)
    hidden_W.append(input_W[0])
    hidden_W.append(input_W[1])
    
    output_W.append(input_W[2])

    for j in range(0,len(hidden_W)):
        total = 0.0
        total = inner_product(hidden_W[j],input_X)
        hidden_cells.append(sigmoid(total))
    
    for k in range(0,len(output_W)):
        total = 0.0
        total = inner_product(output_W[k],hidden_cells)
        output_cells = sigmoid(total)
    
    return output_cells, hidden_cells

def back_propagate(passed_W,data,exp,learn,class_val,label_min,label_max):
    hidden_deltas = []
    last_W = []
    hidden_deltas.clear()
    last_W.clear()
    output_cell, hidden_cells = predict(passed_W,data)
    output_deltas = out_delta(0.0,output_cell)
    
    for i in range(1,len(passed_W)):
        hidden_deltas.append(hide_delta(hidden_cells[i],output_deltas,passed_W[-1][i]))
        #print('hidden_deltas:',hidden_deltas)
    
    
    
    for j in range(0,len(passed_W)-1):
            last_W.append(update_weight(passed_W[j],hidden_deltas[j],data,learn))

    #print('hidden_cells:',hidden_cells)

    last_W.append(update_weight(passed_W[-1],output_deltas,hidden_cells,learn))
    check,rmse_ans = check_exp(last_W,data,exp,class_val,label_min,label_max)
    #print('check:',check)
    #print('last_W:',last_W)
    return rmse_ans,last_W

def training(train_W,data,exp_ans,class_val,Iteration,learningRate):
    learn = float(learningRate)
    rmse = []
    temp_W = []
    label_min,label_max = set_labels(exp_ans)
    for a in train_W:
        temp_W.append(a)
    for i in range(0,int(Iteration)):
        error = 0.0
        error_num = 0
        #print('--------------------------round[',i,']')
        for j in range(0,len(data)):
            rmse_tmp,next_W = back_propagate(temp_W,data[j],exp_ans[j],learn,class_val,label_min,label_max)
            rmse.append(rmse_tmp)
            temp_W.clear()
            for k in next_W:
                temp_W.append(k)
            #print('next temp_W:',temp_W)
            next_W.clear()
        rmse_result = get_rmse(rmse)
        #print('rmse:',rmse)
        rmse.clear()
        #print('rmse_result:', rmse_result)
        #print("error:",error)
        for m in range(0,len(data)):
            check,a = check_exp(temp_W,data[m],exp_ans[m],class_val,label_min,label_max)
            #print('check:',check)
            if check == False:
                error_num += 1
        if error_num == 0:
            break
    #print('-------------------------------')
    return temp_W,rmse_result,label_min,label_max

def testing(train_W,test_data,test_exp,class_val,label_min,label_max):
    errorNum = 0
    test_len = len(test_data)
    for i in range(0,test_len):
        check_val,a = check_exp(train_W,test_data[i],test_exp[i],class_val,label_min,label_max)
        if check_val == False:
            errorNum += 1
    acc_rate = accuracy(test_len, errorNum)
    #print('acc_rate:',acc_rate)
    return acc_rate
    
def run(learningRate,Iteration):
    W = []
    train_data = []
    test_data = []
    train_exp = []
    test_data = []
    test_exp = []
    original_data, data, exp_ans, dim, class_val = load_data()
    len_data = len(data)
    len_train = round(float(2*len_data/3))
    len_test = int(len_data-len_train)
    #print('data:',data)
    #print('exp:',exp_ans)
    for i in range(0,len_train):
        train_data.append(data[i])
        train_exp.append(exp_ans[i])
    for j in range(len_train,len_data):
        test_data.append(data[j])
        test_exp.append(exp_ans[j])
    #print('len_train',len_train)
    #print('train_data',train_data)
    #print('train_exp',train_exp)
    #print('test_data',test_data)
    #print('test_exp',test_exp)
    for i in range(0,dim+1):
        W.append(init_weight(dim))
    #W.append([-1.2,1,1])
    #W.append([0.3,1,1])
    #W.append([0.5,0.4,0.8])
    trained_W,rmseResult,label_min,label_max = training(W,train_data,exp_ans,class_val,Iteration,learningRate)
    accracy = testing(trained_W,test_data,test_exp,class_val,label_min,label_max)
    return original_data, exp_ans, trained_W, accracy,rmseResult

class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.windows = master
        self.grid()
        self.mypath = './data/'
        self.files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.create_windows()

    def get_list(self,event):
        global file_name
        self.index = self.listbox.curselection()[0]
        self.selected = self.listbox.get(self.index)
        file_name = 'data/'+self.selected

    def create_windows(self):
        self.windows.title("Homework2")

        self.listbox = tk.Listbox(windows, width=20, height=6)
        self.listbox.grid(row=0, column=0,columnspan=2,stick=tk.W+tk.E)

        self.yscroll = tk.Scrollbar(command=self.listbox.yview, orient=tk.VERTICAL)
        self.yscroll.grid(row=0, column=2, sticky=tk.W+tk.E)
        self.listbox.configure(yscrollcommand=self.yscroll.set)

        for item in self.files:
            self.listbox.insert(tk.END, item)

        self.listbox.bind('<ButtonRelease-1>', self.get_list)
        
        self.learning = tk.Label(windows, text="Learning rate:").grid(row=1,column=0, sticky=tk.W+tk.E)
        self.iteration = tk.Label(windows, text="Iteration:").grid(row=2,column=0, sticky=tk.W+tk.E)
        self.weight = tk.Label(windows, text="weight result:").grid(row=3,column=0, sticky=tk.W+tk.E)
        self.draw_iteration = tk.Label(windows, text="Accuracy rate:").grid(row=4,column=0, sticky=tk.W+tk.E)
        self.rmse = tk.Label(windows, text = "RMSE:").grid(row=5,column=0, sticky=tk.W+tk.E)
        self.e1 = tk.Entry(windows)
        self.e2 = tk.Entry(windows)
        self.e3 = tk.Entry(windows)
        self.e4 = tk.Entry(windows)
        self.e5 = tk.Entry(windows)
        self.e1.grid(row=1, column=1, sticky=tk.W+tk.E)
        self.e2.grid(row=2, column=1, sticky=tk.W+tk.E)
        self.e3.grid(row=3, column=1, sticky=tk.W+tk.E)
        self.e4.grid(row=4, column=1, sticky=tk.W+tk.E)
        self.e5.grid(row=5, column=1, sticky=tk.W+tk.E)
        self.e1.delete(0,'end')
        self.e2.delete(0,'end')
        self.e1.insert(10,0.01)
        self.e2.insert(10,10)
        self.quit = tk.Button(windows, text='Quit', command=windows.quit).grid(row=6, column=0, sticky=tk.W+tk.E)
        self.show = tk.Button(windows, text='Show', command=self.show_entry_fields).grid(row=6, column=1, sticky=tk.W+tk.E)
        
        self.result_figure = Figure(figsize=(5,4), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, self.windows)
        self.result_canvas.draw()
        self.result_canvas.get_tk_widget().grid(row=7, column=0, columnspan=3, sticky=tk.W+tk.E)
        
    def show_entry_fields(self):
        global learningRate
        global Iteration
        learningRate = self.e1.get()
        Iteration = self.e2.get()
        original_data, exp_ans,new_W,acc_rate,RMSE = run(learningRate,Iteration)
        self.plot_data(original_data,exp_ans,new_W,acc_rate,RMSE)        
    def plot_data(self,inputs,targets,weights,acc_rate,RMSE):
        self.result_figure.clf()
        self.result_figure.a = self.result_figure.add_subplot(111)
        target_0,target_1 = set_labels(targets)
        total = len(inputs)
        len_train = round(float(2*total/3))
        train_inputs = inputs[:len_train]
        train_targets = targets[:len_train]
        test_inputs = inputs[len_train:]
        test_targets = targets[len_train:]
        for input,target in zip(train_inputs,train_targets):
            self.result_figure.a.plot(input[0],input[1],'ro' if (target == target_0) else 'bo')
        for input,target in zip(test_inputs,test_targets):
            self.result_figure.a.plot(input[0],input[1],'gx' if (target == target_0) else 'gx')
    
        self.e3.delete(0,'end')
        self.e4.delete(0,'end')
        self.e5.delete(0,'end')
        self.e3.insert(10,str(weights))
        self.e4.insert(10,str(acc_rate))
        self.e5.insert(10,str(RMSE))
        self.result_figure.a.set_title('Result')
        self.result_canvas.draw()



if __name__ == "__main__":
    windows = tk.Tk()
    app = Application(windows)
    windows.mainloop()





