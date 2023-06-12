import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from vars import Programpath
# 读取txt文件
fpath=Programpath+'res/mnist/ratio_0.5/mnist_acc.csv'





def plot_one_line(x,y,label,title,picpath):
    '''
    :param x: x轴标签列表
    :param y: x轴对应的数据y1
    :param label: y曲线的名称
    picpath:图片存放位置
    title:图表的名称
    :return:
    一条折线，x轴下表一致
    对于复式折线图，应该为每条折线添加图例，可以通过legend()函数来实现
    color  ------  指定折线的颜色
    linewidth   --------  指定折线的宽度
    linestyle   --------  指定折线的样式
    ‘  - ’ ： 表示实线
    ’ - - ‘   ：表示虚线
    ’ ：  ‘：表示点线
    ’ - . ‘  ：表示短线、点相间的虚线
    :return:
    '''
    plt.title(title)
    #设置字体
    #my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    #设置每一条曲线的样式，颜色，形状，宽度，图例信息
    ln1, = plt.plot(x, y, color='red', linewidth=2.0, linestyle='--')
    #plt.xticks(np.arange(0,1,0.2))
    plt.legend(handles=[ln1], labels=[label])
    #设置边框信息
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    print(picpath)
    plt.savefig(picpath, format='svg')
    plt.show()
def plot_listline(path,x,ylist,labellist,colist,title,value):
    '''
    :param x: x轴标签列表
    :param ylist: x轴对应的多条数据y
    :param labellist: x轴对应的多条数据y的曲线的名称
    :param title:图表的名称
    :return:
    多条折线，x轴下表一致
    对于复式折线图，应该为每条折线添加图例，可以通过legend()函数来实现
    color  ------  指定折线的颜色
    linewidth   --------  指定折线的宽度
    linestyle   --------  指定折线的样式
    ‘  - ’ ： 表示实线
    ’ - - ‘   ：表示虚线
    ’ ：  ‘：表示点线
    ’ - . ‘  ：表示短线、点相间的虚线
    :return:
    '''
    #plt.title(title)
    #设置字体
    #my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    #设置每一条曲线的样式，颜色，形状，宽度，图例信息
    lnlist=[]
    for i in range(len(ylist)):
        if len(x)!=len(ylist[i]):
            print("ERROR x and y length not equal")
        if i ==0:
            ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=2.0, linestyle='--')
        else:
            ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=3.0, linestyle='-.')
        lnlist.append(ln)
    # acc1=[0.9 for i in range(len(x))]
    # ln, = plt.plot(x, acc1, color='black', linewidth=1.0, linestyle='--')
    # lnlist.append(ln)
    #
    # acc2 = [0.8 for i in range(len(x))]
    # ln, = plt.plot(x, acc2, color='black', linewidth=1.0, linestyle='--')
    # lnlist.append(ln)

    # # 标记与ln1和ln2交点的横坐标位置
    # for i in range(len(x)):
    #     if ylist[0][i] == acc1[i]:
    #         plt.scatter(x[i], acc1[i], color='red')
    #     if ylist[0][i] == acc2[i]:
    #         plt.scatter(x[i], acc2[i], color='blue')
    # plt.yticks(np.arange(0.75,0.95,0.05))
    # plt.yticks(xyran)
    print("label is ",labellist)
    #labellist=['FedAvg','JSND','FedSGD','Center']
    plt.legend(handles=lnlist, labels=labellist)
    plt.xlabel("Rounds")
    plt.ylabel(value)
    #plt.ylim(0.1,0.88,0.01)
    #设置边框信息
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    #plt.savefig(path,format='svg')
    print('save',path)
    plt.savefig(path, format='eps')
    plt.show()






def plot(fpath,picpath,value,length=-1):
    '''
    绘制不同列的曲线图
    Args:
        value: 列的名称
        length: 制定列的长度
        names:所有列名

    Returns:

    '''
    # 读取txt文件，指定列名
    data = pd.read_csv(fpath, delimiter=',',
                       names=['1', 'case', 'method', 'group', 'Loss', 'Accuracy', 'AUC', 'Std Test Accuracy',
                              'Std Test AUC', 'join_ratio'])
    # methods= data['method'].unique()
    # 按照method分组
    title=value+" trend of different algorithms"
    labellist = []
    ylist = []
    # 按照method分组
    grouped_data = data.groupby('method')
    for name, group in grouped_data:
        if name != 'method' and name != 'FedAAW' and name != 'FedJS':
            smoothed_loss = group[value].rolling(window=5, min_periods=1).mean()
            labellist.append(name)
            if length==-1:
                ylist.append(smoothed_loss.tolist())
            else:
                ylist.append(smoothed_loss.tolist()[:length])
    x = [i for i in range(len(ylist[0]))]
    # colist=['red','yellow','blue','green','black','pink','orange']
    colist = ['red',  'blue', 'green', 'black', 'orange']
    plot_listline(picpath,x,ylist,labellist,colist,title,value)






# plot('Loss')
# plot('Loss',50)
# plot('Accuracy')
# plot(fpath,'Accuracy',50)
value='Accuracy'
length=50
fpath=Programpath+'res/mnist/fedda/mnist_acc.csv'
picpath=Programpath+'res/mnist/fedda/'+'mnist_'+value+str(length)+'.eps'
plot(fpath,picpath,'Accuracy',50)



