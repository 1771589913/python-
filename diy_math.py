import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xlrd
import xlwt
import sympy
from xlutils.copy import copy
from scipy import interpolate as ip
# from scipy.interpolate import make_interp_spline

# 全局变量
nsheet = 0
nfigure = 1
npoints = 300

# 统计计算
def single_var(file_path='table1.xls', data=[]):    # 单变量统计
    """单个变量的统计分析"""
    global nsheet
    if data != []:  # 未传入数据，准备从excel中获取
        n = len(data)   # 个数
        ave = np.mean(data)  # 平均值
        max_value = max(data)
        min_value = min(data)    # 极值
        variance = np.var(data)  # 方差
        standard = np.sqrt(variance)    # 标准差
        u_a = standard / np.sqrt(n-1)   # A类不确定度

        print("元素个数 = %d" % n)
        print("平均值 = %.7f" % ave)
        print("最大值 = %.4f，最小值 = %.4f，极差 = %.4f" %
              (max_value, min_value, max_value - min_value))
        print("方差 = %.7f，标准差 = %.7f" % (variance, standard))
        print("A类不确定度 = %.7lf" % u_a)
        print(' ')

        f = open("单变量统计.txt", "a")
        f.write("数据：")
        for data0 in data:
            f.write("%5.4f" % data0)
            f.write(', ')
        f.write('\n')
        f.write("元素个数 = %d\n" % n)
        f.write("平均值 = %.7f\n" % ave)
        f.write("最大值 = %.4f，最小值 = %.4f，极差 = %.4f\n" %\
            (max_value, min_value, max_value - min_value))
        f.write("方差 = %.7f，标准差 = %.7f\n" % (variance, standard))
        f.write("A类不确定度 = %.7lf\n\n" % u_a)

        return

    wb = xlrd.open_workbook(file_path)
    table = wb.sheet_by_index(nsheet)
    nsheet += 1  # 如果需要的话，为打开第2张表做准备
    nrows = table.nrows
    ncols = table.ncols

    n = []
    ave = []
    max_value = []
    min_value = []
    extre_dif = []
    variance = []
    standard = []
    u_a = []
    for i in range(nrows):
        data = table.row_values(i)  # 读取表的第i行
        print(data)

        n.append(len(data))   # 个数
        ave.append(np.mean(data))  # 平均值
        max_value.append(max(data))
        min_value.append(min(data))    # 极值
        extre_dif.append(max(data)-min(data))
        variance.append(np.var(data))  # 方差
        standard.append(np.sqrt(variance[i]))    # 标准差
        u_a.append(standard[i] / np.sqrt(n[i]-1))   # A类不确定度

        print("元素个数 = %d" % n[i])
        print("平均值 = %.7f" % ave[i])
        print("最大值 = %.4f，最小值 = %.4f，极差 = %.4f" %
              (max_value[i], min_value[i], extre_dif[i]))
        print("方差 = %.7f，标准差 = %.7f" % (variance[i], standard[i]))
        print("A类不确定度 = %.7lf" % u_a[i])
        print(' ')

        f = open("单变量统计.txt", "a")
        f.write("数据：")
        for data0 in data:
            f.write("%5.4f" % data0)
            f.write(', ')
        f.write('\n')
        f.write("元素个数 = %d\n" % n[i])
        f.write("平均值 = %.7f\n" % ave[i])
        f.write("最大值 = %.4f，最小值 = %.4f，极差 = %.4f\n" %\
            (max_value[i], min_value[i], extre_dif[i]))
        f.write("方差 = %.7f，标准差 = %.7f\n" % (variance[i], standard[i]))
        f.write("A类不确定度 = %.7lf\n\n" % u_a[i])

    # 将数据写入excel
    # excel = copy(wb=wb)  # 完成xlrd对象向xlwt对象转换
    # excel_table = excel.get_sheet(nsheet-1)    # 获得要操作的页
    # for i in range(nrows):
    #     excel_table.write(i, ncols, n[i])
    #     excel_table.write(i, ncols+1, ave[i])
    #     excel_table.write(i, ncols+2, max_value[i])
    #     excel_table.write(i, ncols+3, min_value[i])
    #     excel_table.write(i, ncols+4, extre_dif[i])
    #     excel_table.write(i, ncols+5, variance[i])
    #     excel_table.write(i, ncols+6, standard[i])
    #     excel_table.write(i, ncols+7, u_a[i])
    # excel_table.write(nrows, ncols, '元素个数')
    # excel_table.write(nrows, ncols+1, '平均值')
    # excel_table.write(nrows, ncols+2, '最大值')
    # excel_table.write(nrows, ncols+3, '最小值')
    # excel_table.write(nrows, ncols+4, '极差')
    # excel_table.write(nrows, ncols+5, '方差')
    # excel_table.write(nrows, ncols+6, '标准差')
    # excel_table.write(nrows, ncols+7, 'A类不确定度')
    # excel.save(file_path)


def double_var(file_path='table1.xls', x=[], y=[],\
    xlabel='x', ylabel='y', title='y - x', textx=0, texty=0):   # 回归计算
    """双变量的回归计算"""
    global nsheet
    global nfigure

    if x == [] and y == []:
        wb = xlrd.open_workbook(file_path)  # 读入数据
        table = wb.sheet_by_index(nsheet)
        nsheet += 1
        x = table.row_values(0)  # 第一行
        y = table.row_values(1)  # 第2行
    x = np.array(x)         # 转化为np.array数组
    y = np.array(y)         # 便于进行矢量操作
    n = len(x)              # 元素组数

    # 计算统计值
    x_ave = np.mean(x)
    y_ave = np.mean(y)  # 平均值
    xy = x * y          # 求线性回归系数做准备
    x2 = x**2
    xy_ave = np.mean(xy)
    x2_ave = np.mean(x2)
    # 以下开始求回归系数
    a = (xy_ave - x_ave*y_ave) / (x2_ave - x_ave**2)
    b = y_ave - a * x_ave   # 求得线性回归系数
    # 以下求相关系数 r
    tmp1 = np.sum((x-x_ave) * (y-y_ave))
    tmp2 = np.sum((x-x_ave)**2)
    tmp3 = np.sum((y-y_ave)**2)
    r = tmp1 / np.sqrt(tmp2*tmp3)   # 求得相关系数
    text1 = 'y = ' + str(a) + 'x + ' + str(b) + '\n'
    text2 = 'r = ' + str(r)  # 作为说明插入图像中

    s_y = np.sqrt(np.sum((y-b-a*x)**2) / (n-2))
    ua_a = s_y / np.sqrt(np.sum((x-x_ave)**2))
    ua_b = ua_a * np.sqrt(np.sum(x**2) / n)     # 求得两参数的a类不确定度

    # 打印结果
    print("平均值: mean(x) = %.5f, mean(y) = %.5f" % (x_ave, y_ave))
    print("y = ax + b: ")
    print("a = %.4f, b = %.4f, r = %.9f" % (a, b, r))
    print("参数a、b的A类标准不确定度为: ")
    print("s_y = %.5f" % s_y)
    print("u(a) = %.5f, u(b) = %.5f" % (ua_a, ua_b))
    print(' ')
    # 写入文件
    f = open("回归计算.txt", "a")
    f.write("平均值: mean(x) = %.5f,  mean(y) = %.5f\n" % (x_ave, y_ave))
    f.write("y = ax + b: \n")
    f.write("a = %.4f,  b = %.4f,  r = %.9f\n" % (a, b, r))
    f.write("参数a、b的A类标准不确定度为: \n")
    f.write("s_y = %.5f\n" % s_y)
    f.write("u(a) = %.5f,  u(b) = %.5f\n\n" % (ua_a, ua_b))

    # 画图
    y_r = a * x + b  # 对应的回归直线
    plt.figure(nfigure)
    nfigure += 1    # 如果需要的话，准备画下一幅图
    plt.scatter(x, y)   # 原始数据散点图
    plt.plot(x, y_r, 'r-')  # 回归直线
    plt.grid()  # 生成网格
    # 将直线方程打印在图像中
    if textx == 0:
        textx = x[0]
    if texty == 0:
        if max(y) > 0:
            texty = 0.9*max(y)
        else:
            texty = 1.2*max(y)
    plt.text(textx, texty, text1+text2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()


def plot_broken_line(file_path='table1.xls', x=[], y=[],\
        xlabel='x', ylabel='y', title='y - x', textx=0, texty=0):   # 画折线图
    """画出两个变量的折线图"""
    global nsheet
    global nfigure
    if x != [] and y != []:
        x = np.array(x)         # 转化为np.array数组
        y = np.array(y)         # 便于进行矢量操作
        n = len(x)              # 元素组数

        plt.figure(nfigure)
        nfigure += 1    # 如果需要的话，准备画下一幅图
        plt.plot(x, y)
        plt.scatter(x, y)
        plt.grid()  # 生成网格
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.show()

    wb = xlrd.open_workbook(file_path)  # 读入数据
    table = wb.sheet_by_index(nsheet)
    nsheet += 1
    nrows = table.nrows
    x = table.row_values(0)  # 第一行
    yn = []
    for i in range(1, nrows):
        yn.append(table.row_values(i))
    x = np.array(x)         # 转化为np.array数组
    for i in range(nrows-1):
        yn[i] = np.array(yn[i])         # 便于进行矢量操作
    n = len(x)              # 数据点个数

    plt.figure(nfigure)
    nfigure += 1    # 如果需要的话，准备画下一幅图
    for i in range(nrows-1):
        plt.plot(x, yn[i])
        plt.scatter(x, yn[i])
    plt.grid()  # 生成网格
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()


def plot_smooth_line(file_path='table1.xls', x=[], y=[],\
    xlabel='x', ylabel='y', title='y - x', kind='cubic', textx=0, texty=0): # 画曲线图
    """画出两个变量的曲线图"""
    global nsheet
    global nfigure
    if x != [] and y != []:
        x = np.array(x)         # 转化为np.array数组
        y = np.array(y)         # 便于进行矢量操作
        n = len(x)              # 元素组数

        # 平滑化处理
        xnew = np.linspace(x.min(), x.max(), 300)   # 300代表拟合的点的数量
        func = ip.interp1d(x, y, kind=kind)  # 生成 ynew
        ynew = func(xnew)
        # ynew = make_interp_spline(x, y)(xnew)

        y_max = max(ynew)
        y_min = min(ynew)
        max_indx = np.argmax(ynew)
        min_indx = np.argmin(ynew)
        print("极值：")
        print("max(y) = %.6f, min(y) = %.6f\n" % (y_max, y_min))
        f.open("曲线作图.txt", "a")
        f.write("极值：\n")
        f.write("max(y) = %.6f,  min(y) = %.6f\n\n" % (y_max, y_min))
        text = "max = " + str(y_max) + "\nmin = " + str(y_min)

        plt.figure(nfigure)
        nfigure += 1    # 如果需要的话，准备画下一幅图
        plt.scatter(x, y)
        plt.plot(xnew, ynew)
        # 原始数据散点图
        plt.scatter([xnew[max_indx], xnew[min_indx]], [y_max, y_min])
        plt.grid()  # 生成网格
        if textx == 0:
            textx = x[0]
        if texty == 0:
            if max(y) > 0:
                texty = 0.9*max(y)
            else:
                texty = 1.2*max(y)
        plt.text(textx, texty, text)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.show()

    wb = xlrd.open_workbook(file_path)  # 读入数据
    table = wb.sheet_by_index(nsheet)
    nsheet += 1
    nrows = table.nrows
    x = table.row_values(0)  # 第一行
    yn = []
    for i in range(1, nrows):
        yn.append(table.row_values(i))
    x = np.array(x)         # 转化为np.array数组
    for i in range(nrows-1):
        yn[i] = np.array(yn[i])         # 便于进行矢量操作
    n = len(x)              # 数据点个数

    # 平滑化处理
    xnew = np.linspace(x.min(), x.max(), 300)   # 300代表拟合的点的数量
    ynew = []
    for i in range(nrows-1):
        func = ip.interp1d(x, yn[i], kind='cubic')  # 生成 ynew
        ynew.append(func(xnew))
        # ynew.append(make_interp_spline(x, yn[i])(xnew))

    y_max = []; y_min = []
    for i in range(nrows-1):
        y_max.append(max(ynew[i]))
        y_min.append(min(ynew[i]))
    max_indx = []; min_indx = []
    for i in range(nrows-1):
        max_indx.append(np.argmax(ynew[i]))
        min_indx.append(np.argmin(ynew[i]))
    f = open("曲线作图.txt", "a")
    print("极值：")
    f.write("极值：\n")
    for i in range(nrows-1):
        print("max(y%d) = %.6f, min(y%d) = %.6f\n" % (i+1, y_max[i], i+1, y_min[i]))
        f.write("max(y%d) = %.6f, min(y%d) = %.6f\n" % (i+1, y_max[i], i+1, y_min[i]))
    f.write("\n")

    plt.figure(nfigure)
    nfigure += 1    # 如果需要的话，准备画下一幅图
    for i in range(nrows-1):
        plt.scatter(x, yn[i])
        plt.plot(xnew, ynew[i])
        plt.scatter([xnew[max_indx[i]], xnew[min_indx[i]]], [y_max[i], y_min[i]])
    plt.grid()  # 生成网格
    if nrows == 2:
        text = "max = " + str(y_max[0]) + "\nmin = " + str(y_min[0])
        if textx == 0:
            textx = xnew[0]
        if texty == 0:
            if max(ynew[0]) > 0:
                texty = 0.9*max(ynew[0])
            else:
                texty = 1.2*max(ynew[0])
        plt.text(textx, texty, text)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()


def batch_processing(fn, file_path='table1.xls'):   # 批量处理数据
    """对表中的数据批量使用函数f，并将结果写入excel文件"""
    global nsheet
    wb = xlrd.open_workbook(file_path)
    table = wb.sheet_by_index(nsheet)
    nsheet += 1
    ncols = table.ncols  # 待运算的元素组数
    nrows = table.nrows  # 一次运算包含的参数个数

    if callable(fn):
        fn = [fn]
    results = []
    for i in range(len(fn)):
        result = np.zeros(ncols)    # 将结果初始化为0
        for j in range(ncols):  # 对每一列都进行 f 函数运算
            data = table.col_values(j)
            result[j] = fn[i](data)  # 结果储存在 result 中
        results.append(result)
        f = open("批量处理.txt", "a")
        print("运算结果：")
        print(result)
        print(' ')
        f.write("运算结果：\n")
        for result0 in result:
            f.write("%.4f,  " % result0)
        f.write('\n\n')

    # 以下准备将运算结果追加到原来的表中
    excel = copy(wb=wb)  # 完成xlrd对象向xlwt对象转换
    excel_table = excel.get_sheet(nsheet-1)    # 获得要操作的页

    for i in range(len(fn)):
        for j in range(ncols):  # 在原始数据下一行写入计算结果
            excel_table.write(nrows+i, j, results[i][j])

    excel.save(file_path)   # 保存文件


def successional_difference(file_path='table1.xls', data=[]):   # 逐差法
    """逐差法处理数据"""
    global nsheet
    if data == []:  # 未传入数据，准备从excel中获取
        wb = xlrd.open_workbook(file_path)
        table = wb.sheet_by_index(nsheet)
        nsheet += 1
        data = table.row_values(0)

    data = np.array(data)
    n = len(data)
    dx = int(n / 2)
    x1 = data[:dx]      # 前dx项数据
    x2 = data[n-dx:n]   # 后dx项数据
    delta_x = np.sum(x2 - x1) / (dx**2)
    f = open("逐差法.txt", "a")
    print("逐差法求得：delta x = %.7f" % delta_x)
    f.write("逐差法求得：delta x = %.7f\n\n" % delta_x)
    print(' ')


# 画函数图像
def plot_func1(fn, x1=-1, x2=1, y1=-1, y2=1):   # 画一元函数图像
    global npoints
    global nfigure
    plt.figure(nfigure)
    nfigure += 1
    if callable(fn):
        fn = [fn]
    x = np.linspace(x1, x2, npoints)
    n = len(fn)
    yn = []
    for i in range(n):
        yn.append(fn[i](x))
        plt.plot(x, yn[i])
    plt.grid()


def plot_func2(fn, x1=-1, x2=1, y1=-1, y2=1):   # 画二元函数图像
    global npoints
    global nfigure
    fig = plt.figure(nfigure)
    ax = Axes3D(fig)
    nfigure += 1
    if callable(fn):
        fn = [fn]
    x = np.linspace(x1, x2, npoints)
    y = np.linspace(y1, y2, npoints)
    X, Y = np.meshgrid(x, y)
    n = len(fn)
    Zn = []
    for i in range(n):
        Z = fn[i](X, Y)
        Zn.append(Z)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.xlabel('x')
    plt.ylabel('y')
    

def plot_equa(fn, x1=-1, x2=1, y1=-1, y2=1):    # 画平面方程图像
    global npoints
    global nfigure
    plt.figure(nfigure)
    nfigure += 1
    if callable(fn):
        fn = [fn]
    # 设置x和y的坐标范围
    x = np.linspace(x1, x2, npoints)
    y = np.linspace(y1, y2, npoints)
    # 转化为网格
    x, y = np.meshgrid(x, y)
    n = len(fn)
    zn = []
    for i in range(n):
        zn.append(fn[i](x, y))
        plt.contour(x, y, zn[i], 0)
    plt.grid()


# 解方程
def equation_solving(f):            # 解一元方程
    x = sympy.symbols('x')
    a = sympy.solve([f(x)],[x])
    print(a, '\n')
    return a[x]


def binary_equation(f1, f2):        # 解二元方程
    x, y = sympy.symbols('x y')
    a = sympy.solve([f1(x, y), f2(x, y)], [x, y])
    print(a, '\n')
    return [a[x], a[y]]


def ternary_equation(f1, f2, f3):   # 解三元方程
    x, y, z = sympy.symbols('x y z')
    a = sympy.solve([f1(x, y, z), f2(x, y, z), f3(x, y, z)], [x, y, z])
    print(a, '\n')
    return [a[x], a[y], a[z]]


