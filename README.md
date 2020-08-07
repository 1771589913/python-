# 自己常用的数学工具

以下提到的所有功能均在diy_math.py中实现，导入该模块即可使用。

## 实验数据的统计计算
1. 单变量统计single_var()
  
  函数原型 single_var(file_path='table1.xls', data=[])
  
  file_path是excel文件目录，为简化操作，个人习惯将其与工作的.py文件放在同一文件夹下，并设置名称为'table1.xls'，即可直接调用single_val()完成数据处理。data为可选参数，如果以列表作为参数，即可对该列表进行回归运算。

  将一(多)个变量的多次测量值导入excel表中的第一行(前几行)，在python文件中导入模块，使用single_var()函数，运行代码即可自动计算这些变量的统计值，如平均数、方差、标准差、A类不确定度等。也可直接将一个列表作为参数传入函数中，运行代码也将计算它的统计值。
  
  excel中的格式要求：一个变量的多次测量量横向排列，前面不能有空隙，如有多个变量，按顺序向下排列，中间不留空隙。要求文件为.xls格式，不支持.xlsx格式

2. 线性回归计算double_val()

  函数原型 double_var(file_path='table1.xls', x=[], y=[], xlabel='x', ylabel='y', title='y - x', textx=0, texty=0)
  
  file_path、x、y均同上。xlabel、ylabel、title为横纵坐标轴的名称、图标标题，默认为x, y, y - x。坐标图中会打印回归方程的表达式，textx, texty即为它的坐标位置。

  将两个变量的测量值导入excel的前两行，使用double_var()函数，或者直接把两个列表作为参数传入double_val()函数，即可画出拟合的直线图像，并求出各系数及不确定度。

3. 画折线图plot_broken_line()

  函数原型 plot_broken_line(file_path='table1.xls', x=[], y=[], xlabel='x', ylabel='y', title='y - x', textx=0, texty=0)

  暂时先写到这。

4. 画曲线图plot_smooth_line()

  函数原型 plot_smooth_line(file_path='table1.xls', x=[], y=[], xlabel='x', ylabel='y', title='y - x', kind='cubic', textx=0, texty=0)
  
  kind为插值的方法，cubic为三次样条曲线，可选参数为'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'


5. 批量处理数据batch_processing()

  函数原型 batch_processing(fn, file_path='table1.xls')
  
  fn为函数列表，可同时对数据进行多次操作并写入excel，如果只有一个函数，可直接以函数作为参数

6. 逐差法处理数据successional_difference()

  (实在找不到合适的翻译作为函数名。。。。。)
  
  函数原型 successional_difference(file_path='table1.xls', data=[])

## 画图步骤的简化

  简化了画函数图像的步骤，输入待作图的函数，再把该函数作为参数，传入diy_math.py中的相关函数中即可实现作图。简化的代价是图像的可编辑性变弱。
  
1. 一元函数图像plot_func1()
  
  函数原型 plot_func1(fn, x1=-1, x2=1, y1=-1, y2=1)
  
  fn为函数列表，x1, x2, y1, y2为图中显示的坐标轴范围

2. 二元函数图像plot_func2()
  
  函数原型 plot_func2(fn, x1=-1, x2=1, y1=-1, y2=1)

3. 以方程确定的图像plot_equa()
  
  函数原型 plot_equa(fn, x1=-1, x2=1, y1=-1, y2=1)

## 解方程步骤的简化

  简化了解方程的步骤，将方程写为f(x)=0的形式，把f作为参数传入diy_math.py中的相关函数即可实现解方程。

1. 一元方程equation_solving()

  函数原型 equation_solving(f)

2. 二元方程组binary_equation()

  函数原型 binary_equation(f1, f2)

3. 三元方程组ternary_equation()

  函数原型 ternary_equation(f1, f2, f3)
