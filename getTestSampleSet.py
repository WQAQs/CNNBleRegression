import dataProcess as dp
import globalConfig  #不能删除,因为在 globalConfig.py 中更改了当前路径

'''新代码中已配置好默认路径，只需修改globalConfig.py，若无特殊需求不要改此处变量'''
pointTxtRootDir = '.\\raw_data\\test'  # 原始数据存在的文件夹
pointCsvRootDir = '.\\points_csv\\test'  # 转换的csv文件目标文件夹，也作为合并csv的源路径
ibeaconFilePath = '.\\ibeacon_mac_count.csv'  # ibeacon统计文件目标路径及文件名
allPointCsvRootDir = pointCsvRootDir  # 总数据数据文件夹
# allPointCsvRootDir = '.\\newCodeDist\\points_csv\\testprogram'  # 总数据数据文件夹
# 样品数据集目标路径，可以设置为训练集或测试集
dp.sampleDataSetFilePath = '.\\test_set19_3s.csv'

dp.loadAllTxt2Csv(pointTxtRootDir, pointCsvRootDir)  # 将原始数据加载为Csv文件
dp.createSampleDataSet(allPointCsvRootDir, ibeaconFilePath)  # 创建测试集
