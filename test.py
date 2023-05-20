# 导入numpy验证docker是否根据requirements文件安装好了依赖
import numpy as np
import os
if __name__ == '__main__':
    # 在logs目录下创建个名为docker.log的空文件，验证容器和宿主机是否能进行数据同步
    print("This is a test: hello docker")
