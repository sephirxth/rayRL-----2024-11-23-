'''
将xml文件转换为csv文件
'''
import os
import subprocess

def convert_xml_to_csv(folder_path):
    file_count = 0  # 初始化计数器
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            xml_file = os.path.join(folder_path, filename)
            # 构建并执行命令
            command = f"python /home/jian/sumo/tools/xml/xml2csv.py {xml_file}"
            try:
                result = subprocess.run(command, shell=True, check=True)
                # print(f"Processed {filename}")
                file_count += 1  # 成功处理后递增计数器
                
                # 删除原始的xml文件
                # os.remove(xml_file)
                # print(f"Deleted {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process {filename}: {e}")
    
    print(f"一共转换的文件数量： {folder_path}: {file_count}")  # 输出总数

# 文件夹路径
folder_path = '../data/default/tripinfo'
convert_xml_to_csv(folder_path)
