import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 指定固定配时 和 RL 下的输出文件
folders = {
    'Krauss': '../data/default/tripinfo',
}

file_pattern = '*-tripinfo-output.csv'

# 初始化存储结果的列表
all_data = []

for model, file_path in folders.items():
    all_files = glob.glob(os.path.join(file_path, file_pattern))
    
    for filename in all_files:
        # 首先检查文件是否为空
        if os.stat(filename).st_size > 0:
            try:
                df = pd.read_csv(filename, sep=';')
                # 判断DataFrame是否为空
                if not df.empty:
                    # 提取车流量（文件名中的数字部分）
                    traffic_flow = int(os.path.basename(filename).split('-')[0])
                    # 计算tripinfo_timeLoss的平均值
                    mean_time_loss = df['tripinfo_timeLoss'].mean()
                    # 添加到结果列表
                    all_data.append((traffic_flow, mean_time_loss, model))
            except pd.errors.EmptyDataError:
                print(f"Skipping invalid file: {filename}")
        else:
            print(f"Skipping empty file: {filename}")

# 确保all_data非空
if all_data:
    # 创建DataFrame
    results_df = pd.DataFrame(all_data, columns=['TrafficFlow', 'MeanTimeLoss', 'Model'])
    
    # 按照TrafficFlow对数据进行排序
    results_df = results_df.sort_values(by='TrafficFlow')
    
    # 使用Matplotlib绘制车流量与平均timeLoss的曲线图
    plt.figure(figsize=(10, 8))
    
    # 绘制每个模型的曲线
    for model in results_df['Model'].unique():
        model_df = results_df[results_df['Model'] == model]
        plt.plot(model_df['TrafficFlow'], model_df['MeanTimeLoss'], marker='o', label=model)
    
    # 调整标签和标题
    plt.xlabel('Traffic Flow')
    plt.ylabel('Mean Time Loss (s)')
    plt.title('Traffic Flow vs Mean Time Loss')
    plt.legend()
    
    # 创建保存图片的目录
    save_dir = '../data/pictures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'traffic_flow_vs_mean_time_loss.png')
    
    # 保存图表
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    
    # 显示图表
    plt.show()
else:
    print("No valid data found.")