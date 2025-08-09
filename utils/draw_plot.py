import matplotlib.pyplot as plt
import numpy as np


def read_data(models, t):
    average_dice = {}
    group_data = {}
    for model in models:
        if model == "DCR_UNet":
            data_path = f"../logs/dc_resunet/Metric_{t}.txt"
        else:
            data_path = f"../logs/unet/Metric_{t}.txt"

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = f.read()
        data_list = raw_data.split("-----------------------------------")[1:]
        data_list[-1] += "\n"
        average_data = []
        model_data = []
        for i_fold in data_list:
            i_fold_data = list(map(float, i_fold.split("\n")[2:-1]))
            average_data.append(sum(i_fold_data) / len(i_fold_data))
            model_data.append(i_fold_data)
        average_dice[model] = average_data
        group_data[model] = model_data
    return group_data, average_dice


if __name__ == "__main__":
    # 生成数据：2个模型 x 5组数据
    np.random.seed(42)
    models = ["DCR_UNet", "U-Net"]
    t = "dice"
    group_data, average_dice = read_data(models, t)
    for key, value in average_dice.items():
        print(f"{key}：{value}，average {sum(value) / len(value)}")
        if key == "U-Net":
            path = f"../logs/unet/average_{t}.txt"
        else:
            path = f"../logs/dc_resunet/average_{t}.txt"

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{key}：{value}，average {sum(value) / len(value)}")

    # 设置箱线图位置（并列显示）
    positions = np.arange(5)  # 0,1,2,3,4
    width = 0.25  # 每个箱体的宽度

    # 绘制箱线图
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models):
        # 计算每个模型箱体的偏移位置
        offset = width * 0
        box = plt.boxplot(
            group_data[model],
            positions=positions + offset,
            widths=width,
            patch_artist=True,
            showmeans=True  # 显示均值线
        )
        # 自定义颜色
        color = "lightblue" if model == "DCR_UNet" else "orange"
        for patch in box['boxes']:
            patch.set_facecolor(color)
        # 添加图例代理
        plt.plot([], color=color, label=f"{model}")

    # 美化图形
    plt.xticks(positions + width / 2, [f"Fold {i + 1}" for i in range(5)])
    plt.title("Model Performance Comparison (Dice Score)")
    plt.xlabel("Cross-validation Fold")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # 防止标签重叠
    plt.savefig("./box.png")
