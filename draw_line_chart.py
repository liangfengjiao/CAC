import matplotlib.pyplot as plt


def draw_parameter_lambda():
    # 数据
    epsilon = [0.1, 0.3, 0.5, 0.7, 0.9]
    ods = [52.39, 57.32, 56.63, 60.43, 55.33]
    ois = [58.97, 63.45, 63.41, 64.50, 63.34]
    ap = [53.31, 59.55, 58.60, 63.65, 56.95]
    # One_F1 = [52.39, 58.97, ]
    # Two_F1 = [57.32, 63.45, 59.55]
    # Three_F1 = [56.63, 63.41, 58.60]
    # Four_F1 = [60.43, 64.50, 63.65]
    # Five_F1 = [55.33, 63.34, 56.95]

    # epsilon = [0.1, 0.25, 0.5, 0.75, 1]
    # I_AUROC = [96.2, 96.5, 96.6, 96.6, 96.6]
    # P_AUROC = [95.3, 95.3, 95.3, 95.3, 95.3]
    # 绘制折线图
    plt.figure(figsize=(5, 3))
    plt.plot(epsilon, ods, marker='o', label='ODS')
    plt.plot(epsilon, ois, marker='o', label='OIS')
    plt.plot(epsilon, ap, marker='^', label='ap')
    plt.xlabel('parameter $\lambda$')
    # plt.ylabel('AUROC')
    plt.ylim(30, 80)
    # 添加图例
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 1)), bbox_to_anchor=(1, 1)
    plt.legend(loc='lower right')
    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('./parameter-lambda.pdf', bbox_inches='tight')
    plt.show()

def draw_parameter_C():
    # 数据
    epsilon = [1, 3, 5, 7, 9]
    ods = [56.52, 57.04, 60.43, 54.21, 55.86]
    ois = [61.88, 63.13, 64.50, 63.23, 63.21]
    ap = [58.27, 59.26, 63.65, 55.49, 56.97]
    # One_F1 = [52.39, 58.97, ]
    # Two_F1 = [57.32, 63.45, 59.55]
    # Three_F1 = [56.63, 63.41, 58.60]
    # Four_F1 = [60.43, 64.50, 63.65]
    # Five_F1 = [55.33, 63.34, 56.95]

    # epsilon = [0.1, 0.25, 0.5, 0.75, 1]
    # I_AUROC = [96.2, 96.5, 96.6, 96.6, 96.6]
    # P_AUROC = [95.3, 95.3, 95.3, 95.3, 95.3]
    # 绘制折线图
    plt.figure(figsize=(5, 3))
    plt.plot(epsilon, ods, marker='o', label='ODS')
    plt.plot(epsilon, ois, marker='o', label='OIS')
    plt.plot(epsilon, ap, marker='^', label='AP')
    plt.xlabel('parameter $C$')
    # plt.ylabel('AUROC')
    plt.ylim(30, 80)
    # 添加图例
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 1)), bbox_to_anchor=(1, 1)
    plt.legend(loc='lower right')
    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('./parameter-C.pdf', bbox_inches='tight')
    plt.show()

draw_parameter_lambda()