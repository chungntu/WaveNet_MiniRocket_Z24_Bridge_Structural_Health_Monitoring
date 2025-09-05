import utils
# --------------------------------------------
# Vẽ Accuracy (train/val) trên cùng một hình & lưu PNG
#   - utils.find_latest_history_file("./History"): tìm file history *.txt mới nhất
#   - utils.load_history_dict(file): đọc dict {"accuracy":[...], "val_accuracy":[...], ...}
#   - utils.plot_accuracy_from_history(hist, save_path=...): vẽ & lưu hình
# --------------------------------------------
latest_hist_file = utils.find_latest_history_file("./History")
if latest_hist_file:
    print(f"[INFO] Đọc history từ: {latest_hist_file}")
    hist = utils.load_history_dict(latest_hist_file)
    # Lưu hình *_acc.png và đồng thời hiển thị trên màn hình (plt.show() nằm trong utils)
    utils.plot_accuracy_from_history(
        hist,
        save_path=latest_hist_file.replace(".txt", "_acc.png")
    )
