from utils import find_latest_history_file, load_history_dict, plot_accuracy_from_history

latest_hist_file = find_latest_history_file("./History")
if latest_hist_file:
    print(f"[INFO] Đọc history từ: {latest_hist_file}")
    hist = load_history_dict(latest_hist_file)
    plot_accuracy_from_history(hist, save_path=latest_hist_file.replace(".txt", "_acc.png"))

