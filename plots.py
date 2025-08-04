from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

# Load all .csv under chess_logs/
df = load_results("monitor/")
x, y = ts2xy(df, "timesteps")
plot_results(["monitor_dir/"], 1_000_000, plotter_kwargs={"xlabel":"Steps","ylabel":"Reward"})