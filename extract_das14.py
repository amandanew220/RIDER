import wandb
import pandas as pd



api = wandb.Api()
runs = api.runs("amandanew202-trinity-western-university/RIDER",
                order="-created_at")  # most recent first


# TEMPORARY: just check the first run before running the full loop
first_run = runs[0]
history = first_run.scan_history(keys=['Test/score_tm', 'Test/score_gddt', 'Test/score_rmsd'])
df = pd.DataFrame(history)
print(df.columns.tolist())
print(df.head())

results = []
count = 0
for run in runs:
    if count >= 28:
        break
    history = run.scan_history(keys=['Test/score_tm', 'Test/score_gddt', 'Test/score_rmsd'])
    df = pd.DataFrame(history)

    if df.empty:
        peak_tm = peak_gddt = min_rmsd = plateau_tm = collapse_pct = None
    else:
        peak_tm = df['Test/score_tm'].max()
        peak_gddt = df['Test/score_gddt'].max()
        min_rmsd = df['Test/score_rmsd'].min()
        plateau_tm = df['Test/score_tm'].tail(20).mean()

        # compute collapse_pct BEFORE the dict, not inside it
        if 'epoch' in df.columns:
            post_warmup = df[df['epoch'] > 50]
        else:
            post_warmup = df[df.index > 50]
        collapse_pct = (post_warmup['Test/score_tm'] < 0.1).mean() if not post_warmup.empty else None

    results.append({
        'run': run.name,
        'sample_index': run.config.get('sample_index'),
        'oracle': run.config.get('oracle'),
        'peak_tm': peak_tm,
        'peak_gddt': peak_gddt,
        'min_rmsd': min_rmsd,
        'plateau_tm': plateau_tm,
        'collapse_pct': collapse_pct,
    })
    count += 1

out = pd.DataFrame(results)
print(out.to_string())
out.to_csv('das14_results.csv', index=False)
