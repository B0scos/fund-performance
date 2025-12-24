from src.process.load_raw import ProcessRaw


raw = ProcessRaw()

pr = ProcessRaw()
df = pr.concat()
pr.save(df)

# print(raw.path_processed_path)