import os
csv_path='log.csv'

file_name = 'train_ok'
files = [f for f in os.listdir() if '.jpeg' in f]
for i in range(len(files)):
    print(files[i])
    if '.jpeg' in files[i]:
        im_name = f'{file_name}_{i}.jpeg'
        os.rename(files[i], im_name)
    
        with open(csv_path,'a') as f:
            f.write(f'{im_name}, ok\n')
