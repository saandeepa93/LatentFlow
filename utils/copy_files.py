import os 
import shutil
import json 


if __name__ == "__main__":
  with open("./data/train_ll.json", 'r') as fp:
    train_ll = json.load(fp)
  all_files = list(train_ll.keys())

  root_dir = "/data/dataset/raf_db/basic/Image/aligned"
  dest_dir = "./data/incorrect"

  for files in all_files:
    src_path = os.path.join(root_dir, f"{files}.jpg")
    fname = files.split('_')[1]
    dest_path = os.path.join(dest_dir, f"{fname}_{train_ll[files]['pred'][:4]}_{train_ll[files]['true'][:4]}.jpg") 
    shutil.copy(src_path, dest_path)

