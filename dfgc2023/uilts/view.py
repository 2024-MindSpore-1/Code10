import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def view_pred_result(img_path_list, pred_list, test_csv, root_dir, output_path, phase='1'):
    
    test_video_list = pd.read_csv(test_csv, names=['file'])
    test_video_list = list(test_video_list['file'])
    
    score_dict = {}
    for video in test_video_list:
        score_dict[video] = []

    for i, image_path in tqdm(enumerate(img_path_list)):
        video_name = os.path.relpath(image_path, start=root_dir).replace('_frame', '')
        video_name = os.path.dirname(video_name)+'.mp4'
        if video_name in score_dict.keys():
            score_dict[video_name].append(pred_list[i])
        else:
            print(f'{video_name} not found in test video list!')

    
    out_file = os.path.join(output_path, 'Test'+phase+'_preds.txt')
    f = open(out_file, 'w+')
    for video in score_dict.keys():
        score = np.array(score_dict[video]).mean()
        f.write(video+','+str(score)+'\n')
    
    f.close()






if __name__=='__main__':
    root_dir = r'/hd2/sunxianyun/DFGC2023_code/2-achilles10/data'
    pred_file = r'/hd2/sunxianyun/DFGC2023_re/output.csv'
    test_csv = r'/hd2/sunxianyun/DFGC2023_code/2-achilles10/data/label/test_set1.txt'
    pred_df = pd.read_csv(pred_file)
    img_path_list = list(pred_df['img'])
    pred_list = list(pred_df['pred'])
    view_pred_result(img_path_list, pred_list, test_csv, root_dir, output_path='./')