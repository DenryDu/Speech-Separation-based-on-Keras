#*
#* Author: DenryDu 
#* Time: 2020/06/23 10:25:34
#* Description: used to convert filename from GRID Original format to M2_xxx.wav or F2_xxx.wav
#* 

import os
 
class BatchRename():
 
    def __init__(self, path, gender):
        """
        init function with path and gender set
        
        Inputs:
            - path: specify where the folder is to store wav files
            - gender: specify the gender and naming method
        """ 
        self.path = path
        assert(gender == "male" or gender == "female","Neither a male, nor a female, it's a alien?!")
        self.gender = gender

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        prefix = "M2_" if self.gender == "male" else "F2_"
        if not os.path.exists("./" + self.gender):
            os.mkdir("./" + self.gender)
        dst_path = "./"+self.gender
        i = 0
        for item in filelist:
            if item.endswith('.wav'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(dst_path), prefix+str(i+1)+ '.wav')
                try:
                    os.rename(src, dst)
                    i = i + 1
                except:
                    continue

    def prerename(self):
        """
        prerename -- to prevent loss of image caused by overwrite
        
        """
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), 'unnamedimg'+str(i+1)+ '.jpeg')
                try:
                    os.rename(src, dst)
                    i = i + 1
                except:
                    continue
            elif item.endswith('.jpeg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), 'unnamedimg'+str(i+1)+ '.jpeg')
                try:
                    os.rename(src, dst)
                    i = i + 1
                except:
                    continue
 
if __name__ == '__main__':
    male_obj = BatchRename('./data/s2',"male")
    male_obj.rename()
    female_obj = BatchRename('./data/s4',"female")
    female_obj.rename()
