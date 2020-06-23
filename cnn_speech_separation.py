import librosa
import numpy as np
import keras

def waveread():
    for x in range(265): 
        y, sr = librosa.load('male/M2_'+str(x+1)+'.wav', sr=None) 
        y = (np.array(y)).reshape(1, len(y)) 
        if x == 0: 
            malesignal = y 
        else: 
            malesignal = np.c_[malesignal, y] 
        y, sr = librosa.load('female/F2_'+str(x+1)+'.wav', sr=None) 
        y = (np.array(y)).reshape(1, len(y)) 
        if x == 0: 
            femalesignal = np.array(y) 
        else: 
            femalesignal = np.c_[femalesignal, y] 
return malesignal, femalesignal

def wavwrite():
    nchannels = 1 
    sampwidth = 2 
    outwave = wave.open('mixedsignal.wav', 'wb') 
    data_size = min(malesignal.shape[1], femalesignal.shape[1]) 
    framerate = sr 
    nframes = data_size 
    comptype = "NONE" 
    compname = "not compressed" 
    for v in range(data_size): 
            outwave.writeframes(struct.pack('h', int((malesignal[0][v] + femalesignal[0][v]) * 64000 / 4))) 
    outwave.close() 

def enframe(inputsignal, nw): 
    """
    Hamming Frame
    
    Inputs:
        - inputsignal: signal with np format
        - nw: num of windows
    
    Returns:
        - restuctsignal: desc1
    """
    signal_length = inputsignal.shape[1] 
    inc = int(nw/2) 
    nf = int(np.floor((1.0*signal_length-nw+inc)/inc)) 
    indf = np.array(range(0, nf*inc, inc)) 
    indf = indf.reshape(nf, 1) 
    inds = np.array(range(0, nw, 1)) 
    inds = inds.reshape(1, nw) 
    temp1 = np.tile(indf, (1, nw)) 
    temp2 = np.tile(inds, (nf, 1)) 
    temp3 = temp1+temp2 
    frame = np.zeros((temp3.shape[0], temp3.shape[1]), dtype=float) 
    for i in range(temp3.shape[0]): 
        for j in range(temp3.shape[1]): 
            frame[i][j] = inputsignal[0, temp3[i][j]] 
    hamming1 = signal.hamming(nw) 
    hamming1 = hamming1.reshape(1, nw) 
    restuctsignal = frame*hamming1 
return restuctsignal 
 
def overlappadd(inputsignal, nw): 
    """
    Canceling the hamming frame
    
    Inputs:
        - inputsignal: restucsignal
        - nw: num of windows
    
    Returns:
        - name1: desc1
    """
    len_r = len(inputsignal) 
    len_l = nw 
    hamming1 = signal.hamming(nw) 
    hamming1 = hamming1.reshape(1, nw) 
    restuctsignal = inputsignal /hamming1 
    inc = int(nw/2) 
    nf = len_r*inc-inc+nw 
    indf = np.array(range(0, len_r*inc, inc)) 
    indf = indf.reshape(len_r, 1) 
    inds = np.array(range(0, nw, 1)) 
    inds = inds.reshape(1, nw) 
    temp1 = np.tile(indf, (1, nw)) 
    temp2 = np.tile(inds, (len_r, 1)) 
    temp3 = temp1+temp2 
    frame = np.zeros((nf, 1), dtype=float) 
    for i in range(len_r): 
        for j in range(len_l): 
            frame[temp3[i][j]] = restuctsignal[i][j] 
return frame.T 

def SNR(testinputsignal1,testconstructsignal1):
    """
    Signal to Signal Noise Rate
    
    Inputs:
        - testinputsignal1: desc1
        - testconstructsignal1: desc2
    
    Returns:
        - SNR: Signal Noise Rate
    """
    meantestinputsignal = np.mean(testinputsignal1) 
    testinputsignal1 = testinputsignal1 - meantestinputsignal 
    testinputmax = np.max(abs(testinputsignal1)) 
    testsnrinputsignal = testinputsignal1/testinputmax 
    meantestconstructsignal = np.mean(testconstructsignal1) 
    testconstructsignal1 = testconstructsignal1 - meantestconstructsignal 
    testconstructmax = np.max(abs(testconstructsignal1)) 
    testsnrconstructsignal = testconstructsignal1/testconstructmax 
    error = testsnrinputsignal - testsnrconstructsignal 
    temp1 = np.power(testsnrinputsignal, 2) 
    temp2 = np.power(error, 2) 
    temp3 = np.sum(temp1) 
    temp4 = np.sum(temp2) 
    SNR = 10 * math.log(temp3 / temp4) 
return SNR


def main():
    # read wav from male/female folder
    malesignal, femalesignal = waveread()
    
    # split the dataset into train test by axis 1, and reshape them into one line
    # then add hamming frame to these signals
    trainmaleframesignal = (np.array(malesignal[0][0:3840128])).reshape(1, -1) 
    trainmaleframesignal1 = enframe(trainmaleframesignal, 256) 
    trainfemaleframesignal = (np.array(femalesignal[0][0:3840128])).reshape(1, -1) 
    trainfemaleframesignal1 = enframe(trainfemaleframesignal, 256) 
    testmaleframesignal = (np.array(malesignal[0][3840129:4147457])).reshape(1, -1) 
    testmaleframesignal1 = enframe(testmaleframesignal, 256) 
    testfemaleframesignal = (np.array(femalesignal[0][3840129:4147457])).reshape(1, -1) 
    testfemaleframesignal1 = enframe(testfemaleframesignal, 256) 


# 接着，我们分别对训练部分的信号和测试部分的信号进行傅里叶变换，读取傅里叶变换之后的幅值和相位以及幅值的最大值，保存它们以便后面进行训练，如下面代码所示：
    
    
    for x in range(30000): 
        temp = trainmaleframesignal1[x] 
        tempfft = np.fft.fft(temp) 
        tempphase = np.angle(tempfft) 
        trainphasemale.append(tempphase) 
        tempmax = abs(tempfft) 
        tempmax = np.max(tempmax) 
        trainmaxmale.append(tempmax) 
        temp = abs(tempfft) / tempmax 
        trainmale.append(temp) 
        temp2 = trainmaleframesignal1[x] + trainfemaleframesignal1[x] 
        temp2fft = np.fft.fft(temp2) 
        temp2phase = np.angle(temp2fft) 
        temp2max = abs(temp2fft) 
        temp2max = np.max(temp2max) 
        temp = abs(temp2fft) / temp2max 
    trainmixed.append(temp2) 
    
    
    model.add(Dense(1024, activation='sigmoid', input_dim=256)) 
    model.add(Dense(1024, activation='sigmoid')) 
    model.add(Dense(256, activation='sigmoid'))  
    ADAM  =  keras. optimizers. Adam(lr = 0.001, beta_1 = 0.9, beta_2= 0.999, epsilon = None, decay = 0.0, amsgrad = False) 
    model. compile(optimizer = ADAM, loss = 'mean_squared_error', metrics= ['accuracy'])  
    model.fit(X_train, Y_train, epochs=200, batch_size=256, verbose=2) 

    # 其中最重要的即加粗的部分，就是算法的核心部分。具体参数的选择可以参考 keras 的中文文档里有详细的说明。 
    # 训练完成之后我们将要测试的数据进行预测： 
    Y_pred = model.predict(X_test, batch_size=256, verbose=0, steps=None) 
    
    # 然后，我们将预测的数据进行傅里叶反变换，然后用前面已经保存好的幅值的最大值和相位求解，代码如下： 
    testtempinputsignal = testmale[x] * testmaxmale[x] 
        testtempinputphase = testphasemale[x] 
        testtempinputphase = testtempinputphase.reshape(1, 256) 
        temp = 1j * testtempinputphase[:] 
        inputz = testtempinputsignal * np.power(math.e, temp) 
        inputy = (np.fft.ifft(inputz)).real 
        inputy = inputy.reshape(1, 256) 
        if x == 0: 
            testinputsignal = np.array(inputy) 
        else: 
            testinputsignal = np.r_[testinputsignal, inputy] 
# 最后输出测试信号和计算信噪比即可。 

main()
